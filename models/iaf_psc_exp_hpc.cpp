/*
 *  iaf_psc_exp_hpc.cpp
 *
 *  This file is part of NEST.
 *
 *  Copyright (C) 2004 The NEST Initiative
 *
 *  NEST is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  NEST is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with NEST.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include "iaf_psc_exp_hpc.h"

// C++ includes:
#include <algorithm>
#include <cstdint>
#include <limits>
#include <utility>

// Includes from libnestutil:
#include "dict_util.h"
#include "iaf_propagator.h"
#include "numerics.h"

// Includes from nestkernel:
#include "exceptions.h"
#include "kernel_manager.h"
#include "nest_impl.h"
#include "universal_data_logger_impl.h"


namespace nest
{
/* ----------------------------------------------------------------
 * Block of neurons
 * ---------------------------------------------------------------- */

/**
 * Block of the iaf_psc_exp_hpc neurons of one Create() call on one thread, with their state in arrays.
 *
 * update() loops over the steps outside and over the neurons inside; per neuron it performs the operations of
 * iaf_psc_exp in the same order, so that the results stay identical to those of iaf_psc_exp, and the compiler can
 * vectorize the inner loop. The input buffer is ordered by slot, then neuron, then channel, so that the inputs of all
 * neurons for one step are contiguous; the slots of one neuron are therefore number of neurons times number of
 * channels apart (see DirectSpikeInputAddr).
 *
 * The neurons keep their parameters. The values the update reads are copied into the block: I_e, threshold and reset
 * by set_status() of the neuron, because iaf_psc_exp reads them directly in every step; the propagators and the
 * refractory counts by pre_run_hook(), because iaf_psc_exp computes them there.
 */
class IafPscExpHpcBlock : public UpdateBlock
{
public:
  //! Form the block of nodes; their state, their parameters and their frozen flag are copied into the block.
  explicit IafPscExpHpcBlock( const std::vector< Node* >& nodes );

  void update( const Time& origin, const long from, const long to ) override;

private:
  friend class iaf_psc_exp_hpc;

  using Buffers_ = iaf_psc_exp_hpc::Buffers_;
  static constexpr size_t num_channels_ = Buffers_::NUM_INPUT_CHANNELS;

  //! Size the input buffer to the current number of slots, if needed, and set the input of neuron i to zero.
  void init_input_( const size_t i );

  //! Add w to channel ch of neuron i in the given slot.
  void add_input_( const size_t i, const size_t slot, const size_t ch, const double w );

  //! List the neurons that are not frozen and have a recording device (after a change of their flags).
  void list_logged_();

  std::vector< iaf_psc_exp_hpc* > neurons_;

  // State
  std::vector< double > V_m_;
  std::vector< double > i_syn_ex_;
  std::vector< double > i_syn_in_;
  std::vector< double > i_0_;
  std::vector< int > r_ref_;

  // Copies of parameters, written by set_status() of the neuron
  std::vector< double > I_e_;
  std::vector< double > Theta_;
  std::vector< double > V_reset_;
  std::vector< char > frozen_;

  // Values derived from the parameters, written by pre_run_hook() of the neuron
  std::vector< double > P20_;
  std::vector< double > P11ex_;
  std::vector< double > P11in_;
  std::vector< double > P21ex_;
  std::vector< double > P21in_;
  std::vector< double > P22_;
  std::vector< int > refractory_counts_;
  std::vector< char > has_logger_;

  std::vector< size_t > logged_;  //!< neurons recorded after each step, see list_logged_()
  bool logged_changed_ = true;    //!< whether frozen_ or has_logger_ changed since list_logged_()

  size_t num_slots_ = 0;         //!< slots of the ring buffer
  std::vector< double > input_;  //!< input buffer: slot, then neuron, then channel

  std::vector< std::pair< size_t, long > > spikes_;  //!< neuron and step of the spikes of one call of update()
};

IafPscExpHpcBlock::IafPscExpHpcBlock( const std::vector< Node* >& nodes )
{
  const size_t n = nodes.size();
  for ( auto* v : { &V_m_,
          &i_syn_ex_,
          &i_syn_in_,
          &i_0_,
          &I_e_,
          &Theta_,
          &V_reset_,
          &P20_,
          &P11ex_,
          &P11in_,
          &P21ex_,
          &P21in_,
          &P22_ } )
  {
    v->resize( n );
  }
  r_ref_.resize( n );
  refractory_counts_.resize( n );
  frozen_.resize( n );
  has_logger_.resize( n );

  neurons_.reserve( n );
  for ( size_t i = 0; i < n; ++i )
  {
    auto* neuron = static_cast< iaf_psc_exp_hpc* >( nodes[ i ] );
    assert( neuron->block_ == nullptr );
    neurons_.push_back( neuron );
    neuron->block_ = this;
    neuron->block_index_ = i;
    neuron->store_in_block_( neuron->S_ );
  }
}

void
IafPscExpHpcBlock::init_input_( const size_t i )
{
  const size_t num_slots = kernel().connection_manager.get_min_delay() + kernel().connection_manager.get_max_delay();
  if ( num_slots_ != num_slots )
  {
    num_slots_ = num_slots;
    input_.resize( num_slots_ * neurons_.size() * num_channels_ );
  }
  for ( size_t slot = 0; slot < num_slots_; ++slot )
  {
    std::fill_n( input_.begin() + ( slot * neurons_.size() + i ) * num_channels_, num_channels_, 0.0 );
  }
}

void
IafPscExpHpcBlock::add_input_( const size_t i, const size_t slot, const size_t ch, const double w )
{
  assert( slot < num_slots_ );
  input_[ ( slot * neurons_.size() + i ) * num_channels_ + ch ] += w;
}

void
IafPscExpHpcBlock::list_logged_()
{
  logged_.clear();
  for ( size_t i = 0; i < neurons_.size(); ++i )
  {
    if ( has_logger_[ i ] and not frozen_[ i ] )
    {
      logged_.push_back( i );
    }
  }
  logged_changed_ = false;
}

void
IafPscExpHpcBlock::update( const Time& origin, const long from, const long to )
{
  if ( logged_changed_ )
  {
    list_logged_();
  }

  const size_t n = neurons_.size();
  double* __restrict V_m = V_m_.data();
  double* __restrict i_syn_ex = i_syn_ex_.data();
  double* __restrict i_syn_in = i_syn_in_.data();
  double* __restrict i_0 = i_0_.data();
  int* __restrict r_ref = r_ref_.data();
  const double* __restrict I_e = I_e_.data();
  const double* __restrict Theta = Theta_.data();
  const double* __restrict V_reset = V_reset_.data();
  const char* __restrict frozen = frozen_.data();
  const double* __restrict P20 = P20_.data();
  const double* __restrict P11ex = P11ex_.data();
  const double* __restrict P11in = P11in_.data();
  const double* __restrict P21ex = P21ex_.data();
  const double* __restrict P21in = P21in_.data();
  const double* __restrict P22 = P22_.data();
  const int* __restrict refractory_counts = refractory_counts_.data();

  for ( long lag = from; lag < to; ++lag )
  {
    double* __restrict input = input_.data() + kernel().event_delivery_manager.get_modulo( lag ) * n * num_channels_;

    // The operations of iaf_psc_exp for one neuron and one step, in the same order.
    for ( size_t i = 0; i < n; ++i )
    {
      if ( frozen[ i ] )
      {
        continue;
      }
      if ( r_ref[ i ] == 0 )  // neuron not refractory, so evolve V
      {
        V_m[ i ] = V_m[ i ] * P22[ i ] + i_syn_ex[ i ] * P21ex[ i ] + i_syn_in[ i ] * P21in[ i ]
          + ( I_e[ i ] + i_0[ i ] ) * P20[ i ];
      }
      else
      {
        --r_ref[ i ];
      }

      // exponential decaying PSCs
      i_syn_ex[ i ] *= P11ex[ i ];
      i_syn_in[ i ] *= P11in[ i ];

      // the spikes arriving at T+1 have an immediate effect on the state of the neuron
      double* __restrict in = input + i * num_channels_;
      i_syn_ex[ i ] += in[ Buffers_::SYN_EX ];
      i_syn_in[ i ] += in[ Buffers_::SYN_IN ];

      if ( V_m[ i ] >= Theta[ i ] )  // threshold crossing
      {
        r_ref[ i ] = refractory_counts[ i ];
        V_m[ i ] = V_reset[ i ];
        spikes_.emplace_back( i, lag );
      }

      // set new input current and reset the slot
      i_0[ i ] = in[ Buffers_::I0 ];
      in[ Buffers_::SYN_IN ] = 0.0;
      in[ Buffers_::SYN_EX ] = 0.0;
      in[ Buffers_::I0 ] = 0.0;
    }

    for ( const size_t i : logged_ )
    {
      neurons_[ i ]->B_.logger_.record_data( origin.get_steps() + lag );
    }
  }

  // Send in the order of iaf_psc_exp, by neuron, then by step: the order of the spikes in the spike register decides
  // the order in which the targets sum their input, and with it the rounding.
  std::stable_sort( spikes_.begin(), spikes_.end(), []( const auto& a, const auto& b ) { return a.first < b.first; } );
  for ( const auto& [ i, lag ] : spikes_ )
  {
    neurons_[ i ]->set_spiketime( Time::step( origin.get_steps() + lag + 1 ) );
    SpikeEvent se;
    kernel().event_delivery_manager.send( *neurons_[ i ], se, lag );
  }
  spikes_.clear();
}

/* ----------------------------------------------------------------
 * Recordables map
 * ---------------------------------------------------------------- */

RecordablesMap< iaf_psc_exp_hpc > iaf_psc_exp_hpc::recordablesMap_;

void
register_iaf_psc_exp_hpc( const std::string& name )
{
  register_node_model< iaf_psc_exp_hpc >( name );
}

// Override the create() method with one call to RecordablesMap::insert_()
// for each quantity to be recorded.
template <>
void
RecordablesMap< iaf_psc_exp_hpc >::create()
{
  // use standard names wherever you can for consistency!
  insert_( names::V_m, &iaf_psc_exp_hpc::get_V_m_ );
  insert_( names::I_syn_ex, &iaf_psc_exp_hpc::get_I_syn_ex_ );
  insert_( names::I_syn_in, &iaf_psc_exp_hpc::get_I_syn_in_ );
}

/* ----------------------------------------------------------------
 * Default constructors defining default parameters and state
 * ---------------------------------------------------------------- */

iaf_psc_exp_hpc::Parameters_::Parameters_()
  : Tau_( 10.0 )              // in ms
  , C_( 250.0 )               // in pF
  , t_ref_( 2.0 )             // in ms
  , E_L_( -70.0 )             // in mV
  , I_e_( 0.0 )               // in pA
  , Theta_( -55.0 - E_L_ )    // relative E_L_
  , V_reset_( -70.0 - E_L_ )  // in mV
  , tau_ex_( 2.0 )            // in ms
  , tau_in_( 2.0 )            // in ms
{
}

iaf_psc_exp_hpc::State_::State_()
  : i_0_( 0.0 )
  , i_syn_ex_( 0.0 )
  , i_syn_in_( 0.0 )
  , V_m_( 0.0 )
  , r_ref_( 0 )
{
}

/* ----------------------------------------------------------------
 * Parameter and state extractions and manipulation functions
 * ---------------------------------------------------------------- */

void
iaf_psc_exp_hpc::Parameters_::get( Dictionary& d ) const
{
  d[ names::E_L ] = E_L_;  // resting potential
  d[ names::I_e ] = I_e_;
  d[ names::V_th ] = Theta_ + E_L_;  // threshold value
  d[ names::V_reset ] = V_reset_ + E_L_;
  d[ names::C_m ] = C_;
  d[ names::tau_m ] = Tau_;
  d[ names::tau_syn_ex ] = tau_ex_;
  d[ names::tau_syn_in ] = tau_in_;
  d[ names::t_ref ] = t_ref_;
}

double
iaf_psc_exp_hpc::Parameters_::set( const Dictionary& d, Node* node )
{
  // if E_L_ is changed, we need to adjust all variables defined relative to
  // E_L_
  const double ELold = E_L_;
  update_value_param( d, names::E_L, E_L_, node );
  const double delta_EL = E_L_ - ELold;

  if ( update_value_param( d, names::V_reset, V_reset_, node ) )
  {
    V_reset_ -= E_L_;
  }
  else
  {
    V_reset_ -= delta_EL;
  }

  if ( update_value_param( d, names::V_th, Theta_, node ) )
  {
    Theta_ -= E_L_;
  }
  else
  {
    Theta_ -= delta_EL;
  }

  update_value_param( d, names::I_e, I_e_, node );
  update_value_param( d, names::C_m, C_, node );
  update_value_param( d, names::tau_m, Tau_, node );
  update_value_param( d, names::tau_syn_ex, tau_ex_, node );
  update_value_param( d, names::tau_syn_in, tau_in_, node );
  update_value_param( d, names::t_ref, t_ref_, node );
  if ( V_reset_ >= Theta_ )
  {
    throw BadProperty( "Reset potential must be smaller than threshold." );
  }
  if ( C_ <= 0 )
  {
    throw BadProperty( "Capacitance must be strictly positive." );
  }
  if ( Tau_ <= 0 or tau_ex_ <= 0 or tau_in_ <= 0 )
  {
    throw BadProperty( "Membrane and synapse time constants must be strictly positive." );
  }
  if ( t_ref_ < 0 )
  {
    throw BadProperty( "Refractory time must not be negative." );
  }

  return delta_EL;
}

void
iaf_psc_exp_hpc::State_::get( Dictionary& d, const Parameters_& p ) const
{
  d[ names::V_m ] = V_m_ + p.E_L_;  // Membrane potential
}

void
iaf_psc_exp_hpc::State_::set( const Dictionary& d, const Parameters_& p, double delta_EL, Node* node )
{
  if ( update_value_param( d, names::V_m, V_m_, node ) )
  {
    V_m_ -= p.E_L_;
  }
  else
  {
    V_m_ -= delta_EL;
  }
}

iaf_psc_exp_hpc::Buffers_::Buffers_( iaf_psc_exp_hpc& n )
  : logger_( n )
{
}

iaf_psc_exp_hpc::Buffers_::Buffers_( const Buffers_&, iaf_psc_exp_hpc& n )
  : logger_( n )
{
}

/* ----------------------------------------------------------------
 * Default and copy constructor for node
 * ---------------------------------------------------------------- */

iaf_psc_exp_hpc::iaf_psc_exp_hpc()
  : ArchivingNode()
  , P_()
  , S_()
  , B_( *this )
{
  recordablesMap_.create();
}

iaf_psc_exp_hpc::iaf_psc_exp_hpc( const iaf_psc_exp_hpc& n )
  : ArchivingNode( n )
  , P_( n.P_ )
  , S_( n.S_ )
  , B_( n.B_, *this )
{
}

/* ----------------------------------------------------------------
 * State in the block
 * ---------------------------------------------------------------- */

std::unique_ptr< UpdateBlock >
iaf_psc_exp_hpc::make_update_block( const std::vector< Node* >& nodes )
{
  return std::make_unique< IafPscExpHpcBlock >( nodes );
}

UpdateBlock*
iaf_psc_exp_hpc::get_update_block() const
{
  return block_;
}

iaf_psc_exp_hpc::State_
iaf_psc_exp_hpc::get_state_() const
{
  if ( not block_ )
  {
    return S_;
  }
  const auto& b = *block_;
  const size_t i = block_index_;
  State_ s;
  s.i_0_ = b.i_0_[ i ];
  s.i_syn_ex_ = b.i_syn_ex_[ i ];
  s.i_syn_in_ = b.i_syn_in_[ i ];
  s.V_m_ = b.V_m_[ i ];
  s.r_ref_ = b.r_ref_[ i ];
  return s;
}

void
iaf_psc_exp_hpc::store_in_block_( const State_& s )
{
  assert( block_ );
  auto& b = *block_;
  const size_t i = block_index_;
  b.i_0_[ i ] = s.i_0_;
  b.i_syn_ex_[ i ] = s.i_syn_ex_;
  b.i_syn_in_[ i ] = s.i_syn_in_;
  b.V_m_[ i ] = s.V_m_;
  b.r_ref_[ i ] = s.r_ref_;
  b.I_e_[ i ] = P_.I_e_;
  b.Theta_[ i ] = P_.Theta_;
  b.V_reset_[ i ] = P_.V_reset_;
  b.frozen_[ i ] = is_frozen();
  b.logged_changed_ = true;
}

double
iaf_psc_exp_hpc::get_V_m_() const
{
  return block_->V_m_[ block_index_ ] + P_.E_L_;
}

double
iaf_psc_exp_hpc::get_I_syn_ex_() const
{
  return block_->i_syn_ex_[ block_index_ ];
}

double
iaf_psc_exp_hpc::get_I_syn_in_() const
{
  return block_->i_syn_in_[ block_index_ ];
}

void
iaf_psc_exp_hpc::get_status( Dictionary& d ) const
{
  P_.get( d );
  get_state_().get( d, P_ );
  ArchivingNode::get_status( d );

  d[ names::recordables ] = recordablesMap_.get_list();
}

void
iaf_psc_exp_hpc::set_status( const Dictionary& d )
{
  Parameters_ ptmp = P_;                        // temporary copy in case of errors
  const double delta_EL = ptmp.set( d, this );  // throws if BadProperty
  State_ stmp = get_state_();                   // temporary copy in case of errors
  stmp.set( d, ptmp, delta_EL, this );          // throws if BadProperty

  // We now know that (ptmp, stmp) are consistent. We do not
  // write them back to (P_, S_) before we are also sure that
  // the properties to be set in the parent class are internally
  // consistent.
  ArchivingNode::set_status( d );

  // if we get here, temporaries contain consistent set of properties
  P_ = ptmp;
  if ( block_ )
  {
    store_in_block_( stmp );
  }
  else
  {
    S_ = stmp;
  }
}

/* ----------------------------------------------------------------
 * Node initialization functions
 * ---------------------------------------------------------------- */

void
iaf_psc_exp_hpc::init_buffers_()
{
  block_->init_input_( block_index_ );
  B_.logger_.reset();
  ArchivingNode::clear_history();
}

void
iaf_psc_exp_hpc::pre_run_hook()
{
  // ensures initialization in case mm connected after Simulate
  B_.logger_.init();

  const double h = Time::get_resolution().get_ms();
  auto& b = *block_;
  const size_t i = block_index_;

  // these P are independent
  b.P11ex_[ i ] = std::exp( -h / P_.tau_ex_ );
  b.P11in_[ i ] = std::exp( -h / P_.tau_in_ );

  b.P22_[ i ] = std::exp( -h / P_.Tau_ );

  // these are determined according to a numeric stability criterion
  b.P21ex_[ i ] = IAFPropagatorExp( P_.tau_ex_, P_.Tau_, P_.C_ ).evaluate( h );
  b.P21in_[ i ] = IAFPropagatorExp( P_.tau_in_, P_.Tau_, P_.C_ ).evaluate( h );

  b.P20_[ i ] = P_.Tau_ / P_.C_ * ( 1.0 - b.P22_[ i ] );

  // t_ref_ specifies the length of the absolute refractory period as a double in ms. The model can only handle
  // refractory periods that are integer multiples of the computation step size (h); the conversion is carried out via
  // nest::Time, as in iaf_psc_exp.
  b.refractory_counts_[ i ] = Time( Time::ms( P_.t_ref_ ) ).get_steps();
  // since t_ref_ >= 0, this can only fail in error
  assert( b.refractory_counts_[ i ] >= 0 );

  b.has_logger_[ i ] = B_.logger_.has_loggers();
  b.frozen_[ i ] = is_frozen();
  b.logged_changed_ = true;
}

void
iaf_psc_exp_hpc::update( const Time&, const long, const long )
{
  assert( false and "neurons of iaf_psc_exp_hpc are updated by their block (IafPscExpHpcBlock)" );
}

void
iaf_psc_exp_hpc::handle( SpikeEvent& e )
{
  assert( e.get_delay_steps() > 0 );

  const size_t input_buffer_slot = kernel().event_delivery_manager.get_modulo(
    e.get_rel_delivery_steps( kernel().simulation_manager.get_slice_origin() ) );

  const double s = e.get_weight() * e.get_multiplicity();

  // separate buffer channels for excitatory and inhibitory inputs
  block_->add_input_( block_index_, input_buffer_slot, s > 0 ? Buffers_::SYN_EX : Buffers_::SYN_IN, s );
}

void
iaf_psc_exp_hpc::handle( CurrentEvent& e )
{
  assert( e.get_delay_steps() > 0 );

  const double c = e.get_current();
  const double w = e.get_weight();

  const size_t input_buffer_slot = kernel().event_delivery_manager.get_modulo(
    e.get_rel_delivery_steps( kernel().simulation_manager.get_slice_origin() ) );

  block_->add_input_( block_index_, input_buffer_slot, Buffers_::I0, w * c );
}

void
iaf_psc_exp_hpc::handle( DataLoggingRequest& e )
{
  B_.logger_.handle( e );
}

DirectSpikeInputAddr
iaf_psc_exp_hpc::get_direct_spike_input()
{
  // handle( SpikeEvent& ) adds the weight to channel SYN_EX (w > 0) or SYN_IN of the slot given by the delivery step;
  // in the input buffer of the block the slots of the neuron are number of neurons times number of channels apart
  auto& b = *block_;
  const size_t stride = b.neurons_.size() * b.num_channels_;
  assert( stride <= std::numeric_limits< uint32_t >::max() );
  return DirectSpikeInputAddr(
    b.input_.data() + block_index_ * b.num_channels_, stride, Buffers_::SYN_EX, Buffers_::SYN_IN );
}

}  // namespace nest
