/*
 *  connector_base_impl.h
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

#include "connector_base.h"

// C++ includes:
#include <algorithm>
#include <climits>
#include <typeinfo>

// Includes from nestkernel:
#include "kernel_manager.h"

// Includes from models:
#include "weight_recorder.h"

#ifndef CONNECTOR_BASE_IMPL_H
#define CONNECTOR_BASE_IMPL_H

namespace nest
{

template < typename ConnectionT >
size_t
Connector< ConnectionT >::send_direct_( const size_t tid,
  const size_t lcid,
  const typename ConnectionT::CommonPropertiesType& cp,
  Event& e )
{
  EventDeliveryManager& edm = kernel().event_delivery_manager;
  if ( not edm.get_direct_spike_delivery() or cp.get_weight_recorder() != nullptr or typeid( e ) != typeid( SpikeEvent )
    or static_cast< SpikeEvent& >( e ).get_multiplicity() != 1 )
  {
    return 0;
  }
  DeliveryWindow all {
    edm.get_moduli_data(), kernel().simulation_manager.get_slice_origin().get_steps(), 0, 0, 0, true, 0
  };
  return send_segment_< false >( tid, lcid, cp, e, all );
}

template < typename ConnectionT >
template < bool windowed >
size_t
Connector< ConnectionT >::send_segment_( const size_t tid,
  const size_t lcid,
  const typename ConnectionT::CommonPropertiesType& cp,
  Event& e,
  DeliveryWindow& window )
{
  const long* const moduli = window.moduli;
  // as SpikeEvent::get_rel_delivery_steps(), without the delay
  const long stamp_m1 = e.get_stamp().get_steps() - 1;
  const long rel0 = stamp_m1 - window.origin_steps;
  long next_due = LONG_MAX;

  // The index advances unconditionally and the flag only decides whether the loop continues: the address of the next
  // connection must not depend on data of the current one, so that the loads of consecutive connections can overlap.
  size_t i = lcid;
  bool more_targets;
  do
  {
    assert( i < C_.size() );
    ConnectionT& conn = C_[ i ];
    if ( not conn.is_disabled() )
    {
      const DirectSpikeInputAddr& addr = direct_table_.get( conn, tid );
      if ( addr.is_supported() )
      {
        const long d = conn.get_delay_steps();
        if constexpr ( windowed )
        {
          const long arrival = stamp_m1 + d;
          if ( arrival >= window.cap ? window.first_pass : ( arrival >= window.lo and arrival < window.hi ) )
          {
            addr.add( moduli[ rel0 + d ], conn.get_weight() );
          }
          else if ( arrival >= window.hi and arrival < window.cap )
          {
            next_due = std::min( next_due, arrival );
          }
        }
        else
        {
          addr.add( moduli[ rel0 + d ], conn.get_weight() );
        }
      }
      else if ( not windowed or window.first_pass )
      {
        e.set_port( i );
        conn.send( e, tid, cp );
      }
    }
    more_targets = conn.source_has_more_targets();
    ++i;
  } while ( more_targets );

  window.next_due = next_due;
  return i - lcid;
}

template < typename ConnectionT >
bool
Connector< ConnectionT >::supports_deferred_delivery( const std::vector< ConnectorModel* >& cm ) const
{
  if constexpr ( ConnectionT::supports_direct_delivery )
  {
    const auto& cp = static_cast< GenericConnectorModel< ConnectionT >* >( cm[ syn_id_ ] )->get_common_properties();
    return kernel().event_delivery_manager.get_direct_spike_delivery() and cp.get_weight_recorder() == nullptr;
  }
  return false;
}

template < typename ConnectionT >
void
Connector< ConnectionT >::send_window( const size_t tid,
  const size_t lcid,
  const std::vector< ConnectorModel* >& cm,
  SpikeEvent& e,
  DeliveryWindow& window )
{
  if constexpr ( ConnectionT::supports_direct_delivery )
  {
    const auto& cp = static_cast< GenericConnectorModel< ConnectionT >* >( cm[ syn_id_ ] )->get_common_properties();
    send_segment_< true >( tid, lcid, cp, e, window );
  }
  else
  {
    assert( false and "send_window() requires a connector that supports deferred delivery" );
  }
}

template < typename ConnectionT >
std::pair< long, long >
Connector< ConnectionT >::segment_delay_range( const size_t tid, const size_t lcid )
{
  if constexpr ( not ConnectionT::supports_direct_delivery )
  {
    return { 1, 0 };  // never called: such connectors do not defer
  }
  long min_delay = LONG_MAX;
  long max_delay = 0;
  size_t i = lcid;
  bool more_targets;
  do
  {
    ConnectionT& conn = C_[ i ];
    if ( not conn.is_disabled() )
    {
      const long d = conn.get_delay_steps();
      max_delay = std::max( max_delay, d );
      if constexpr ( ConnectionT::supports_direct_delivery )
      {
        // a target without direct delivery is served in the first pass, which is due at once
        min_delay = direct_table_.get( conn, tid ).is_supported() ? std::min( min_delay, d ) : 1;
      }
      else
      {
        min_delay = 1;
      }
    }
    more_targets = conn.source_has_more_targets();
    ++i;
  } while ( more_targets );
  return { min_delay == LONG_MAX ? 1 : min_delay, max_delay };
}



template < typename ConnectionT >
void
Connector< ConnectionT >::send_weight_event( const size_t tid,
  const unsigned int lcid,
  Event& e,
  const CommonSynapseProperties& cp )
{
  // If the pointer to the receiver node in the event is invalid,
  // the event was not sent, and a WeightRecorderEvent is therefore not created.
  if ( cp.get_weight_recorder() and e.receiver_is_valid() )
  {
    // Create new event to record the weight and copy relevant content.
    WeightRecorderEvent wr_e;
    wr_e.set_port( e.get_port() );
    wr_e.set_rport( e.get_rport() );
    wr_e.set_stamp( e.get_stamp() );
    // Sender is not available for SecondaryEvents, and not needed, so we do not
    // set it to avoid undefined behavior.
    wr_e.set_sender_node_id( kernel().connection_manager.get_source_node_id( tid, syn_id_, lcid ) );
    wr_e.set_weight( e.get_weight() );
    wr_e.set_delay_steps( e.get_delay_steps() );
    wr_e.set_receiver( *static_cast< Node* >( cp.get_weight_recorder() ) );
    // Set the node_id of the postsynaptic node as receiver node ID
    wr_e.set_receiver_node_id( e.get_receiver_node_id() );
    wr_e();
  }
}

}  // of namespace nest

#endif
