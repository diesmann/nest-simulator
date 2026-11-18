/*
 *  event_delivery_manager.h
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

#ifndef EVENT_DELIVERY_MANAGER_H
#define EVENT_DELIVERY_MANAGER_H

// C++ includes:
#include <atomic>
#include <cassert>
#include <climits>
#include <limits>
#include <utility>
#include <vector>

// Includes from libnestutil:
#include "manager_interface.h"
#include "stopwatch.h"

// Includes from nestkernel:
#include "buffer_resize_log.h"
#include "event.h"
#include "mpi_manager.h"  // OffGridSpike
#include "nest_time.h"
#include "nest_types.h"
#include "node.h"
#include "per_thread_bool_indicator.h"
#include "secondary_event.h"
#include "spike_data.h"
#include "target_table.h"
#include "vp_manager.h"


namespace nest
{
typedef MPIManager::OffGridSpike OffGridSpike;

class TargetData;
class SendBufferPosition;
class TargetSendBufferPosition;


class EventDeliveryManager : public ManagerInterface
{
public:
  EventDeliveryManager();
  ~EventDeliveryManager() override;

  void initialize( const bool ) override;
  void finalize( const bool ) override;
  void set_status( const Dictionary& ) override;
  void get_status( Dictionary& ) override;

  /**
   * Standard routine for sending events.
   *
   * This method decides if the event has to be delivered locally or globally. It exists
   * to keep a clean and unitary interface for the event sending mechanism.
   * @note Only specializations of SpikeEvent send remotely. A specialization for
   *       DSSpikeEvent exists to avoid that these events are sent to remote processes.
   * \see send_local()
   */
  template < class EventT >
  void send( Node& source, EventT& e, const long lag = 0 );

  /**
   * Send a secondary event remote.
   */
  void send_secondary( Node& source, SecondaryEvent& e );

  /**
   * Send event e to all targets of node source on thread t
   */
  void send_local( size_t t, Node& source, Event& e );

  /**
   * Add node ID of event sender to the spike_register.
   *
   * An event sent through this method will remain in the queue until
   * the network time has advanced by min_delay_ steps. After this period
   * the buffers are collocated and sent to the partner machines.

   * Old documentation from network.h:
   * Place an event in the global event queue.
   * Add event to the queue to be delivered
   * when it is due.
   * At the delivery time, the target list of the sender is iterated
   * and the event is delivered to all targets.
   * The event is guaranteed to arrive at the receiver when all
   * elements are updated and the system is
   * in a synchronised (single threaded) state.
   * @see send_to_targets()
   */
  void send_remote( size_t tid, SpikeEvent&, const long lag = 0 );

  /**
   * Add node ID of event sender to the spike_register.
   *
   * Store event offset with node ID.
   * An event sent through this method will remain in the queue until
   * the network time has advanced by min_delay_ steps. After this period
   * the buffers are collocated and sent to the partner machines.

   * Old documentation from network.h:
   * Place an event in the global event queue.
   * Add event to the queue to be delivered
   * when it is due.
   * At the delivery time, the target list of the sender is iterated
   * and the event is delivered to all targets.
   * The event is guaranteed to arrive at the receiver when all
   * elements are updated and the system is
   * in a synchronised (single threaded) state.
   * @see send_to_targets()
   */
  void send_off_grid_remote( size_t tid, SpikeEvent& e, const long lag = 0 );

  /**
   * Send event e directly to its target node.
   *
   * This should be
   * used only where necessary, e.g. if a node wants to reply
   * to a *RequestEvent immediately.
   */
  void send_to_node( Event& e );

  /**
   * return current communication style.
   * A result of true means off_grid, false means on_grid communication.
   */
  bool get_off_grid_communication() const;

  /**
   * set communication style to off_grid (true) or on_grid
   */
  void set_off_grid_communication( bool off_grid_spiking );

  /**
   * Return 0 for even, 1 for odd time slices.
   *
   * This is useful for buffers that need to be written alternatingly
   * by time slice. The value is given by get_slice_() % 2.
   * @see read_toggle
   */
  size_t write_toggle() const;

  /**
   * Return 1 - write_toggle().
   *
   * This is useful for buffers that need to be read alternatingly
   * by slice. The value is given by 1-write_toggle().
   * @see write_toggle
   */
  size_t read_toggle() const;

  /**
   * Return (T+d) mod max_delay.
   */
  long get_modulo( long d );

  //! Pointer p with p[ d ] == get_modulo( d ), valid until the clock advances (next update_moduli()).
  const long* get_moduli_data() const;

  //! Whether Connector::send() may deliver spikes directly into the input buffers of the targets.
  bool get_direct_spike_delivery() const;

  /**
   * Decide before a simulation whether the threads defer the delivery of spikes; not thread safe.
   *
   * The kernel parameter deferred_spike_delivery takes effect only with direct spike delivery, the single barrier per
   * cycle, compressed spikes and delays larger than the minimum delay; see deliver_events_deferred_().
   */
  void configure_deferred_delivery( const bool single_cycle_barrier );

  //! The connection infrastructure was rebuilt; the tables of the deferred delivery must be rebuilt.
  void invalidate_deferred_tables();

  //! Whether the threads defer the delivery of spikes in this simulation.
  bool is_deferred_delivery_active() const;

  /**
   * Prepare the deferred delivery of thread tid at the start of a simulation.
   *
   * Builds the table of the smallest delay per source and synapse type if the connections changed, and decides whether
   * the thread defers: a thread whose targets with direct spike delivery also receive spikes through a synapse type
   * without it delivers as before, since the order of the additions to an input slot would change.
   */
  void prepare_deferred_delivery( const size_t tid );

  /**
   * Complete the oldest open entry of thread tid; false if there is none.
   *
   * Called by the thread while it waits at the barrier, see TreeBarrier::wait().
   */
  bool deliver_deferred_one( const size_t tid );

  //! Deliver all open entries of thread tid at the end of a simulation, with the time base of the current clock.
  void finish_deferred_delivery( const size_t tid );


  /**
   * Index to slice-based buffer.
   * Return ((T+d)/min_delay) % ceil(max_delay/min_delay).
   */
  long get_slice_modulo( long d );

  /**
   * Resize spike_register and comm_buffer to correct dimensions.
   *
   * Resizes also off_grid_*_buffer_.
   * This is done by simulate() when called for the first time.
   * The spike buffers cannot be reconfigured later, whence neither
   * the number of local threads or the min_delay can change after
   * simulate() has been called. ConnectorModel::check_delay() and
   * Network::set_status() ensure this.
   */
  void configure_spike_data_buffers();

  void configure_spike_register();

  void resize_send_recv_buffers_target_data();

  void configure_secondary_buffers();

  /**
   * Collocates spikes from register to MPI buffers, communicates via
   * MPI and delivers events to targets.
   */
  void gather_spike_data();

  /**
   * True if the spike registers are collocated by several threads in parallel.
   *
   * This is the case if the kernel parameter collocation_groups is larger than one, there is more than one thread and
   * the spikes are on-grid. The threads are divided into groups of consecutive threads. After the first barrier of a
   * cycle the first thread of every group copies the spike registers of its group into buffers of the group (one per
   * target rank), all groups in parallel (collocate_groups()). With several MPI processes the master thread then
   * concatenates the buffers of the groups, in the order of the groups, into the send buffer, which is what the
   * collocation of a single thread would have produced; the MPI communication is unchanged. With a single MPI process
   * (and compressed spikes) there is no communication at all: after the second barrier every thread delivers directly
   * from the buffers of the groups.
   */
  bool use_group_collocation() const;

  /**
   * Collocation of the group of thread tid by its first thread (no operation for the other threads).
   *
   * Called by all threads between the two barriers of a cycle, if use_group_collocation() was true before the first
   * barrier.
   */
  void collocate_groups( const size_t tid );

  /**
   * Collocation of group g, by any one thread, after all threads of the group have finished their update.
   *
   * Called by collocate_groups() (first thread of the group, two barriers) and as the group step of the single barrier
   * (last thread of the group to arrive, see TreeBarrier).
   */
  void collocate_group( const size_t g );

  //! First thread of each collocation group, followed by the number of threads; valid after
  //! configure_spike_data_buffers().
  std::vector< size_t > get_collocation_group_bounds() const;

  /**
   * Collocates presynaptic connection information, communicates via
   * MPI and creates presynaptic connection infrastructure.
   */
  void gather_target_data( const size_t tid );

  void gather_target_data_compressed( const size_t tid );


  /**
   * Delivers events to targets.
   */
  void deliver_events( const size_t tid );

  /**
   * Collocates presynaptic connection information for secondary events (MPI
   * buffer offsets), communicates via MPI and create presynaptic connection
   * infrastructure for secondary events.
   */
  void gather_secondary_target_data();

  void write_done_marker_secondary_events_( const bool done );

  void gather_secondary_events( const bool done );

  bool deliver_secondary_events( const size_t tid, const bool called_from_wfr_update );

  /**
   * Advance the modulo tables to the current network clock.
   *
   * This function is called by the master thread whenever the network clock advances (once per time slice). The tables
   * themselves are fixed (see init_moduli()); only the offsets into them change. The tables are read for every event
   * written into a ring buffer, by all threads; rewriting them in every slice would force every thread to fetch them
   * again from the cache of the master thread in every slice.
   */
  void update_moduli();

  /**
   * Initialize the modulo tables for the present min_delay and max_delay and set the offsets to the network clock.
   *
   * TODO: can probably be private
   */
  void init_moduli();

  /**
   * Set local spike counter to zero.
   */
  virtual void reset_counters();

  /**
   * Set time measurements for internal profiling to zero (reg. prep.)
   */
  virtual void reset_timers_for_preparation();

  /**
   * Set time measurements for internal profiling to zero (reg. sim. dyn.)
   */
  virtual void reset_timers_for_dynamics();

#ifdef CYCLE_TIMERS
  /**
   * Returns the elapsed time of sw_communicate_spike_data_
   * Used in SimulationManager::update_() for cycle timers
   */
  Stopwatch< StopwatchGranularity::Detailed, StopwatchParallelism::MasterOnly > get_sw_communicate_spike_data() const;

  /**
   * Accumulates the spike counts stored in local_spike_counter_
   * Used in SimulationManager::update_() for cycle timers
   */
  size_t get_local_spike_counter() const;
#endif

private:
  //! A spike entry in the FIFO of the deferred delivery of a thread.
  struct DeferredEntry_
  {
    synindex syn_id_;
    size_t idx_;       //!< row in the compressed spike data
    size_t lcid_;      //!< start of the segment on the thread; invalid_lcid until the first visit
    long stamp_;       //!< time stamp of the spike in steps
    double offset_;    //!< offset of the spike for off-grid spiking
    long done_below_;  //!< connections with arrival steps below are delivered; LONG_MIN before the first visit
    long next_due_;    //!< smallest arrival step not yet delivered (a lower bound), LONG_MAX when complete
    long cap_;         //!< arrival steps from here on are delivered in the first visit (spikes of devices)
    bool deferrable_;  //!< false: the synapse type is delivered completely in the first visit
  };

  //! State of the deferred delivery of one thread; written only by its thread.
  struct alignas( 64 ) DeferredThread_
  {
    bool active_ = false;           //!< whether the thread defers in this simulation
    const long* moduli_ = nullptr;  //!< moduli of the current cycle of the thread
    long origin_steps_ = 0;         //!< slice origin of the current cycle of the thread
    std::vector< DeferredEntry_ > fifo_;
    size_t head_ = 0;  //!< entries before head_ are complete
    /**
     * Per synapse type and row of the compressed spike data: smallest and largest delay of the segment on the thread.
     *
     * The smallest is 0 if the row has no target on the thread. Stored in 16 bits, since the table is read for every
     * spike; a larger smallest delay is stored as the maximum (the entry is visited earlier than needed), a larger
     * largest delay as the maximum, which counts as beyond any cap (the entry is visited before the update).
     */
    std::vector< std::vector< std::pair< unsigned short, unsigned short > > > delays_of_;
    long device_delay_ = LONG_MAX;    //!< ConnectionManager::min_delay_from_spike_devices() for the thread
    double skippable_share_ = 0.0;    //!< share of the sources whose spikes need no visit before the update
    std::vector< char > deferrable_;  //!< per synapse type
    bool tables_valid_ = false;       //!< delays_of_ and device_delay_ fit the connections
    size_t status_changes_ = 0;       //!< ConnectionManager::get_synapse_status_changes() when the tables were built
  };

  template < typename SpikeDataT >
  void gather_spike_data_( std::vector< SpikeDataT >& send_buffer, std::vector< SpikeDataT >& recv_buffer );

  void resize_send_recv_buffers_spike_data_();

  /**
   * Moves spikes from on grid and off grid spike registers to correct
   * locations in MPI buffers.
   *
   * Accumulates count of spikes to be sent to any rank in num_spikes_per_rank.
   * Passed as argument, so that values accumulate as we call once for plain and once for offgrid spikes.
   */
  template < typename SpikeDataWithRankT, typename SpikeDataT >
  void collocate_spike_data_buffers_( SendBufferPosition& send_buffer_position,
    std::vector< std::vector< SpikeDataWithRankT >* >& spike_register,
    std::vector< SpikeDataT >& send_buffer,
    std::vector< size_t >& num_spikes_per_rank );

  /**
   * Set end marker for per-rank-chunks signalling completion and providing shrink/grow information.
   */
  template < typename SpikeDataT >
  void set_end_marker_( const SendBufferPosition& send_buffer_position,
    std::vector< SpikeDataT >& send_buffer,
    size_t local_max_spikes_per_rank );

  /**
   * Resets marker in MPI buffer that signals end of communication
   * across MPI ranks.
   */
  template < typename SpikeDataT >
  void reset_complete_marker_spike_data_( const SendBufferPosition& send_buffer_position,
    std::vector< SpikeDataT >& send_buffer ) const;

  /**
   * Get required buffer size.
   *
   * @returns maximum of required buffer sizes communicated by all ranks
   */
  template < typename SpikeDataT >
  size_t get_global_max_spikes_per_rank_( const SendBufferPosition& send_buffer_position,
    std::vector< SpikeDataT >& recv_buffer ) const;


  /**
   * Reads spikes from MPI buffers and delivers them to ringbuffer of
   * nodes.
   */
  template < typename SpikeDataT >
  void deliver_events_( const size_t tid, const std::vector< SpikeDataT >& recv_buffer );

  //! Delivers events directly from the buffers of the groups (single MPI process, see use_group_collocation()).
  void deliver_events_from_groups_( const size_t tid );

  /**
   * Delivery before the update with deferral (kernel parameter deferred_spike_delivery).
   *
   * An event needs to be delivered before the update of the step in which it arrives, not before the next update. Every
   * spike entry with targets on the thread is appended to a FIFO of the thread and delivered in windows of arrival
   * steps a = stamp - 1 + delay. Before the update of the cycle with slice origin O, the entries still open from the
   * previous cycle are completed, then the new entries deliver their connections with a < O + min_delay, in FIFO order.
   * While the thread waits at the barrier, it completes the oldest entries (deliver_deferred_one()). Waiting longer
   * than one cycle did not pay: a thread that had no time at the barrier accumulated entries, and PD14 became slower
   * (measured 2026-11-16 on Intel Xeon 6980P and AMD EPYC 9754: completion after one cycle 15 to 17 % faster than
   * without deferral, after two cycles 9 to 14 %).
   *
   * Entries of synapse types that cannot be deferred are kept in the FIFO as well and delivered completely in their
   * first visit, so that targets without direct delivery receive all spikes in the order of the regular delivery.
   *
   * Devices such as poisson_generator add their spikes to the input slots during the update. A device connection with
   * delay d reaches arrival steps from O + d on, where O is the origin of the cycle of the update. Arrival steps from
   * this cap on (with the smallest delay from such a device to a target with direct delivery on the thread) are
   * therefore delivered in the first visit of an entry, before the update; only the arrival steps between the next
   * cycle and the cap are deferred. Then every input slot receives its additions in the order of the regular
   * delivery: the contributions of one entry to one slot all have the same arrival step and fall into the same pass,
   * per slot the contributions of older entries precede those of younger ones, and no deferred contribution goes to a
   * slot that a device reaches in the meantime. The results are the same as with the regular delivery.
   *
   * A thread defers only if at least half of its sources need no visit before the update (see
   * prepare_deferred_delivery()); with devices whose delays are as short as those of the network, almost every entry
   * would be visited twice.
   */
  void deliver_events_deferred_( const size_t tid );

  /**
   * Deliver the windows up to hi of the entries [ begin, end ) of the FIFO of thread tid that have a connection below
   * hi.
   *
   * In batches: the look-ups in the compressed spike data of a batch are prefetched together, then its first
   * connections, then the events are delivered; a look-up and a first connection each miss the cache.
   */
  void deliver_deferred_batch_( const size_t tid, const size_t begin, const size_t end, const long hi );

  //! Deliver the window [ entry.done_below, hi ) of one entry and advance the entry.
  void deliver_deferred_window_( const size_t tid, DeferredEntry_& entry, const long hi );

  //! Copies the buffers of the groups into the send buffer like collocate_spike_data_buffers_() does for the registers.
  template < typename SpikeDataT >
  void concatenate_group_buffers_( SendBufferPosition& send_buffer_position,
    std::vector< SpikeDataT >& send_buffer,
    std::vector< size_t >& num_spikes_per_rank );

  /**
   * Deliver the compressed spikes in recv_buffer with software prefetching (spike_delivery_prefetch_).
   *
   * Same as deliver_events_() for compressed spikes, but the full batches of every rank are delivered by
   * deliver_batches_prefetch_(). A separate function, so that deliver_events_() is compiled from unchanged source.
   */
  template < typename SpikeDataT >
  void deliver_events_prefetch_( const size_t tid, const std::vector< SpikeDataT >& recv_buffer );

  /**
   * Deliver the full batches of compressed-spike entries of one rank with software prefetching.
   *
   * The batches are delivered in the same order and with the same operations as in deliver_events_(), so the results
   * are identical. In addition, the memory that later batches will need is prefetched: the header and the entry of
   * the compressed spike data four and three batches ahead, and the first connection of every entry with a target on
   * the thread two batches ahead, where the look-up of the compressed spike data is done.
   */
  template < typename SpikeDataT >
  void deliver_batches_prefetch_( const size_t tid,
    const SpikeDataT* const entries,
    const size_t num_batches,
    const std::vector< Time >& prepared_timestamps,
    const std::vector< ConnectorModel* >& cm );

  /**
   * Deletes all spikes from spike registers and resets spike
   * counters.
   */
  void reset_spike_register_( const size_t tid );

  //! First thread of collocation group g; for g = number of groups, the number of threads.
  size_t collocation_group_begin_( const size_t g ) const;

  /**
   * Returns true if spike has been moved to MPI buffer, such that it
   * can be removed by clean_spike_register. Required static function
   * by std::remove_if.
   */
  static bool is_marked_for_removal_( const Target& target );

  /**
   * Fills MPI buffer for communication of connection information from
   * presynaptic to postsynaptic side. Builds TargetData objects from
   * SourceTable and connections information.
   */
  bool collocate_target_data_buffers_( const size_t tid,
    const AssignedRanks& assigned_ranks,
    TargetSendBufferPosition& send_buffer_position );

  bool collocate_target_data_buffers_compressed_( const size_t tid,
    const AssignedRanks& assigned_ranks,
    TargetSendBufferPosition& send_buffer_position );

  /**
   * Sets marker in MPI buffer that signals end of communication
   * across MPI ranks.
   */
  void set_complete_marker_target_data_( const AssignedRanks& assigned_ranks,
    const TargetSendBufferPosition& send_buffer_position );

  /**
   * Reads TargetData objects from MPI buffers and creates Target
   * objects on TargetTable (presynaptic part of connection
   * infrastructure).
   */
  bool distribute_target_data_buffers_( const size_t tid );

  /**
   * Sends event e to all targets of node source. Delivers events from
   * devices directly to targets.
   */
  template < class EventT >
  void send_local_( Node& source, EventT& e, const long lag );
  void send_local_( Node& source, SecondaryEvent& e, const long lag );

  //--------------------------------------------------//

  bool off_grid_spiking_;  //!< indicates whether spikes are not constrained to
                           //!< the grid

  //! Number of groups of threads that collocate the spike registers in parallel (1: master thread alone).
  size_t collocation_groups_;

  //! Buffers of one group of threads for the collocation of the spike registers (see use_group_collocation()).
  struct CollocationGroup_
  {
    size_t first_thread_ = 0;  //!< first thread of the group; the group ends where the next one begins
    //! [ target rank ] spikes of the last cycle, read by the delivery or the MPI exchange
    std::vector< std::vector< SpikeData > > spikes_;
    //! [ target rank ] spikes collected in the current cycle; swapped with spikes_ in gather_spike_data()
    std::vector< std::vector< SpikeData > > spikes_next_;
  };

  /**
   * The collocation groups.
   *
   * Two sets of spikes per group, because with one barrier per cycle a group collocates as soon as its own threads have
   * arrived, while threads of other groups may still deliver the spikes of the last cycle. The master thread swaps the
   * sets within each group; the vector of the groups does not change during a simulation, so that every thread can
   * read the groups and their first threads at any time. Both sets live in this one member, because an additional
   * member of the manager changes the memory layout of the kernel, which cost up to 4 % on PD14.
   */
  std::vector< CollocationGroup_ > group_buffers_;

  //! Counter on a cache line of its own (see groups_collocated_).
  struct alignas( 64 ) PaddedCounter_
  {
    std::atomic< size_t > value_ = 0;
  };

  /**
   * Number of groups that have finished their collocation in the current cycle.
   *
   * Atomic, because the collocating threads of all groups increment it concurrently while the master thread spins on
   * it in gather_spike_data() before it swaps the group buffers; release and acquire make the group buffers visible to
   * the master. On a cache line of its own, because it is written in every cycle while all threads read the
   * neighboring members (moduli_ for every delivered spike); sharing a line cost up to 13 % on PD14 (one process of
   * 288 threads on four NVIDIA Grace chips).
   */
  PaddedCounter_ groups_collocated_;

  //! Deliver spikes of hpc static synapses directly into the input buffer of supporting targets.
  bool direct_spike_delivery_;

  /**
   * Software prefetching in the delivery of compressed spikes (kernel parameter spike_delivery_prefetch, default
   * false).
   *
   * See deliver_events_prefetch_(). Kept next to off_grid_spiking_ so that it occupies padding and the size of the
   * manager does not change.
   */
  bool spike_delivery_prefetch_;

  /**
   * Table of pre-computed modulos.
   *
   * This table is used to map time steps, given as offset from now, to ring-buffer bins. There are
   * n = min_delay + max_delay bins in a ring buffer. The table holds i mod n for i = 0, ..., 2n - 1 and does not change
   * during a simulation; the bin of offset d is moduli_[ moduli_offset_ + d ], with moduli_offset_ = T mod n for the
   * network clock T (see CycleOffsets_).
   * @see RingBuffer
   */
  std::vector< long > moduli_;

  /**
   * Table of pre-computed slice-based modulos.
   *
   * This table is used to map time steps, given as offset from now, to slice-based ring-buffer bins. There are
   * nbuff = ceil( ( min_delay + max_delay ) / min_delay ) bins in a slice-based ring buffer, one per slice. The value
   * for the step T + d is ( ( T + d ) / min_delay ) mod nbuff, which is periodic in T with the period
   * p = min_delay * nbuff. The table holds ( i / min_delay ) mod nbuff for i = 0, ..., p + min_delay + max_delay - 1
   * and does not change during a simulation; the bin of offset d is slice_moduli_[ slice_moduli_offset_ + d ], with
   * slice_moduli_offset_ = T mod p.
   * @see SliceRingBuffer
   */
  std::vector< long > slice_moduli_;

  //! Number of entries of a ring buffer, min_delay + max_delay (the valid offsets d are 0, ..., num_moduli_ - 1).
  long num_moduli_;

  /**
   * Offsets into the modulo tables, the only part that changes when the network clock advances.
   *
   * They have a cache line of their own, so that the rest of the object, which all threads read for every event, is
   * not invalidated in every time slice.
   */
  struct alignas( 64 ) CycleOffsets_
  {
    long moduli_offset_ = 0;
    long slice_moduli_offset_ = 0;
  };
  CycleOffsets_ offsets_;
  long slice_moduli_period_;  //!< min_delay * nbuff, period of the slice-based modulos

  /**
   * Register of emitted spikes.
   *
   * All spikes to be delivered non-locally are first written to this register by the thread generating the spike.
   * They are later transferred to communication buffers and exchanged globally.
   *
   * The outer dimension represents the thread generating the spikes, the second dimension the individual spikes.
   *
   * @note We store here pointers to the vectors for the individual threads so that those vectors, including their
   * administrative metadata will be stored in thread-local memory.
   */
  std::vector< std::vector< SpikeDataWithRank >* > emitted_spikes_register_;

  /**
   * Register of emitted off-grid spikes.
   *
   * All off-grid spikes to be delivered non-locally are first written to this register by the thread generating the
   * spike. They are later transferred to communication buffers and exchanged globally.
   *
   * The outer dimension represents the thread generating the spikes, the second dimension the individual spikes.
   *
   * @note We store here pointers to the vectors for the individual threads so that those vectors, including their
   * administrative metadata will be stored in thread-local memory.
   */
  std::vector< std::vector< OffGridSpikeDataWithRank >* > off_grid_emitted_spikes_register_;

  /**
   * Buffer to collect the secondary events after serialization.
   */
  std::vector< unsigned int > send_buffer_secondary_events_;
  std::vector< unsigned int > recv_buffer_secondary_events_;

  /**
   * Number of generated spike events (both off- and on-grid) during the last call to simulate.
   */
  std::vector< unsigned long > local_spike_counter_;

  std::vector< SpikeData > send_buffer_spike_data_;
  std::vector< SpikeData > recv_buffer_spike_data_;
  std::vector< OffGridSpikeData > send_buffer_off_grid_spike_data_;
  std::vector< OffGridSpikeData > recv_buffer_off_grid_spike_data_;

  std::vector< TargetData > send_buffer_target_data_;
  std::vector< TargetData > recv_buffer_target_data_;

  //! whether size of MPI buffer for communication of connections was changed
  bool buffer_size_target_data_has_changed_;

  /**
   * Largest number of spikes sent from any rank to any other rank in last spike exchange round.
   *
   * The spike buffer section for any rank must be at least this size. Therefore, this number controls
   * buffer resizing.
   */
  size_t global_max_spikes_per_rank_;

  double send_recv_buffer_shrink_limit_;  //!< shrink buffer only if below this limit
  double send_recv_buffer_shrink_spare_;  //!< leave this fraction more space than minimally needed
  double send_recv_buffer_grow_extra_;    //!< when growing, add this fraction extra space

  /**
   * Log all resize events.
   *
   * This is maintained by the main thread, which is responsible for communication and resizing.
   */
  BufferResizeLog send_recv_buffer_resize_log_;

  PerThreadBoolIndicator gather_completed_checker_;

  // private stop watches for benchmarking purposes
  // (intended for internal core developers, not for use in the public API)
  Stopwatch< StopwatchGranularity::Detailed, StopwatchParallelism::MasterOnly > sw_collocate_spike_data_;
  Stopwatch< StopwatchGranularity::Detailed, StopwatchParallelism::MasterOnly > sw_communicate_spike_data_;
  Stopwatch< StopwatchGranularity::Detailed, StopwatchParallelism::MasterOnly > sw_communicate_target_data_;

  bool deferred_spike_delivery_;   //!< kernel parameter deferred_spike_delivery
  bool deferred_delivery_active_;  //!< decided by configure_deferred_delivery()
  std::vector< DeferredThread_ > deferred_;
};

inline void
EventDeliveryManager::reset_spike_register_( const size_t tid )
{
  emitted_spikes_register_[ tid ]->clear();
  off_grid_emitted_spikes_register_[ tid ]->clear();
}

inline bool
EventDeliveryManager::is_marked_for_removal_( const Target& target )
{
  return target.is_processed();
}

inline void
EventDeliveryManager::send_to_node( Event& e )
{
  e();
}

inline const long*
EventDeliveryManager::get_moduli_data() const
{
  // entry d of the returned array is get_modulo( d ) (fixed tables with an offset, see moduli_)
  return moduli_.data() + offsets_.moduli_offset_;
}

inline bool
EventDeliveryManager::get_direct_spike_delivery() const
{
  return direct_spike_delivery_;
}

inline bool
EventDeliveryManager::is_deferred_delivery_active() const
{
  return deferred_delivery_active_;
}

inline bool
EventDeliveryManager::get_off_grid_communication() const
{
  return off_grid_spiking_;
}

inline void
EventDeliveryManager::set_off_grid_communication( bool off_grid_spiking )
{
  off_grid_spiking_ = off_grid_spiking;
}

inline size_t
EventDeliveryManager::read_toggle() const
{
  // define in terms of write_toggle() to ensure consistency
  return 1 - write_toggle();
}

inline long
EventDeliveryManager::get_modulo( long d )
{
  // Note, here d may be 0, since bin 0 represents the "current" time
  // when all events due are read out.
  assert( 0 <= d and d < num_moduli_ );

  return moduli_[ offsets_.moduli_offset_ + d ];
}

inline long
EventDeliveryManager::get_slice_modulo( long d )
{
  // Note, here d may be 0, since bin 0 represents the "current" time
  // when all events due are read out.
  assert( 0 <= d and d < num_moduli_ );

  return slice_moduli_[ offsets_.slice_moduli_offset_ + d ];
}

inline size_t
EventDeliveryManager::collocation_group_begin_( const size_t g ) const
{
  return g < group_buffers_.size() ? group_buffers_[ g ].first_thread_ : emitted_spikes_register_.size();
}

inline std::vector< size_t >
EventDeliveryManager::get_collocation_group_bounds() const
{
  std::vector< size_t > bounds( group_buffers_.size() + 1 );
  for ( size_t g = 0; g < bounds.size(); ++g )
  {
    bounds[ g ] = collocation_group_begin_( g );
  }
  return bounds;
}

}  // namespace nest

#endif /* EVENT_DELIVERY_MANAGER_H */
