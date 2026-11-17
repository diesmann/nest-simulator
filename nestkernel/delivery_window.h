/*
 *  delivery_window.h
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

#ifndef DELIVERY_WINDOW_H
#define DELIVERY_WINDOW_H

namespace nest
{

/**
 * Window of arrival steps for one pass of the deferred spike delivery.
 *
 * A pass delivers the connections of a segment whose arrival step a = stamp - 1 + delay lies in [ lo_, hi_ ) and
 * reports the smallest arrival step at or above hi_ in next_due_. The slots are computed with the moduli and the slice
 * origin of the cycle in which the thread delivers, not with the current values of the kernel: the master thread
 * advances the clock while other threads still deliver at the barrier. Targets without direct spike delivery are served
 * by the regular path, which uses the current values of the kernel, so they are served only in the first pass of an
 * entry, which is always made before the update. Arrival steps from cap_ on can also receive spikes of devices in the
 * next update and are therefore delivered in the first pass as well. See
 * EventDeliveryManager::deliver_events_deferred_().
 */
struct DeliveryWindow
{
  const long* moduli_;  //!< moduli of the cycle in which the thread delivers
  long origin_steps_;   //!< slice origin of that cycle
  long lo_;             //!< smallest arrival step to deliver
  long hi_;             //!< arrival steps from hi_ on are left for a later pass
  long cap_;            //!< arrival steps from cap_ on are delivered in the first pass
  bool first_pass_;     //!< whether targets without direct spike delivery and arrival steps from cap_ on are served
  long next_due_;       //!< output: smallest arrival step in [ hi_, cap_ ), LONG_MAX if none
};

}  // namespace nest

#endif /* DELIVERY_WINDOW_H */
