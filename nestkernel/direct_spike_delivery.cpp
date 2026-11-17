/*
 *  direct_spike_delivery.cpp
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

#include "direct_spike_delivery.h"

// Includes from nestkernel:
#include "node.h"

namespace nest
{

const DirectSpikeInputAddr&
DirectSpikeTable::resolve_( const size_t idx, Node* target )
{
  if ( idx >= entries_.size() )
  {
    entries_.resize( idx + 1 );
  }
  DirectSpikeInputAddr& entry = entries_[ idx ];
  entry = target->get_direct_spike_input();
  entry.resolved_ = true;
  return entry;
}

}  // namespace nest
