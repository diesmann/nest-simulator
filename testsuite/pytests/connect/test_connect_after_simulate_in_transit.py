# -*- coding: utf-8 -*-
#
# test_connect_after_simulate_in_transit.py
#
# This file is part of NEST.
#
# Copyright (C) 2004 The NEST Initiative
#
# NEST is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# NEST is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with NEST.  If not, see <http://www.gnu.org/licenses/>.

"""
Spikes in transit across a change of connectivity between two ``Simulate()`` calls.

The spikes of the last time slice of a ``Simulate()`` call are delivered at the beginning of the next call. A
``Connect()`` or ``Disconnect()`` in between must not change where these spikes go (#3489). The tests compare
compressed and uncompressed spike delivery; each MPI process compares the spikes it records.
"""

import nest
import numpy as np
import pytest

pytestmark = pytest.mark.skipif_missing_threads


def recorded(sr):
    ev = sr.events
    order = np.lexsort((ev["senders"], ev["times"]))
    return list(zip(ev["times"][order], ev["senders"][order]))


def parrot_case(compressed):
    """
    p_a spikes at 9.5 ms, in the last slice of the first Simulate(), and is connected to p_t only. The new
    connection p_new -> p_x has a smaller source node ID than p_a and shifts the compressed indices.
    """

    nest.ResetKernel()
    nest.use_compressed_spikes = compressed
    p_new, p_a, p_t, p_x = nest.Create("parrot_neuron", 4)
    sg = nest.Create("spike_generator", params={"spike_times": [8.5]})
    sr = nest.Create("spike_recorder")
    nest.Connect(sg, p_a, syn_spec={"delay": 1.0})
    nest.Connect(p_a, p_t, syn_spec={"delay": 1.0})
    nest.Connect(p_a + p_t + p_x, sr)
    nest.Simulate(10.0)
    nest.Connect(p_new, p_x, syn_spec={"delay": 1.0})
    nest.Simulate(5.0)
    return recorded(sr), (p_a.global_id, p_t.global_id)


def network_case(compressed, disconnect):
    nest.ResetKernel()
    nest.set(local_num_threads=2, rng_seed=123, use_compressed_spikes=compressed)
    nest.CopyModel("static_synapse", "syn_new", {"weight": -20.0, "delay": 1.0})
    n_ex, n_in = 80, 20
    neurons = nest.Create("iaf_psc_exp", n_ex + n_in, params={"I_e": 300.0})
    neurons.V_m = nest.random.uniform(-70.0, -55.0)
    noise = nest.Create("poisson_generator", params={"rate": 8000.0})
    sr = nest.Create("spike_recorder")
    nest.Connect(noise, neurons, syn_spec={"weight": 15.0, "delay": 1.0})
    nest.Connect(neurons[:n_ex], neurons, {"rule": "fixed_indegree", "indegree": 20}, {"weight": 10.0, "delay": 1.0})
    nest.Connect(neurons[n_ex:], neurons, {"rule": "fixed_indegree", "indegree": 10}, {"weight": -40.0, "delay": 1.0})
    nest.Connect(neurons, sr)
    nest.Simulate(100.0)
    nest.Connect(neurons[n_ex:], neurons[:n_ex], {"rule": "pairwise_bernoulli", "p": 0.2}, {"synapse_model": "syn_new"})
    if disconnect:
        # The new connections are not yet sorted; Disconnect() must find them. They are the only ones of their
        # synapse model, so GetConnections() lists them in the order of creation with and without compression.
        nest.GetConnections(synapse_model="syn_new")[::2].disconnect()
    nest.Simulate(100.0)
    return recorded(sr)


def test_parrot_spike_in_transit_reaches_original_target():
    spikes = {c: parrot_case(c) for c in (False, True)}
    assert spikes[True] == spikes[False]
    events, (p_a, p_t) = spikes[True]
    if nest.NumProcesses() == 1:
        assert events == [(9.5, p_a), (10.5, p_t)]


@pytest.mark.parametrize("disconnect", [False, True])
def test_network_compressed_equals_uncompressed(disconnect):
    assert network_case(True, disconnect) == network_case(False, disconnect)
