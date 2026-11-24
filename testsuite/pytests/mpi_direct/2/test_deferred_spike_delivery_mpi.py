# -*- coding: utf-8 -*-
#
# test_deferred_spike_delivery_mpi.py
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
Test that the deferred spike delivery (kernel parameter ``deferred_spike_delivery``) gives exactly the same results as
the regular delivery with several MPI processes (to be run with mpirun; the results of each process are compared). All
assertions follow all calls of Simulate, so that the processes call the collective operations alike.
"""

import nest
import numpy as np
import pytest

pytestmark = pytest.mark.skipif_missing_threads


def build_and_run(deferred, threads, poisson_delay=None):
    nest.ResetKernel()
    nest.set(
        local_num_threads=threads,
        rng_seed=321,
        direct_spike_delivery=True,
        tree_barrier=True,
        single_cycle_barrier=True,
        deferred_spike_delivery=deferred,
    )
    n_ex, n_in = 160, 40
    neurons = nest.Create("iaf_psc_exp", n_ex + n_in)
    neurons.I_e = nest.random.uniform(360.0, 420.0)
    neurons.V_m = nest.random.uniform(-70.0, -55.0)
    if poisson_delay:
        noise = nest.Create("poisson_generator", params={"rate": 4000.0})
        nest.Connect(noise, neurons, syn_spec={"weight": 10.0, "delay": poisson_delay})
    rec = nest.Create("spike_recorder")
    delay_ex = nest.math.redraw(nest.random.normal(mean=2.0, std=0.3), min=0.05, max=np.inf)
    delay_in = nest.math.redraw(nest.random.normal(mean=1.0, std=0.15), min=0.05, max=np.inf)
    syn_ex = {"synapse_model": "static_synapse_hpc", "weight": nest.random.normal(20.0, 4.0), "delay": delay_ex}
    syn_in = {"synapse_model": "static_synapse_hpc", "weight": -90.0, "delay": delay_in}
    nest.Connect(neurons[:n_ex], neurons, {"rule": "fixed_indegree", "indegree": 40}, syn_ex)
    nest.Connect(neurons[n_ex:], neurons, {"rule": "fixed_indegree", "indegree": 10}, syn_in)
    nest.Connect(neurons, rec)
    nest.Simulate(50.0)
    nest.Simulate(50.0)  # open entries of the first call are completed at its end
    ev = rec.events
    order = np.lexsort((ev["senders"], ev["times"]))
    return ev["senders"][order], ev["times"][order], np.array(neurons.get("V_m")), nest.deferred_spike_delivery_threads


@pytest.mark.parametrize("threads", [1, 4])
@pytest.mark.parametrize("poisson_delay", [None, 5.0])
def test_identical_results(threads, poisson_delay):
    ref = build_and_run(False, threads, poisson_delay)
    res = build_and_run(True, threads, poisson_delay)
    assert ref[3] == 0
    assert (res[3] > 0) == (threads > 1), "the test must exercise the deferral; thread 0 does not defer"
    for a, b in zip(ref[:3], res[:3]):
        np.testing.assert_array_equal(a, b)
