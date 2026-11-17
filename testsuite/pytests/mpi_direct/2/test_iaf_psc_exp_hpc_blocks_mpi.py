# -*- coding: utf-8 -*-
#
# test_iaf_psc_exp_hpc_blocks_mpi.py
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
Test that the neurons of ``iaf_psc_exp_hpc``, which are updated in blocks, give exactly the same results as
``iaf_psc_exp`` with several MPI processes (to be run with mpirun; the results of each process are compared). All
assertions follow all calls of Simulate, so that the processes call the collective operations alike.
"""

import nest
import numpy as np
import pytest

pytestmark = pytest.mark.skipif_missing_threads


def build_and_run(model, threads, direct):
    nest.ResetKernel()
    nest.set(local_num_threads=threads, rng_seed=321, direct_spike_delivery=direct)
    syn = "static_synapse_hpc" if direct else "static_synapse"
    pop_a = nest.Create(model, 120, params={"I_e": 380.0})
    pop_b = nest.Create(model, 41, params={"I_e": 360.0})  # a second block per thread, of uneven size
    neurons = pop_a + pop_b
    neurons.V_m = nest.random.uniform(-70.0, -55.0)
    rec = nest.Create("spike_recorder")
    mm = nest.Create("multimeter", params={"record_from": ["V_m"], "interval": 0.1})
    syn_ex = {"synapse_model": syn, "weight": nest.random.normal(20.0, 4.0), "delay": nest.random.uniform(1.0, 2.5)}
    syn_in = {"synapse_model": syn, "weight": -90.0, "delay": nest.random.uniform(1.0, 2.5)}
    nest.Connect(pop_a, neurons, {"rule": "fixed_indegree", "indegree": 40}, syn_ex)
    nest.Connect(pop_b, neurons, {"rule": "fixed_indegree", "indegree": 10}, syn_in)
    nest.Connect(neurons, rec)
    nest.Connect(mm, neurons[::7])
    nest.Simulate(50.0)
    nest.Simulate(50.0)
    ev = rec.events
    tr = mm.events
    order = np.lexsort((ev["senders"], ev["times"]))
    tr_order = np.lexsort((tr["senders"], tr["times"]))
    return ev["senders"][order], ev["times"][order], tr["V_m"][tr_order]


@pytest.mark.parametrize("threads", [1, 4])
@pytest.mark.parametrize("direct", [False, True], ids=["handle", "direct"])
def test_identical_results(threads, direct):
    ref = build_and_run("iaf_psc_exp", threads, direct)
    res = build_and_run("iaf_psc_exp_hpc", threads, direct)
    for a, b in zip(ref, res):
        np.testing.assert_array_equal(a, b)
