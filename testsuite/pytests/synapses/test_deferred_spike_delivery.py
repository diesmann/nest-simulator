# -*- coding: utf-8 -*-
#
# test_deferred_spike_delivery.py
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
Test the deferred spike delivery (kernel parameter ``deferred_spike_delivery``).

With deferral, a thread delivers before the update only the spikes that arrive in the next cycle and the others
later. Every input slot receives its additions in the same order as with the regular delivery, so spike trains and
membrane potentials must be identical, also with generators that add spikes to the same input slots.
"""

import nest
import numpy as np
import pytest

pytestmark = pytest.mark.skipif_missing_threads


def build_and_run(
    deferred,
    threads=2,
    min_delay_steps=1,
    target_model="iaf_psc_exp",
    synapse_model="static_synapse_hpc",
    between=None,
    poisson=False,
    poisson_delay=1.5,
):
    nest.ResetKernel()
    nest.set(
        local_num_threads=threads,
        rng_seed=123,
        direct_spike_delivery=True,
        tree_barrier=True,
        single_cycle_barrier=True,
        deferred_spike_delivery=deferred,
    )
    h = nest.resolution
    d_min = min_delay_steps * h

    n_ex, n_in = 160, 40
    neurons = nest.Create(target_model, n_ex + n_in)
    neurons.I_e = nest.random.uniform(360.0, 420.0)
    neurons.V_m = nest.random.uniform(-70.0, -55.0)
    if poisson:
        noise = nest.Create("poisson_generator", params={"rate": 4000.0})
        nest.Connect(noise, neurons, syn_spec={"weight": 10.0, "delay": poisson_delay})
    rec = nest.Create("spike_recorder")
    mm = nest.Create("multimeter", params={"record_from": ["V_m"], "interval": h})

    # normally distributed delays as in the microcircuit model (Potjans and Diesmann, 2014), longer, so that most
    # sources have no connection due in the next cycle and the threads defer; never below the minimum delay
    mean_ex, mean_in = max(2.0, 4 * d_min), max(1.0, 4 * d_min)
    delay_ex = nest.math.redraw(nest.random.normal(mean=mean_ex, std=0.15 * mean_ex), min=d_min - 0.5 * h, max=np.inf)
    delay_in = nest.math.redraw(nest.random.normal(mean=mean_in, std=0.15 * mean_in), min=d_min - 0.5 * h, max=np.inf)
    syn_ex = {"synapse_model": synapse_model, "weight": nest.random.normal(20.0, 4.0), "delay": delay_ex}
    syn_in = {"synapse_model": synapse_model, "weight": -90.0, "delay": delay_in}
    nest.Connect(neurons[:n_ex], neurons, {"rule": "fixed_indegree", "indegree": 40}, syn_ex)
    nest.Connect(neurons[n_ex:], neurons, {"rule": "fixed_indegree", "indegree": 10}, syn_in)
    nest.Connect(neurons, rec)
    nest.Connect(mm, neurons[:10])

    for t in (40.0, h, 13.3):  # several calls, one of a single step
        nest.Simulate(t)
    if between:
        between(neurons, rec, syn_ex)
    nest.Simulate(60.0)
    ev = rec.get("events")
    order = np.lexsort((ev["senders"], ev["times"]))
    vm = mm.get("events")
    vm_order = np.lexsort((vm["senders"], vm["times"]))
    return {
        "senders": ev["senders"][order],
        "times": ev["times"][order],
        "V_m": vm["V_m"][vm_order],
        "deferring_threads": nest.deferred_spike_delivery_threads,
    }


def assert_identical(a, b):
    """Spike trains and membrane potentials of two runs are the same."""
    for key in ("senders", "times", "V_m"):
        np.testing.assert_array_equal(a[key], b[key])


def add_nodes(neurons, rec, syn_ex):
    """New nodes and connections: the tables of the deferred delivery must be rebuilt."""
    new = nest.Create("iaf_psc_exp", 20, params={"I_e": 380.0})
    nest.Connect(neurons[:40], new, {"rule": "fixed_indegree", "indegree": 10}, syn_ex)
    nest.Connect(new, neurons[:40], {"rule": "fixed_indegree", "indegree": 5}, syn_ex)
    nest.Connect(new, rec)


def add_unsupported_targets(neurons, rec, syn_ex):
    """Targets without direct delivery among the targets of some sources: they are served in the first visit."""
    delta = nest.Create("iaf_psc_delta", 20, params={"I_e": 380.0})
    nest.Connect(neurons[:20], delta, {"rule": "fixed_indegree", "indegree": 5}, syn_ex)
    nest.Connect(delta, rec)


def shorten_delays(neurons, rec, syn_ex):
    """Delays changed by SetStatus, without a rebuild of the connections: connections due earlier must not be late."""
    conns = nest.GetConnections(neurons[:10], neurons)
    conns.delay = nest.min_delay  # the minimum delay cannot change after Simulate


def test_default_is_off_and_resets():
    nest.ResetKernel()
    assert nest.deferred_spike_delivery is False
    nest.deferred_spike_delivery = True
    assert nest.deferred_spike_delivery is True
    nest.ResetKernel()
    assert nest.deferred_spike_delivery is False


@pytest.mark.parametrize("threads", [1, 2, 4])
@pytest.mark.parametrize("min_delay_steps", [1, 10])
def test_identical_to_regular_delivery(threads, min_delay_steps):
    ref = build_and_run(False, threads=threads, min_delay_steps=min_delay_steps)
    assert len(ref["senders"]) > 100, "network must spike"
    assert ref["deferring_threads"] == 0
    res = build_and_run(True, threads=threads, min_delay_steps=min_delay_steps)
    assert res["deferring_threads"] > 0, "the test must exercise the deferral"
    assert_identical(ref, res)


@pytest.mark.parametrize("between", [add_nodes, add_unsupported_targets, shorten_delays])
def test_identical_with_changes_between_simulations(between):
    ref = build_and_run(False, threads=4, between=between)
    res = build_and_run(True, threads=4, between=between)
    assert res["deferring_threads"] > 0, "the test must exercise the deferral"
    assert_identical(ref, res)


def test_unsupported_target_model():
    """With only targets without direct delivery, every source needs a visit before the update: no thread defers."""
    ref = build_and_run(False, target_model="iaf_psc_delta")
    assert len(ref["senders"]) > 100, "network must spike"
    assert_identical(ref, build_and_run(True, target_model="iaf_psc_delta"))


def test_synapse_without_direct_delivery():
    """With static_synapse, nothing is deferred."""
    ref = build_and_run(False, synapse_model="static_synapse")
    assert_identical(ref, build_and_run(True, synapse_model="static_synapse"))


def test_mixed_synapse_types_fall_back():
    """A target receiving through synapse types with and without direct delivery: the thread does not defer."""

    def add_static(neurons, rec, syn_ex):
        nest.Connect(neurons[:20], neurons[20:40], "all_to_all", {"synapse_model": "static_synapse", "weight": 5.0})

    ref = build_and_run(False, between=add_static)
    assert_identical(ref, build_and_run(True, between=add_static))


@pytest.mark.parametrize("threads", [1, 4])
@pytest.mark.parametrize("poisson_delay", [1.5, 5.0])
def test_identical_with_poisson_input(threads, poisson_delay):
    """A generator adds spikes to the input slots during the update; deferred spikes must not change the order.

    With delay 1.5 ms of the generator the threads do not defer (most spikes would be visited twice); with 5 ms they
    defer the arrival steps up to the delay of the generator.
    """
    ref = build_and_run(False, threads=threads, poisson=True, poisson_delay=poisson_delay)
    assert len(ref["senders"]) > 100, "network must spike"
    res = build_and_run(True, threads=threads, poisson=True, poisson_delay=poisson_delay)
    assert (res["deferring_threads"] > 0) == (poisson_delay > 1.5)
    assert_identical(ref, res)


def run_order_probe(deferred):
    """Two spikes of weight 1 and a device spike of weight 1e16 in the same input slot.

    In the order of the regular delivery the two small additions come first and survive (1e16 + 2); if the device
    spike came first, they would be lost to rounding (1e16).
    """
    nest.ResetKernel()
    nest.set(
        local_num_threads=1,
        direct_spike_delivery=True,
        tree_barrier=True,
        single_cycle_barrier=True,
        deferred_spike_delivery=deferred,
    )
    target = nest.Create("iaf_psc_exp", params={"V_th": 1e30, "tau_syn_ex": 1e6})
    trigger = nest.Create("spike_generator", params={"spike_times": [10.0]})
    parrots = nest.Create("parrot_neuron", 2)
    nest.Connect(trigger, parrots, syn_spec={"delay": 0.1})
    # the parrots spike at 10.1 ms; their spikes arrive with the device spike of 10.2 ms and delay 1.5 ms
    nest.Connect(parrots, target, syn_spec={"synapse_model": "static_synapse_hpc", "weight": 1.0, "delay": 1.6})
    # silent sources with short delays, so that most sources need no visit before the update and the thread defers
    silent = nest.Create("parrot_neuron", 10)
    nest.Connect(silent, target, syn_spec={"synapse_model": "static_synapse_hpc", "weight": 1.0, "delay": 0.5})
    device = nest.Create("spike_generator", params={"spike_times": [10.2]})
    nest.Connect(device, target, syn_spec={"weight": 1e16, "delay": 1.5})
    mm = nest.Create("multimeter", params={"record_from": ["I_syn_ex"], "interval": 0.1})
    nest.Connect(mm, target)
    nest.Simulate(15.0)
    return mm.get("events")["I_syn_ex"].max()


def test_order_with_device_spikes():
    assert run_order_probe(False) == 1e16 + 2
    assert run_order_probe(True) == 1e16 + 2
    assert nest.deferred_spike_delivery_threads == 1, "the probe must exercise the deferral"
