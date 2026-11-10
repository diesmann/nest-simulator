# -*- coding: utf-8 -*-
#
# test_issue_3802.py
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
Regression test for Issue #3802 (GitHub).

Installing a changed module under the same file name after ``ResetKernel()`` must not use the earlier version.
Whether the system removes the earlier version from memory depends on compiler and loader (GCC keeps every library
that defines a template static data member); NEST then refuses the changed module.

The test compiles two versions of a minimal module with the compiler and flags reported by ``nest-config``.
"""

import os
import platform
import shlex
import shutil
import subprocess
import uuid

import nest
import pytest

MODULE_SOURCE = """
#include "iaf_psc_alpha.h"
#include "nest_extension_interface.h"

// An odr-used template static data member, as in synapse models; GCC gives it binding STB_GNU_UNIQUE.
template < int N >
struct Counter_{tag}
{
  static int value;
};
template < int N >
int Counter_{tag}< N >::value = 0;

class reloadmodule : public nest::NESTExtensionInterface
{
public:
  void
  initialize() override
  {
    ++Counter_{tag}< 1 >::value;
    nest::register_node_model< nest::iaf_psc_alpha >( "{model}" );
  }
};

reloadmodule reloadmodule_LTX_module;
"""


def nest_config(*args):
    return subprocess.run(["nest-config", *args], check=True, capture_output=True, text=True).stdout.split()


def build_module(directory, model, tag):
    """Compile the module registering ``model`` into ``directory/reloadmodule.so``."""

    directory.mkdir()
    source = directory / "reloadmodule.cpp"
    source.write_text(MODULE_SOURCE.replace("{model}", model).replace("{tag}", tag))
    link = ["-undefined", "dynamic_lookup"] if platform.system() == "Darwin" else []
    cmd = [
        *nest_config("--compiler"),
        *nest_config("--cflags"),
        *nest_config("--includes"),
        "-fPIC",
        "-shared",
        str(source),
        "-o",
        str(directory / "reloadmodule.so"),
        *link,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        pytest.skip(f"cannot compile a module ({shlex.join(cmd)}):\n{result.stderr}")
    return directory / "reloadmodule.so"


@pytest.fixture
def modules(tmp_path):
    """
    Build both versions of the module.

    glibc keeps only the first library that defines a given unique symbol in memory, so each test uses a template
    of its own name; otherwise the outcome would depend on the tests that ran before.
    """

    if shutil.which("nest-config") is None:
        pytest.skip("requires nest-config")
    tag = uuid.uuid4().hex
    return {model: build_module(tmp_path / model, model, tag) for model in ("a_model", "b_model")}


def place(module, target):
    """Install a module file as a rebuild does: a new file under the same name."""

    target.parent.mkdir(exist_ok=True)
    shutil.copy(module, target.with_suffix(".tmp"))
    os.replace(target.with_suffix(".tmp"), target)
    return str(target)


@pytest.mark.parametrize("second", ["a_model", "b_model"])
def test_reinstall_after_reset_never_uses_earlier_version(modules, tmp_path, second):
    """Ensure that a reinstalled module provides its own models, or that NEST refuses it."""

    target = tmp_path / "lib" / "reloadmodule.so"
    nest.ResetKernel()
    nest.Install(place(modules["a_model"], target))
    assert "a_model" in nest.node_models
    nest.Create("a_model")

    nest.ResetKernel()
    assert "a_model" not in nest.node_models
    path = place(modules[second], target)
    if second == "a_model":
        nest.Install(path)  # unchanged content is always accepted
    else:
        try:
            nest.Install(path)
        except nest.NESTErrors.DynamicModuleManagementError as err:
            assert "changed after an earlier version was loaded" in str(err)
            return
        assert "a_model" not in nest.node_models
    assert second in nest.node_models
    nest.ResetKernel()
