"""Derivative tests for FDEM problems."""

import itertools

import numpy as np
from scipy.constants import mu_0
from discretize.tests import check_derivative
from simpeg.electromagnetics.utils.testing_utils import (
    getFDEMProblem,
    get_FDEM_hierarchical_problem,
)
import pytest

TOL = 1e-5
FLR = 1e-20  # "zero", so if residual below this --> pass regardless of order
CONDUCTIVITY = 1e1
MU = mu_0
freq = 3.16

SIM_TYPES = ["e", "b", "e_hier", "b_hier"]
SRC_TYPES = ["MagDipole", "CircularLoop", "MagDipole_Bfield", "RawVec", "LineCurrent"]
RECEIVER_TYPES = [
    "ElectricField",
    "MagneticFluxDensity",
    "CurrentDensity",
    "MagneticField",
]
RECEIVER_DIRS = ["x", "y", "z"]
RECEIVER_COMPS = ["r", "i"]
ALL_RECEIVERS = list(itertools.product(RECEIVER_TYPES, RECEIVER_DIRS, RECEIVER_COMPS))


class SharedSolves:
    """Memoize ``dpred`` and ``Jvec`` of one simulation.

    ``check_derivative`` draws its step direction from the seed and the size of
    ``x0`` alone, so every receiver's test asks for the same models. Caching the
    full data vector means each model is solved once for all of the receivers.
    """

    def __init__(self, sim):
        self.sim = sim
        self._dpred = {}
        self._jvec = {}

    def dpred(self, m):
        key = m.tobytes()
        if key not in self._dpred:
            self._dpred[key] = self.sim.dpred(m)
        return self._dpred[key]

    def jvec(self, m, v):
        key = (m.tobytes(), v.tobytes())
        if key not in self._jvec:
            self._jvec[key] = self.sim.Jvec(m, v)
        return self._jvec[key]


@pytest.fixture(scope="module", params=SIM_TYPES)
def shared_sim(request):
    """One simulation per formulation, with every source and receiver in it.

    The sources do not couple (each is its own set of columns), so the
    simulation does not need to be built or factored again per source.
    """
    sim_type = request.param
    if "hier" in sim_type:
        sim = get_FDEM_hierarchical_problem(sim_type[0], ALL_RECEIVERS, SRC_TYPES, freq)
    else:
        sim = getFDEMProblem(sim_type, ALL_RECEIVERS, SRC_TYPES, freq)
    assert len(sim.survey.source_list) == len(SRC_TYPES)
    return SharedSolves(sim)


# Previously this only tested "electric fields
@pytest.mark.parametrize("receiver_comp", RECEIVER_COMPS)
@pytest.mark.parametrize("receiver_dir", RECEIVER_DIRS)
@pytest.mark.parametrize("receiver_type", RECEIVER_TYPES)
@pytest.mark.parametrize("src_type", SRC_TYPES)
def test_deriv(shared_sim, src_type, receiver_type, receiver_dir, receiver_comp):
    """Perform derivative test for one source-receiver pair."""
    sim = shared_sim.sim
    src = sim.survey.source_list[SRC_TYPES.index(src_type)]
    rx_index = ALL_RECEIVERS.index((receiver_type, receiver_dir, receiver_comp))
    rx = src.receiver_list[rx_index]
    data_slice = sim.survey.get_slice(src, rx)

    x0 = np.log(np.ones(sim.sigmaMap.nP) * CONDUCTIVITY)  # should work

    def fun(x):
        return (
            shared_sim.dpred(x)[data_slice],
            lambda dx: shared_sim.jvec(x0, dx)[data_slice],
        )

    check_derivative(fun, x0, num=2, random_seed=66, plotIt=False, eps=FLR)


@pytest.mark.parametrize("sim_type", ["e", "b"])
def test_deriv_single_source(sim_type):
    """Derivative test of a simulation with one source and one receiver.

    Unlike the shared simulations above, this has a single source, so the
    fields are stored as a single column.
    """
    rx_type = ("ElectricField", "x", "r")
    sim = getFDEMProblem(sim_type, rx_type, ["MagDipole"], freq)
    x0 = np.log(np.ones(sim.sigmaMap.nP) * CONDUCTIVITY)

    def fun(x):
        return sim.dpred(x), lambda dx: sim.Jvec(x0, dx)

    check_derivative(fun, x0, num=2, random_seed=66, plotIt=False, eps=FLR)
