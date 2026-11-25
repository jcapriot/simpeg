"""Derivative tests for FDEM problems with a non-uniform conductivity."""

import itertools

import numpy as np
import pytest
from simpeg import tests
from simpeg.electromagnetics.utils.testing_utils import getFDEMProblem

from tests.utils.shared_solves import SharedSolves

FLR = 1e-20  # "zero", so if residual below this --> pass regardless of order
CONDUCTIVITY = 1e1
freq = 3.16

SIM_TYPES = ["e", "b", "j", "h"]
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


@pytest.fixture(scope="module", params=SIM_TYPES)
def shared_sim(request):
    """One simulation per formulation, with every source and receiver in it."""
    sim = getFDEMProblem(request.param, ALL_RECEIVERS, SRC_TYPES, freq)

    rng = np.random.default_rng(seed=42)
    x0 = np.log(np.ones(sim.sigmaMap.nP) * CONDUCTIVITY)
    x0 = x0 + rng.normal(size=sim.sigmaMap.nP) * np.log(CONDUCTIVITY) * 1e-1
    return SharedSolves(sim), x0


@pytest.mark.parametrize("receiver_comp", RECEIVER_COMPS)
@pytest.mark.parametrize("receiver_dir", RECEIVER_DIRS)
@pytest.mark.parametrize("receiver_type", RECEIVER_TYPES)
def test_deriv(shared_sim, receiver_type, receiver_dir, receiver_comp):
    """Derivative test of one receiver, over the data from every source."""
    shared, x0 = shared_sim
    survey = shared.sim.survey
    rx_index = ALL_RECEIVERS.index((receiver_type, receiver_dir, receiver_comp))
    data_inds = np.concatenate(
        [
            np.arange(
                *survey.get_slice(src, src.receiver_list[rx_index]).indices(survey.nD)
            )
            for src in survey.source_list
        ]
    )

    def fun(x):
        return (
            shared.dpred(x)[data_inds],
            lambda dx: shared.jvec(x0, dx)[data_inds],
        )

    assert tests.check_derivative(
        fun, x0, num=2, plotIt=False, eps=FLR, random_seed=6521
    )
