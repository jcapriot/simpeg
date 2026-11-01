"""Derivative and adjoint tests."""

import pytest
import numpy as np
import discretize
from simpeg import maps, tests
from simpeg.utils import get_default_solver
from simpeg.electromagnetics import time_domain as tdem

from tests.utils.shared_solves import SharedSolves

plotIt = False

ADJ_ATOL = 1e-11
ADJ_RTOL = 2e-5


@pytest.fixture(scope="module")
def mesh():
    """Return test mesh."""
    cs = 10.0
    ncx = 4
    ncy = 4
    ncz = 4
    npad = 2
    # hx = [(cs, ncx), (cs, npad, 1.3)]
    # hz = [(cs, npad, -1.3), (cs, ncy), (cs, npad, 1.3)]
    return discretize.TensorMesh(
        [
            [(cs, npad, -1.5), (cs, ncx), (cs, npad, 1.5)],
            [(cs, npad, -1.5), (cs, ncy), (cs, npad, 1.5)],
            [(cs, npad, -1.5), (cs, ncz), (cs, npad, 1.5)],
        ],
        "CCC",
    )


@pytest.fixture(scope="module")
def locations():
    """Return receiver locations."""
    return np.array([[15, 0.0, -1e-2]])


@pytest.fixture(scope="module")
def times():
    """Return receiver times."""
    return np.logspace(-4, -3, 20)


def get_survey(receiver_type, orientation, locations, times):
    """Return test survey."""
    return get_multi_receiver_survey([(receiver_type, orientation)], locations, times)


def get_multi_receiver_survey(receivers, locations, times):
    """Return test survey with every (receiver type, orientation) at both sources.

    Each source gets its own receiver objects.
    """
    source_list = []
    for source_location in [np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 8.0])]:
        rx_list = [
            getattr(tdem.receivers, "Point" + receiver_type)(
                locations=locations, times=times, orientation=orientation
            )
            for receiver_type, orientation in receivers
        ]
        source_list.append(tdem.sources.MagDipole(rx_list, location=source_location))

    return tdem.Survey(source_list)


def get_simulation(simulation_class, mesh, survey):
    """Return test simulation."""
    sim = getattr(tdem, simulation_class)(mesh, survey=survey)

    if "Hierarchical" in simulation_class:
        sigma_map, tau_map, kappa_map = get_wire_mappings(mesh)
        sim.sigmaMap = sigma_map
        sim.face_conductance_map = tau_map
        sim.edge_area_conductance_map = kappa_map
    else:
        sigma_map = get_sigma_mapping(mesh)
        sim.sigmaMap = sigma_map

    sim.solver = get_default_solver()
    sim.time_steps = [(1e-05, 10), (5e-05, 10), (2.5e-4, 10)]

    return sim


def get_sigma_mapping(mesh):
    """Return mapping for sigma."""
    active = mesh.cell_centers_z < 0.0
    activeMap = maps.InjectActiveCells(
        mesh, active, np.log(1e-8), nC=mesh.shape_cells[2]
    )
    return maps.ExpMap(mesh) * maps.SurjectVertical1D(mesh) * activeMap


def get_wire_mappings(mesh):
    """Return mapping for cell, faces and edges."""
    active_cells = mesh.cell_centers[:, -1] < 0.0
    active_faces = mesh.faces[:, -1] < 0.0
    active_edges = mesh.edges[:, -1] < 0.0
    n_active_cells = np.sum(active_cells)
    n_active_faces = np.sum(active_faces)
    n_active_edges = np.sum(active_edges)

    wire_map = maps.Wires(
        ("log_sigma", n_active_cells),
        ("log_tau", n_active_faces),
        ("log_kappa", n_active_edges),
    )

    sigma_map = (
        maps.InjectActiveCells(mesh, active_cells, 1e-8)
        * maps.ExpMap(nP=n_active_cells)
        * wire_map.log_sigma
    )
    tau_map = (
        maps.InjectActiveFaces(mesh, active_faces, 0)
        * maps.ExpMap(nP=n_active_faces)
        * wire_map.log_tau
    )
    kappa_map = (
        maps.InjectActiveEdges(mesh, active_edges, 0)
        * maps.ExpMap(nP=n_active_edges)
        * wire_map.log_kappa
    )

    return sigma_map, tau_map, kappa_map


CASES_LIST = [
    ("Simulation3DElectricField", "MagneticFluxTimeDerivative", "z"),
    ("Simulation3DElectricField", "MagneticFluxTimeDerivative", "x"),
    ("Simulation3DElectricField", "ElectricField", "y"),
    ("Simulation3DElectricField", "MagneticFieldTimeDerivative", "z"),
    ("Simulation3DElectricField", "MagneticFieldTimeDerivative", "x"),
    ("Simulation3DElectricField", "CurrentDensity", "y"),
    ("Simulation3DHierarchicalElectricField", "MagneticFluxTimeDerivative", "z"),
    ("Simulation3DHierarchicalElectricField", "MagneticFluxTimeDerivative", "x"),
    ("Simulation3DHierarchicalElectricField", "ElectricField", "y"),
    ("Simulation3DHierarchicalElectricField", "MagneticFieldTimeDerivative", "z"),
    ("Simulation3DHierarchicalElectricField", "MagneticFieldTimeDerivative", "x"),
    ("Simulation3DHierarchicalElectricField", "CurrentDensity", "y"),
    ("Simulation3DMagneticFluxDensity", "MagneticFluxTimeDerivative", "z"),
    ("Simulation3DMagneticFluxDensity", "MagneticFluxTimeDerivative", "x"),
    ("Simulation3DMagneticFluxDensity", "MagneticFluxDensity", "z"),
    ("Simulation3DMagneticFluxDensity", "MagneticFluxDensity", "x"),
    ("Simulation3DMagneticFluxDensity", "MagneticFieldTimeDerivative", "z"),
    ("Simulation3DMagneticFluxDensity", "MagneticFieldTimeDerivative", "x"),
    ("Simulation3DMagneticFluxDensity", "MagneticField", "z"),
    ("Simulation3DMagneticFluxDensity", "MagneticField", "x"),
    ("Simulation3DMagneticFluxDensity", "CurrentDensity", "y"),
    ("Simulation3DHierarchicalMagneticFluxDensity", "MagneticFluxDensity", "z"),
    ("Simulation3DHierarchicalMagneticFluxDensity", "MagneticFluxDensity", "x"),
    ("Simulation3DHierarchicalMagneticFluxDensity", "MagneticFluxTimeDerivative", "z"),
    ("Simulation3DHierarchicalMagneticFluxDensity", "MagneticFluxTimeDerivative", "x"),
    ("Simulation3DHierarchicalMagneticFluxDensity", "MagneticField", "z"),
    ("Simulation3DHierarchicalMagneticFluxDensity", "MagneticField", "x"),
    ("Simulation3DHierarchicalMagneticFluxDensity", "MagneticFieldTimeDerivative", "z"),
    ("Simulation3DHierarchicalMagneticFluxDensity", "MagneticFieldTimeDerivative", "x"),
    ("Simulation3DHierarchicalMagneticFluxDensity", "CurrentDensity", "y"),
    ("Simulation3DMagneticField", "MagneticField", "z"),
    ("Simulation3DMagneticField", "MagneticField", "x"),
    ("Simulation3DMagneticField", "MagneticFieldTimeDerivative", "z"),
    ("Simulation3DMagneticField", "MagneticFieldTimeDerivative", "x"),
    ("Simulation3DMagneticField", "MagneticFluxDensity", "z"),
    ("Simulation3DMagneticField", "MagneticFluxDensity", "x"),
    ("Simulation3DMagneticField", "MagneticFluxTimeDerivative", "z"),
    ("Simulation3DMagneticField", "MagneticFluxTimeDerivative", "x"),
    ("Simulation3DMagneticField", "ElectricField", "y"),
    ("Simulation3DMagneticField", "CurrentDensity", "y"),
    ("Simulation3DCurrentDensity", "MagneticFieldTimeDerivative", "z"),
    ("Simulation3DCurrentDensity", "MagneticFieldTimeDerivative", "x"),
    ("Simulation3DCurrentDensity", "MagneticFluxTimeDerivative", "z"),
    ("Simulation3DCurrentDensity", "MagneticFluxTimeDerivative", "x"),
    ("Simulation3DCurrentDensity", "CurrentDensity", "y"),
]


CASES_BY_CLASS = {}
for _class, _rx_type, _orientation in CASES_LIST:
    CASES_BY_CLASS.setdefault(_class, []).append((_rx_type, _orientation))


class SharedSimulation:
    """A simulation of one class that has every receiver of that class's cases.

    The cases of a simulation class only differ by their receivers, so the
    simulation is built once and each case works with its slice of the data.
    """

    def __init__(self, simulation_class, mesh, locations, times):
        self.cases = CASES_BY_CLASS[simulation_class]
        self.survey = get_multi_receiver_survey(self.cases, locations, times)
        self.sim = get_simulation(simulation_class, mesh, self.survey)

        rng = np.random.default_rng(1234)
        if "Hierarchical" in simulation_class:
            n_sigma = np.sum(mesh.cell_centers[:, -1] < 0.0)
            n_tau = np.sum(mesh.faces[:, -1] < 0.0)
            n_kappa = np.sum(mesh.edges[:, -1] < 0.0)
            self.m0 = np.r_[
                np.log(1e-1) * np.ones(n_sigma) + 1e-3 * rng.normal(size=n_sigma),
                np.log(10 * 1e-1) * np.ones(n_tau) + 1e-3 * rng.normal(size=n_tau),
                np.log(100 * 1e-1) * np.ones(n_kappa) + 1e-3 * rng.normal(size=n_kappa),
            ]
        else:
            n_param = self.sim.sigmaMap.nP
            self.m0 = np.log(1e-1) * np.ones(n_param) + 1e-3 * rng.normal(size=n_param)

        self.solves = SharedSolves(self.sim)
        self._jvec_with_fields = {}

    def jvec_with_fields(self, v, f):
        """Cached ``Jvec`` (tests that ask for the same vector share it)."""
        key = v.tobytes()
        if key not in self._jvec_with_fields:
            self._jvec_with_fields[key] = self.sim.Jvec(self.m0, v, f=f)
        return self._jvec_with_fields[key]

    def data_inds(self, receiver_type, orientation):
        """Indices of a case's data in the full data vector."""
        case_index = self.cases.index((receiver_type, orientation))
        inds = []
        for src in self.survey.source_list:
            slc = self.survey.get_slice(src, src.receiver_list[case_index])
            inds.append(np.arange(slc.start, slc.stop))
        return np.concatenate(inds)


class SimulationCache(dict):
    """Build the simulation of a class the first time a test asks for it."""

    def __init__(self, factory):
        super().__init__()
        self.factory = factory

    def __missing__(self, simulation_class):
        self[simulation_class] = self.factory(simulation_class)
        return self[simulation_class]


@pytest.fixture(scope="module")
def shared_sims(mesh, locations, times):
    """Simulations for the derivative tests."""
    return SimulationCache(
        lambda simulation_class: SharedSimulation(
            simulation_class, mesh, locations, times
        )
    )


@pytest.fixture(scope="module")
def adjoint_sims(mesh, locations, times):
    """Simulations, and their fields at `m0`, for the adjoint tests.

    These are kept apart from the simulations of the derivative tests, so that
    nothing else solves with them and replaces what the stored fields need.
    """

    def build(simulation_class):
        shared = SharedSimulation(simulation_class, mesh, locations, times)
        return shared, shared.sim.fields(shared.m0)

    return SimulationCache(build)


@pytest.mark.parametrize("simulation_class, receiver_type, orientation", CASES_LIST)
class TestDerivatives:
    """Derivative and adjoint test for primary-secondary formulation."""

    def test_misfit(self, simulation_class, receiver_type, orientation, shared_sims):
        """Test derivative."""
        shared = shared_sims[simulation_class]
        inds = shared.data_inds(receiver_type, orientation)

        passed = tests.check_derivative(
            lambda m: (
                shared.solves.dpred(m)[inds],
                lambda mx: shared.solves.jvec(shared.m0, mx)[inds],
            ),
            shared.m0,
            plotIt=False,
            random_seed=42,
            num=3,
        )

        assert passed

    def test_adjoint(self, simulation_class, receiver_type, orientation, adjoint_sims):
        """Test adjoint."""
        shared, f = adjoint_sims[simulation_class]
        sim, m0 = shared.sim, shared.m0
        inds = shared.data_inds(receiver_type, orientation)

        def jtvec(v):
            # J^T applied to a vector that is zero outside of this case's data
            full = np.zeros(shared.survey.nD, dtype=v.dtype)
            full[inds] = v
            return sim.Jtvec(m0, full, f=f)

        tests.assert_isadjoint(
            lambda u: shared.jvec_with_fields(u, f)[inds],
            jtvec,
            m0.shape,
            (len(inds),),
            rtol=ADJ_RTOL,
            atol=ADJ_ATOL,
            random_seed=44,
        )


def _same_factors(sim, factors):
    """Check that a simulation holds exactly these factor objects."""
    stored = sim._stored_factors
    return stored.keys() == factors.keys() and all(
        stored[key] is factor for key, factor in factors.items()
    )


@pytest.mark.parametrize(
    "simulation_class",
    [
        "Simulation3DElectricField",
        "Simulation3DMagneticFluxDensity",
        "Simulation3DMagneticField",
        "Simulation3DCurrentDensity",
    ],
)
def test_store_factors(simulation_class, mesh, locations, times):
    """Test that storing the factorizations does not change the results."""
    rng = np.random.default_rng(0)
    survey = get_survey("MagneticFluxTimeDerivative", "z", locations, times)
    sim_store = get_simulation(simulation_class, mesh, survey)
    sim_store.store_factors = True
    sim_plain = get_simulation(simulation_class, mesh, survey)

    n_param = sim_plain.sigmaMap.nP
    m = np.log(1e-1) * np.ones(n_param) + 1e-2 * rng.normal(size=n_param)
    v = rng.normal(size=n_param)
    w = rng.normal(size=survey.nD)

    f_store = sim_store.fields(m)
    f_plain = sim_plain.fields(m)

    # time steps repeat, so there are fewer factors than steps
    factors = dict(sim_store._stored_factors)
    assert 0 < len(factors) <= 3

    np.testing.assert_allclose(
        sim_store.dpred(m, f=f_store), sim_plain.dpred(m, f=f_plain)
    )
    np.testing.assert_allclose(
        sim_store.Jvec(m, v, f=f_store), sim_plain.Jvec(m, v, f=f_plain)
    )
    np.testing.assert_allclose(
        sim_store.Jtvec(m, w, f=f_store), sim_plain.Jtvec(m, w, f=f_plain)
    )
    # Jvec and the transposed solves in Jtvec reuse the very same factors
    assert _same_factors(sim_store, factors)

    # the same model keeps the factors, even when fields is called again
    sim_store.model = m.copy()
    assert _same_factors(sim_store, factors)
    sim_store.fields(m)
    assert _same_factors(sim_store, factors)

    # a new model drops them, and new ones get built
    sim_store.model = m + 0.1
    assert not hasattr(sim_store, "_stored_factors")
    sim_store.fields(m + 0.1)
    assert factors.keys() == sim_store._stored_factors.keys()
    assert all(sim_store._stored_factors[k] is not factors[k] for k in factors)

    # turning it off frees everything
    sim_store.store_factors = False
    assert len(sim_store._stored_factors) == 0
