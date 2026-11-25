"""Derivative and adjoint tests for 3D simulations."""

import pytest
import numpy as np
from discretize import TensorMesh, tests
from simpeg import maps
from simpeg.utils import mkvc, model_builder, get_default_solver
from simpeg.electromagnetics import natural_source as nsem

ADJ_RTOL = 1e-8


@pytest.fixture(scope="module")
def mesh():
    """Return test mesh."""
    return TensorMesh(
        [
            [(200, 6, -1.5), (200.0, 4), (200, 6, 1.5)],
            [(200, 6, -1.5), (200.0, 4), (200, 6, 1.5)],
            [(200, 8, -1.5), (200.0, 8), (200, 8, 1.5)],
        ],
        "CCC",
    )


@pytest.fixture(scope="module")
def active_cells(mesh):
    """Return active cells."""
    return mesh.cell_centers[:, 2] < 0.0


@pytest.fixture(scope="module")
def mapping(mesh, active_cells):
    """Return mapping."""
    return maps.InjectActiveCells(mesh, active_cells, 1e-8) * maps.ExpMap(
        nP=np.sum(active_cells)
    )


@pytest.fixture(scope="module")
def locations():
    """Return receiver locations."""
    elevation = 0.0
    rx_x, rx_y = np.meshgrid(np.arange(-350, 350, 200), np.arange(-350, 350, 200))
    return np.hstack(
        (mkvc(rx_x, 2), mkvc(rx_y, 2), elevation + np.zeros((np.prod(rx_x.shape), 1)))
    )


@pytest.fixture(scope="module")
def frequencies():
    """Return frequencies."""
    return [1e-1, 2e-1]


@pytest.fixture(scope="module")
def sigma_hs(mesh, active_cells):
    """Return background conductivity."""
    sigma_hs = 1e-8 * np.ones(mesh.nC)
    sigma_hs[active_cells] = 1e1
    return sigma_hs


def get_receivers(survey_type, orientations, components, locations):
    """Return the receivers (for one frequency) of a test case."""
    if not isinstance(orientations, list):
        orientations = [orientations]

    if not isinstance(components, list):
        components = [components]

    rx_list = []

    # MT data types (Zxx, Zxy, Zyx, Zyy)
    if survey_type == "impedance":
        for orient in orientations:
            rx_list.extend(
                [
                    nsem.receivers.Impedance(
                        locations,
                        orientation=orient,
                        component=comp,
                    )
                    for comp in components
                ]
            )

    # ZTEM data types (Txx, Tyx, Tzx, Txy, Tyy, Tzy)
    elif survey_type == "tipper":
        for orient in orientations:
            rx_list.extend(
                [
                    nsem.receivers.Tipper(
                        locations_h=locations,
                        locations_base=np.zeros_like(locations),
                        orientation=orient,
                        component=comp,
                    )
                    for comp in components
                ]
            )

    # Admittance data types (Yxx, Yyx, Yzx, Yxy, Yyy, Yzy)
    elif survey_type == "admittance":
        for orient in orientations:
            rx_list.extend(
                [
                    nsem.receivers.Admittance(
                        locations_h=locations,
                        locations_e=np.zeros_like(locations),
                        orientation=orient,
                        component=comp,
                    )
                    for comp in components
                ]
            )

    # MobileMT is app_cond
    elif survey_type == "apparent_conductivity":
        rx_list.extend(
            [
                nsem.receivers.ApparentConductivity(
                    locations_h=locations,
                    locations_e=np.zeros_like(locations),
                    formula=components[0],
                )
            ]
        )

    # Horizontal determinanet
    elif survey_type == "det_horizontal":
        rx_list = [
            nsem.receivers.HorizontalDeterminant(
                locations_h=locations,
                locations_base=np.zeros_like(locations),
                base_type=orientations[0],
                component=comp,
            )
            for comp in components
        ]

    # Determinant ampliutde of the Gram matrix
    elif survey_type == "gram_amp":
        rx_list = [
            nsem.receivers.RootGramDeterminant(
                locations,
                locations_base=np.zeros_like(locations),
                base_type=orientations[0],
            )
        ]

    # Determinant amplitude of the cross product
    elif survey_type == "cross_amp":
        rx_list = [
            nsem.receivers.CrossProductAmplitude(
                locations,
                locations_base=np.zeros_like(locations),
                base_type=orientations[0],
            )
        ]

    return rx_list


CASES_LIST = [
    ("impedance", ["xy", "yx"], ["real", "imag"]),
    ("impedance", ["xx", "yy"], ["real", "imag"]),
    ("impedance", ["xy", "yx"], ["app_res"]),
    ("impedance", ["xx", "yy"], ["app_res"]),
    ("impedance", ["xy", "yx"], ["phase"]),
    ("tipper", ["zx", "zy"], ["real", "imag"]),
    ("tipper", ["xx", "yy"], ["real", "imag"]),
    ("tipper", ["xy", "yx"], ["real", "imag"]),
    ("admittance", ["xy", "yx"], ["real", "imag"]),
    ("admittance", ["xx", "yy"], ["real", "imag"]),
    ("admittance", ["zx", "zy"], ["real", "imag"]),
    ("cross_amp", "electric", None),
    ("cross_amp", "magnetic", None),
    ("gram_amp", "electric", None),
    ("gram_amp", "magnetic", None),
    ("det_horizontal", "electric", ["real", "imag", "amp"]),
    ("det_horizontal", "magnetic", ["real", "imag", "amp"]),
    ("apparent_conductivity", None, "root_gram_determinant"),
    ("apparent_conductivity", None, "cross_product_amplitude"),
    ("apparent_conductivity", None, "horizontal_determinant"),
]


class SharedSimulation:
    """A simulation whose survey holds the receivers of every test case.

    Each case only changes the receivers, so the simulation is built once and
    a case works with its own slice of the data.
    """

    def __init__(self, mesh, mapping, sigma_hs, locations, frequencies, m0):
        self.m0 = m0
        source_list = []
        self._receivers = []  # [source][case] -> list of receivers
        for f in frequencies:
            rx_per_case = [get_receivers(*case, locations) for case in CASES_LIST]
            self._receivers.append(rx_per_case)
            rx_list = [rx for rx_case in rx_per_case for rx in rx_case]
            source_list.append(nsem.sources.PlanewaveXYPrimary(rx_list, f))
        self.survey = nsem.survey.Survey(source_list)

        self.sim = nsem.simulation.Simulation3DPrimarySecondary(
            mesh,
            survey=self.survey,
            sigmaMap=mapping,
            sigmaPrimary=sigma_hs,
            solver=get_default_solver(),
        )
        self._dpred = {}
        self._jvec = {}

    def data_inds(self, case):
        """Indices of a case's data in the full data vector."""
        case_index = CASES_LIST.index(case)
        inds = []
        for src, rx_per_case in zip(self.survey.source_list, self._receivers):
            for rx in rx_per_case[case_index]:
                slc = self.survey.get_slice(src, rx)
                inds.append(np.arange(slc.start, slc.stop))
        return np.concatenate(inds)

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


@pytest.fixture(scope="module")
def m0(mesh, active_cells):
    """Return the model to linearize about."""
    n_active = np.sum(active_cells)
    rng = np.random.default_rng(4412)

    m0 = np.log(1e1) * np.ones(n_active)
    ind = model_builder.get_indices_block(
        np.r_[-200.0, -200.0, -600.0],
        np.r_[200.0, 200.0, -200.0],
        mesh.cell_centers[active_cells, :],
    )
    m0[ind] = np.log(1e0)
    m0 += 0.01 * rng.uniform(low=-1, high=1, size=n_active)
    return m0


@pytest.fixture(scope="module")
def shared_sim(mesh, mapping, sigma_hs, locations, frequencies, m0):
    """Simulation for the derivative tests (its fields change with the model)."""
    return SharedSimulation(mesh, mapping, sigma_hs, locations, frequencies, m0)


@pytest.fixture(scope="module")
def adjoint_sim(mesh, mapping, sigma_hs, locations, frequencies, m0):
    """Simulation and fields at `m0` for the adjoint tests.

    This is kept apart from `shared_sim` so that nothing else solves with it
    and replaces the factorizations that belong to its stored fields.
    """
    shared = SharedSimulation(mesh, mapping, sigma_hs, locations, frequencies, m0)
    return shared, shared.sim.fields(m0)


@pytest.mark.parametrize("survey_type, orientations, components", CASES_LIST)
class TestDerivatives:
    """Perform derivative convergence test."""

    def test_misfit(self, survey_type, orientations, components, shared_sim):
        """Test data misfit."""
        inds = shared_sim.data_inds((survey_type, orientations, components))

        passed = tests.check_derivative(
            lambda m: (
                shared_sim.dpred(m)[inds],
                lambda mx: shared_sim.jvec(m, mx)[inds],
            ),
            shared_sim.m0,
            plotIt=False,
            num=3,
            random_seed=412,
        )

        assert passed

    def test_adjoint(self, survey_type, orientations, components, adjoint_sim):
        """Perform adjoint test."""
        shared, f = adjoint_sim
        sim, m0 = shared.sim, shared.m0
        inds = shared.data_inds((survey_type, orientations, components))

        def jtvec(v):
            # J^T applied to a vector that is zero outside of this case's data
            full = np.zeros(shared.survey.nD, dtype=v.dtype)
            full[inds] = v
            return sim.Jtvec(m0, full, f=f)

        tests.assert_isadjoint(
            lambda u: sim.Jvec(m0, u, f=f)[inds],
            jtvec,
            m0.shape,
            (len(inds),),
            rtol=ADJ_RTOL,
            random_seed=32,
        )
