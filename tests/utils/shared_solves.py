"""Share the solves of one simulation between many derivative tests."""


class SharedSolves:
    """Memoize ``dpred`` and ``Jvec`` of one simulation.

    ``check_derivative`` draws its step direction from the seed and the size of
    ``x0`` alone, so tests that use the same seed ask for the same models.
    Caching the full data vector means each model is solved once, and every
    test (one per receiver for example) can check its own slice of it.
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
