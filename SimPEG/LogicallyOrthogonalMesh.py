import numpy as np
from BaseMesh import BaseMesh
from DiffOperators import DiffOperators
from utils import mkvc, ndgrid, volTetra, indexCube, faceInfo


class LogicallyOrthogonalMesh(BaseMesh, DiffOperators):  # , LOMGrid
    """
    LogicallyOrthogonalMesh is a mesh class that deals with logically orthogonal meshes.

    """
    def __init__(self, nodes, x0=None):
        assert type(nodes) == list, "'nodes' variable must be a list of np.ndarray"

        for i, nodes_i in enumerate(nodes):
            assert type(nodes_i) == np.ndarray, ("nodes[%i] is not a numpy array." % i)
            assert nodes_i.shape == nodes[0].shape, ("nodes[%i] is not the same shape as nodes[0]" % i)

        assert len(nodes[0].shape) == len(nodes), "Dimension mismatch"
        assert len(nodes[0].shape) > 1, "Not worth using LOM for a 1D mesh."

        super(LogicallyOrthogonalMesh, self).__init__(np.array(nodes[0].shape)-1, x0)

        assert len(nodes[0].shape) == len(self.x0), "Dimension mismatch. x0 != len(h)"

        # Save nodes to private variable _gridN as vectors
        self._gridN = np.ones((nodes[0].size, self.dim))
        for i, node_i in enumerate(nodes):
            self._gridN[:, i] = mkvc(node_i)

    def gridCC():
        doc = "Cell-centered grid."

        def fget(self):
            if self._gridCC is None:
                ccV = (self.nodalVectorAve*mkvc(self.gridN))
                self._gridCC = ccV.reshape((-1, self.dim), order='F')
            return self._gridCC
        return locals()
    _gridCC = None  # Store grid by default
    gridCC = property(**gridCC())

    def gridN():
        doc = "Nodal grid."

        def fget(self):
            if self._gridN is None:
                raise Exception("Someone deleted this. I blame you.")
            return self._gridN
        return locals()
    _gridN = None  # Store grid by default
    gridN = property(**gridN())

    # --------------- Geometries ---------------------
    #
    #
    # ------------------- 2D -------------------------
    #
    #         node(i,j)          node(i,j+1)
    #              A -------------- B
    #              |                |
    #              |    cell(i,j)   |
    #              |        I       |
    #              |                |
    #             D -------------- C
    #         node(i+1,j)        node(i+1,j+1)
    #
    # ------------------- 3D -------------------------
    #
    #
    #             node(i,j,k+1)       node(i,j+1,k+1)
    #                 E --------------- F
    #                /|               / |
    #               / |              /  |
    #              /  |             /   |
    #       node(i,j,k)         node(i,j+1,k)
    #            A -------------- B     |
    #            |    H ----------|---- G
    #            |   /cell(i,j)   |   /
    #            |  /     I       |  /
    #            | /              | /
    #            D -------------- C
    #       node(i+1,j,k)      node(i+1,j+1,k)
    def vol():
        doc = "Construct cell volumes of the 3D model as 1d array."

        def fget(self):
            if(self._vol is None):
                if self.dim == 2:
                    A, B, C, D = indexCube('ABCD', np.array([self.nNx, self.nNy]))
                    normal, area, length = faceInfo(np.c_[self.gridN, np.zeros((self.nN, 1))], A, B, C, D)
                    self._vol = area
                elif self.dim == 3:
                    # Each polyhedron can be decomposed into 5 tetrahedrons
                    # T1 = [A B D E]; % cutted edge
                    # T2 = [B E F G]; % cutted edge
                    # T3 = [B D E G]; % mid
                    # T4 = [B C D G]; % cutted edge
                    # T5 = [D E G H]; % cutted edge
                    A, B, C, D, E, F, G, H = indexCube('ABCDEFGH', np.array([self.nNx, self.nNy, self.nNz]))

                    v1 = volTetra(self.gridN, A, B, D, E)  # cutted edge
                    v2 = volTetra(self.gridN, B, E, F, G)  # cutted edge
                    v3 = volTetra(self.gridN, B, D, E, G)  # mid
                    v4 = volTetra(self.gridN, B, C, D, G)  # cutted edge
                    v5 = volTetra(self.gridN, D, E, G, H)  # cutted edge

                    self._vol = v1 + v2 + v3 + v4 + v5
            return self._vol
        return locals()
    _vol = None
    vol = property(**vol())


if __name__ == '__main__':
    nc = 5
    h1 = np.cumsum(np.r_[0, np.ones(nc)/(nc)])
    h2 = np.cumsum(np.r_[0, np.ones(nc)/(nc)])
    h3 = np.cumsum(np.r_[0, np.ones(nc)/(nc)])
    dee3 = False
    if dee3:
        X, Y, Z = ndgrid(h1, h2, h3, vector=False)
        M = LogicallyOrthogonalMesh([X, Y, Z])
    else:
        X, Y = ndgrid(h1, h2, vector=False)
        M = LogicallyOrthogonalMesh([X, Y])

    # print M.r(M.gridCC, format='M')
    # print M.gridN[:, 0]
    print np.sum(M.vol)
