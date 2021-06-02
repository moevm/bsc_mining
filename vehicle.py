import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot

class VehicleSimulator:
    def __init__(self, x, y, v, max_v, yaw, a, omega, w, L, d):
        self.x = float(x)
        self.y = float(y)
        self.yaw = np.deg2rad(float(yaw))
        self.v = float(v)
        self.a = float(a)
        self.omega = float(omega)
        self.max_v = float(max_v)
        self.W = float(w)
        self.L = float(L)
        self.d = d
        self.n_edges = []
        self.edges = []
        self._calc_vehicle_contour()
        self.calc_global_contour()

    def update(self, dt):
        self.x += self.v * np.cos(self.yaw) * dt
        self.y += self.v * np.sin(self.yaw) * dt
        self.yaw += self.omega * dt
        self.v += self.a * dt
        if self.v >= self.max_v:
            self.v = self.max_v
        self.calc_global_contour()

    def plot(self):
        
        plt.plot(self.x, self.y, ".b")

        # convert global coordinate
        gx, gy = self.calc_global_contour()
        plt.plot(gx, gy, color="k")

    def calc_global_contour(self):
        rot = Rot.from_euler('z', np.deg2rad(90) - self.yaw).as_matrix()[0:2, 0:2]
        gxy = np.stack([self.vc_x, self.vc_y]).T @ rot
        gx = gxy[:, 0] + self.x
        gy = gxy[:, 1] + self.y

        coords = np.stack([gx, gy], -1)

        edges = [[], [], [], []]
        edges[0] = coords[:self.n_edges[0]]
        edges[1] = coords[self.n_edges[0]:self.n_edges[1]]
        edges[2] = coords[self.n_edges[1]:self.n_edges[2]]
        edges[3] = coords[self.n_edges[2]:self.n_edges[3]]

        d_cr = [np.hypot(*edges[0][0]), np.hypot(*edges[1][0]), np.hypot(*edges[2][0]), np.hypot(*edges[3][0])]
        
        far_cr = d_cr.index(max(d_cr))

        c1 = (far_cr + 1) % 4
        c2 = (far_cr + 2) % 4

        self.visible_coords = np.vstack([edges[c1], edges[c2]])

        return gx, gy

    def _calc_vehicle_contour(self):

        self.vc_x = []
        self.vc_y = []

        self.vc_x.append(self.L / 2.0)
        self.vc_y.append(self.W / 2.0)

        self.vc_x.append(self.L / 2.0)
        self.vc_y.append(-self.W / 2.0)

        self.vc_x.append(-self.L / 2.0)
        self.vc_y.append(-self.W / 2.0)

        self.vc_x.append(-self.L / 2.0)
        self.vc_y.append(self.W / 2.0)

        self.vc_x.append(self.L / 2.0)
        self.vc_y.append(self.W / 2.0)

        self.vc_x, self.vc_y = self._interpolate(self.vc_x, self.vc_y)

    def _interpolate(self, x, y):
        rx, ry = [], []
        d_theta = min(self.d / self.W, self.d / self.L)
        for i in range(len(x) - 1):
            rx.extend([(1.0 - theta) * x[i] + theta * x[i + 1]
                       for theta in np.arange(0.0, 1.0, d_theta)])
            ry.extend([(1.0 - theta) * y[i] + theta * y[i + 1]
                       for theta in np.arange(0.0, 1.0, d_theta)])
            self.n_edges.append(len(rx))

        rx.extend([(1.0 - theta) * x[len(x) - 1] + theta * x[1]
                   for theta in np.arange(0.0, 1.0, d_theta)])
        ry.extend([(1.0 - theta) * y[len(y) - 1] + theta * y[1]
                   for theta in np.arange(0.0, 1.0, d_theta)])

        return rx, ry
