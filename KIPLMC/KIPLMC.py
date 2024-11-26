import numpy as np
from KIPLMC.base import IPS


class KIPLMC1(IPS):
    def __init__(self, g, h, N, D, model, **kwargs):
        super().__init__(g, h, N, D, model, **kwargs)

        self.psi_0 = np.exp(-g * h)
        self.psi_1 = -1 / g * self.psi_0 + 1 / g
        self.psi_2 = h / g - 1 / g * self.psi_1

        C = np.array([[(1 - self.psi_0 ** 2) / (2 * g), (-self.psi_0 + self.psi_0 ** 2 / 2 + 1 / 2) / g ** 2],
                      [(-self.psi_0 + self.psi_0 ** 2 / 2 + 1 / 2) / g ** 2,
                       (h + 2 / g * self.psi_0 - self.psi_0 ** 2 / (2 * g) - 3 / (
                               2 * g)) / g ** 2]])  # covariance matrix
        self.L = np.linalg.cholesky(C)              # Cholesky matrix

    def update(self, l, f, thk, Zk, Xk, Vk):
        B1 = (np.sqrt(2 * self.g / self.N) *
              np.matmul(self.L, np.random.normal(size=tuple(thk.shape)+(2,))[..., None]).squeeze())
        B2 = np.sqrt(2 * self.g) * np.matmul(self.L, np.random.normal(size=tuple(Xk.shape)+(2,))[..., None]).squeeze()

        th_ = thk + self.psi_1 * Zk - self.psi_2 * self.grad_theta(thk, Xk) + B1[..., 0]
        Z_ = self.psi_0 * Zk - self.psi_1 * self.grad_theta(thk, Xk) + B1[..., 1]
        X_ = Xk + self.psi_1 * Vk - self.psi_2 * self.grad_x(thk, Xk, l, f) + B2[..., 0]
        V_ = self.psi_0 * Vk - self.psi_1 * self.grad_x(thk, Xk, l, f) + B2[..., 1]

        return th_, Z_, X_, V_


class KIPLMC2(IPS):
    def __init__(self, g, h, N, D, model, **kwargs):
        super().__init__(g, h, N, D, model, **kwargs)
        self.d = np.exp(-self.h * self.g/2)
        self.d_ = np.sqrt(1 - self.d ** 2)

    def update(self, l, f, thk, Zk, Xk, Vk):
        brownmot1 = np.sqrt(2) * np.random.normal(0, 1, tuple(thk.shape))
        brownmot2 = np.sqrt(2) * np.random.normal(0, 1, tuple(Xk.shape))
        g_th1 = self.grad_theta(thk, Xk)
        g_x1 = self.grad_x(thk, Xk, l, f)
        th_ = thk + self.h * (self.d * Zk + self.d_ / np.sqrt(self.N) * brownmot1) - self.h ** 2 / 2 * g_th1
        X_ = Xk + self.h * (self.d * Vk + self.d_ * brownmot2) - self.h ** 2 / 2 * g_x1

        g_th2 = self.grad_theta(th_, X_)
        g_x2 = self.grad_x(th_, X_, l, f)
        Z_ = (self.d ** 2 * Zk - self.d * self.h / 2 * (g_th1 + g_th2) + self.d_ / np.sqrt(self.N) *
              (self.d * brownmot1 - np.random.normal(0, 1, tuple(thk.shape))))
        V_ = (self.d ** 2 * Vk - self.d * self.h / 2 * (g_x1 + g_x2) + np.sqrt(self.h) * self.d_ *
              (self.d * brownmot2 - np.random.normal(0, 1, tuple(Xk.shape))))

        return th_, Z_, X_, V_