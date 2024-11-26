import numpy as np
from KIPLMC.base import IPS


class MPGD(IPS):
    def __init__(self, g, h, N, D, model, **kwargs):
        super().__init__(g, h, N, D, model, **kwargs)

        self.psi_0 = np.exp(-g * h)
        self.psi_1 = -1 / g * self.psi_0 + 1 / g
        self.psi_2 = h / g - 1 / g * self.psi_1

        C = np.array([[(1 - self.psi_0 ** 2) / (2 * g), (-self.psi_0 + self.psi_0 ** 2 / 2 + 1 / 2) / g ** 2],
                      [(-self.psi_0 + self.psi_0 ** 2 / 2 + 1 / 2) / g ** 2,
                       (h + 2 / g * self.psi_0 - self.psi_0 ** 2 / (2 * g) - 3 / (
                               2 * g)) / g ** 2]])  # covariance matrix
        self.L = np.linalg.cholesky(C)  # Cholesky matrix

    def update(self, l, f, thk, Zk, Xk, Vk):
        B2 = np.sqrt(2 * self.g) * np.matmul(self.L, np.random.normal(size=tuple(Xk.shape) + (2,))[..., None]).squeeze()

        th_ = thk + self.psi_1 * Zk - self.psi_2 * self.grad_theta(thk + self.psi_1 * Zk, Xk)
        Z_ = self.psi_0 * Zk - self.psi_1 * self.grad_theta(thk + self.psi_1 * Zk, Xk)
        X_ = Xk + self.psi_1 * Vk - self.psi_2 * self.grad_x(th_, Xk, l, f) + B2[..., 0]
        V_ = self.psi_0 * Vk - self.psi_1 * self.grad_x(th_, Xk, l, f) + B2[..., 1]

        return th_, Z_, X_, V_


class MPGD_nobar(IPS):
    def __init__(self, g, h, N, D, model, **kwargs):
        super().__init__(g, h, N, D, model, **kwargs)

    def update(self, l, f, thk, Zk, Xk, Vk):
        th_ = thk + self.h * Zk
        Z_ = Zk - self.h * (self.g * Zk + self.grad_theta(thk, Xk))
        X_ = Xk + self.h * Vk
        V_ = (Vk - self.h * (self.g * Vk + self.grad_x(thk, Xk, l, f))
              + np.sqrt(2 * self.g * self.h) * np.random.normal(size=tuple(Vk.shape)))
        return th_, Z_, X_, V_


class SOUL(IPS):
    def __init__(self, g, h, N, D, model, M):
        super().__init__(g, h, N, D, model)
        self.M = M

    def update(self, l, f, thk, Zk, Xk, Vk):
        X_ = Xk
        Xn = Xk.reshape(self.D, self.N)
        for n in range(self.M):
            X_ += -self.h * self.grad_x(thk, X_, l, f) + np.sqrt(2*self.h) * np.random.normal(0, 1, Xk.shape)
            Xn = np.append(Xn, X_)
        th_ = thk - self.h * self.grad_theta(thk, Xn)
        return th_, Zk, X_, Vk


class IPLA(IPS):
    def __init__(self, g, h, N, D, model):
        super().__init__(g, h, N, D, model)

    def update(self, l, f, thk, Zk, Xk, Vk):
        th_ = thk - self.h * self.grad_theta(thk, Xk) + np.sqrt(2*self.h/self.N) * np.random.normal(0, 1, thk.shape)
        X_ = Xk - self.h * self.grad_x(thk, Xk, l, f) + np.sqrt(2 * self.h) * np.random.normal(0, 1, Xk.shape)

        return th_, Zk, X_, Vk
