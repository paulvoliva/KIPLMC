import torch
import numpy as np
from abc import abstractmethod


class IPS:
    """
        The IPS (Interacting Particle System) class is a base class for implementing various particle-based algorithms.

        Attributes:
            g (float): The friction parameter.
            h (float): The step-size parameter.
            N (int): The number of particles.
            D (int): The dimensionality of the particles.
            model (object): The model object that contains the gradient functions.
            report (callable, optional): A function to report the progress of the algorithm.
            errs (list): A list to store error metrics during the algorithm's execution.

        Methods:
            run(K, l=None, f=None, X=None, V=None, Z=None, th=None, store_latent=False):
                Runs the particle system for K iterations.
            grad_x (callable): A function to compute the gradient with respect to x.
            grad_theta (callable): A function to compute the gradient with respect to theta.
            update(l, f, thk, Zk, Xk, Vk): Abstract method to update the particle states.
            initialiser(*args, shape=None): Static method to initialize particle states.
    """

    def __init__(self, g, h, N, D, model, report=None):
        self.g = g
        self.h = h
        self.N = N
        self.D = D

        self.grad_x = model.grad_x
        self.grad_theta = model.grad_theta

        self.report = report
        self.errs = []

    def run(self, K, l=None, f=None, X=None, V=None, Z=None, th=None, store_latent=False):
        X, V = self.initialiser(X, V, shape=(self.D, self.N))
        th, Z = self.initialiser(th, Z, shape=(1, 1))

        # List of variable names
        var_names = ['th', 'Z', 'X', 'V']

        for k in range(K):
            Xk = X[..., -self.N:]
            Vk = V[..., -self.N:]
            thk = th[..., -1:]
            Zk = Z[..., -1:]

            out_ = self.update(l, f, thk, Zk, Xk, Vk)

            if store_latent:
                # Loop over the variable names and modify them directly
                for i, var_name in enumerate(var_names):
                    if type(out_[i]) is torch.Tensor:
                        locals()[var_name] = torch.cat((locals()[var_name], out_[i]), dim=-1)
                    else:
                        locals()[var_name] = np.append(locals()[var_name], out_[i], axis=-1)
            else:
                if type(out_[0]) is torch.Tensor:
                    th = torch.cat((th, out_[0]), dim=-1)
                else:
                    th = np.append(th, out_[0], axis=-1)
                X = out_[2]
                V = out_[3]
                Z = out_[1]

            if self.report:
                out = self.report(thk, Zk, Xk, Vk, k)
                self.errs.append(list(out))

        return th, Z, X, V

    @abstractmethod
    def update(self, l, f, thk, Zk, Xk, Vk):
        pass

    @staticmethod
    def initialiser(*args, shape=None):
        if len(args) == 1:
            return np.zeros(shape) if args[0] is None else args[0]
        return [np.zeros(shape) if arg is None else arg for arg in args]