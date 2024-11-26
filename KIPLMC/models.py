import numpy as np
import jax
import jax.numpy as jnp


"""
All models are implemented as classes with methods for gradient computation. These methods are:

        grad_x(thk, Xk, l, f):
            Computes the gradient with respect to the data X.
        grad_theta(thk, Xk):
            Computes the gradient with respect to the parameters theta.

For simplicity and to allow for further parameters to be introduced, these classes must be initialised with __init__.
"""


class LogisticModel:
    """
        The LogisticModel class implements a logistic regression model for gradient computation.

        Attributes:
            var (float): The variance parameter for the model.
    """

    def __init__(self, var=5):
        self.var = var

    def grad_x(self, thk, Xk, l, f):
        if f is None:
            return
        s = 1 / (1 + np.exp(- np.matmul(f, Xk)))
        return (Xk - thk) / self.var - np.matmul((l - s).transpose(), f).transpose()

    def grad_theta(self, thk, Xk):
        return -np.mean(Xk - thk, axis=1).sum() / self.var


class NormalModel:
    """
        The NormalModel class implements a normal distribution model for gradient computation.
    """

    def __init__(self):
        pass

    def grad_theta(self, thk, Xk):
        return Xk[:, 0].size*(thk - Xk.mean(axis=(0, -1)))

    def grad_x(self, thk, Xk, y, l):
            return 2 * Xk - y - thk


class NeuralNet:
    """
        The NeuralNet class implements a neural network model for gradient computation and performance evaluation.

        Attributes:
            D (int): The dimensionality of the input data.
            Dw (int): The dimensionality of the weights w.
            Dv (int): The dimensionality of the weights v.
            itrain (array): Training data inputs.
            ltrain (array): Training data labels.

        Methods:
            log_pointwise_predrictive_density(w, v, images, labels): Returns the log pointwise predictive density (LPPD) for a set of images and labels.
            test_error(w, v, images, labels): Computes the test error for a set of images and labels.
    """

    def __init__(self, D, Dw, Dv, itrain, ltrain):
        self.D = D
        self.Dw = Dw
        self.Dv = Dv
        self.itrain = itrain
        self.ltrain = ltrain

    def grad_theta(self, thk, Xk, *args):
        a = thk[0]
        b = thk[1]
        w = Xk[:self.D].transpose(1, 0, -1)
        v = Xk[self.D:]
        out = np.array((ave_grad_param(w, a)/self.Dw, ave_grad_param(v, b)/self.Dv))
        return -out.reshape(-1, 1)

    def grad_x(self, thk, Xk, *args):
        a = thk[0].item()
        b = thk[1].item()
        w = Xk[:self.D].transpose(1, 0, -1)
        v = Xk[self.D:]
        l_w = wgrad(w, v, a, b, self.itrain, self.ltrain).transpose(1, 0, -1)
        l_v = vgrad(w, v, a, b, self.itrain, self.ltrain)

        return -np.concatenate((l_w, l_v), axis=0)

    @staticmethod
    def _nn(w, v, image):
        arg = jnp.dot(v, jnp.tanh(jnp.dot(w, image.reshape((28 ** 2)))))
        return jax.nn.softmax(arg)

    def _nn_vec(self, w, v, images):
        return jax.vmap(self._nn, in_axes=(None, None, 0))(w, v, images)

    def _nn_vec_vec(self, w, v, images):
        return jax.vmap(self._nn_vec, in_axes=(2, 2, None), out_axes=2)(w, v, images)

    def log_pointwise_predrictive_density(self, w, v, images, labels):
        """Returns LPPD for set of (test) images and labels."""
        s = self._nn_vec_vec(w, v, images).mean(2)
        return jnp.log(s[jnp.arange(labels.size), labels]).mean()

    def _predict(self, w, v, images):
        s = self._nn_vec_vec(w, v, images).mean(2)
        return jnp.argmax(s, axis=1)

    def test_error(self, w, v, images, labels):
        return jnp.abs(labels - self._predict(w, v, images)).mean()


@jax.jit
def wgrad(w, v, a, b, images, labels):
    grad = jax.grad(_log_density, argnums=0)
    gradv = jax.vmap(grad, in_axes=(2, 2, None, None, None, None), out_axes=2)
    return gradv(w, v, a, b, images, labels)

@jax.jit
def vgrad(w, v, a, b, images, labels):
    grad = jax.grad(_log_density, argnums=1)
    gradv = jax.vmap(grad, in_axes=(2, 2, None, None, None, None), out_axes=2)
    return gradv(w, v, a, b, images, labels)

def _log_density(w, v, a, b, images, labels):
    # Log of model density, vectorized over particles.
    out = _log_prior(w, a) + _log_prior(v, b)
    return out + _log_likelihood(w, v, images, labels)


def _log_nn(w, v, image):
    arg = jnp.dot(v, jnp.tanh(jnp.dot(w, image.reshape((28 ** 2)))))
    return jax.nn.log_softmax(arg)

def _log_nn_vec(w, v, images):
    return jax.vmap(_log_nn, in_axes=(None, None, 0))(w, v, images)


def _log_prior(x, lsig):
    v = x.reshape((x.size))
    sig = jnp.exp(lsig)
    return -jnp.dot(v, v) / (2 * sig ** 2) - x.size * (jnp.log(2 * jnp.pi) / 2 + lsig)


def _log_likelihood(w, v, images, labels):
    return (_log_nn_vec(w, v, images)[jnp.arange(labels.size), labels]).sum()


@jax.jit
def ave_grad_param(w, lsig):
    grad = jax.vmap(_grad_param, in_axes=(2, None))(w, lsig)
    return grad.mean()


def _grad_param(x, lsig):
    # Parameter gradient of one of the two log-priors.
    v = x.reshape((x.size))
    sig = jnp.exp(lsig)
    return jnp.dot(v, v) / (sig ** 2) - x.size
