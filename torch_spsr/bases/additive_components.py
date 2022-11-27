import math
import sympy as sp
import torch
from abc import ABC, abstractmethod


class AdditiveComponent(ABC):
    """
    Abstract base class for all elementary functions q_u (x)
        You can inherent this class for more functions -- integrals are automatically pre-computed using sympy
    """
    def __init__(self):
        pass

    @abstractmethod
    def __repr__(self):
        """
        Used to determine the saved file name for pre-computed basis
        """
        pass

    def visualize(self):
        x = sp.symbols('x')
        sp.plot(self.symbolic(x), (x, -3, 3))

    @abstractmethod
    def symbolic(self, x):
        pass

    @abstractmethod
    def evaluate(self, xyz):
        """
        Evaluate the function
        :param xyz: torch.Tensor, supposed to be arbitrary shape
            Note that the function still maps from 1D to 1D, there should be no interactions among x,y,z!
        :return: torch.Tensor, same shape as xyz
        """
        pass

    @abstractmethod
    def evaluate_derivative(self, xyz):
        """
        Evaluate gradient of the function w.r.t. the input
        :param xyz: torch.Tensor, supposed to be arbitrary shape
        :return: torch.Tensor, same shape as xyz
        """
        pass


class Product(AdditiveComponent):
    """
    q(x) = ca(x) * cb(x)
    """
    def __init__(self, ca, cb):
        super().__init__()
        self.ca = ca
        self.cb = cb

    def __repr__(self):
        return f"Product({self.ca}|{self.cb})"

    def symbolic(self, x):
        return self.ca.symbolic(x) * self.cb.symbolic(x)

    def evaluate(self, xyz):
        return self.ca.evaluate(xyz) * self.cb.evaluate(xyz)

    def evaluate_derivative(self, xyz):
        return self.ca.evaluate(xyz) * self.cb.evaluate_derivative(xyz) + \
               self.ca.evaluate_derivative(xyz) * self.cb.evaluate(xyz)


class Power(AdditiveComponent):
    """
    q(x) = x ** power
    """
    def __init__(self, power):
        super().__init__()
        self.power = power

    def __repr__(self):
        return f"Power({self.power:.2f})"

    def symbolic(self, x):
        return sp.Piecewise(
            (0, x < -1.5), (x ** self.power, x < 1.5), (0, True)
        )

    def evaluate(self, xyz):
        if self.power == 0:
            return torch.ones_like(xyz)
        return xyz ** self.power

    def evaluate_derivative(self, xyz):
        if self.power == 0:
            return torch.zeros_like(xyz)
        elif self.power == 1:
            return torch.ones_like(xyz)
        return self.power * (xyz ** (self.power - 1))


class Bezier(AdditiveComponent):
    """
    q(x) = bezier(x)
    """
    def __repr__(self):
        return "Bezier()"

    def symbolic(self, x):
        return sp.Piecewise(
            (0, x < -1.5), ((x + 1.5) ** 2, x < -0.5), (-2 * x ** 2 + 1.5, x < 0.5),
            ((x - 1.5) ** 2, x < 1.5), (0, True)
        )

    def evaluate(self, xyz):
        b1 = (xyz + 1.5) ** 2
        b2 = -2 * (xyz ** 2) + 1.5
        b3 = (xyz - 1.5) ** 2
        m1 = (xyz >= -1.5) & (xyz < -0.5)
        m2 = (xyz >= -0.5) & (xyz < 0.5)
        m3 = (xyz >= 0.5) & (xyz < 1.5)
        return m1 * b1 + m2 * b2 + m3 * b3

    def evaluate_derivative(self, xyz):
        b1 = 2 * xyz + 3
        b2 = -4 * xyz
        b3 = 2 * xyz - 3
        m1 = (xyz >= -1.5) & (xyz < -0.5)
        m2 = (xyz >= -0.5) & (xyz < 0.5)
        m3 = (xyz >= 0.5) & (xyz < 1.5)
        return m1 * b1 + m2 * b2 + m3 * b3


class Sinusoid(AdditiveComponent):
    """
    q(x) = sin(beta * x + gamma)
    """
    def __init__(self, beta, gamma):
        super().__init__()
        self.beta = beta
        self.gamma = gamma

    def __repr__(self):
        return f"Sin({self.beta:.2f}-{self.gamma:.2f})"

    def symbolic(self, x):
        return sp.Piecewise(
            (0, x < -1.5), (sp.sin(self.beta * x + self.gamma), x < 1.5), (0, True)
        )

    def evaluate(self, xyz):
        return torch.sin(self.beta * xyz + self.gamma)

    def evaluate_derivative(self, xyz):
        return self.beta * torch.cos(self.beta * xyz + self.gamma)


class SinPeaks(Sinusoid):
    def __init__(self, n_peaks):
        super().__init__(math.pi * n_peaks / 3, -0.5 * math.pi * n_peaks + math.pi)
        self.n_peaks = n_peaks

    def __repr__(self):
        return f"SinPeaks({self.n_peaks})"
