import abc
from typing import Any, Dict, Type
import numpy as np

# =============================================================================
# Base abstract class for potentials
# =============================================================================

class ScalarPotential(abc.ABC):
    def __init__(self, name: str):
        self.name = name
        self.params: Dict[str, Any] = {}

    @abc.abstractmethod
    def value(self, phi: float) -> float:
        pass

    @abc.abstractmethod
    def derivative(self, phi: float) -> float:
        pass

    @abc.abstractmethod
    def to_latex(self) -> str:
        """Returns a LaTeX string representation for plotting legends."""
        pass

    def __repr__(self) -> str:
        return f"{self.name} Potential with params: {self.params}"

# =============================================================================
# 2. Specific Potential Implementations
# =============================================================================

class ConstantPotential(ScalarPotential):
    def __init__(self, V0: float):
        super().__init__("Constant")
        self.params['V0'] = V0

    def value(self, phi: float) -> float:
        return self.params['V0']

    def derivative(self, phi: float) -> float:
        return 0.0

    def to_latex(self) -> str:
        v0 = self.params['V0']
        return fr"$V(\phi) = V_0$" + "\n" + fr"($V_0 \approx {v0:.3e}$)"

class ExponentialPotential(ScalarPotential):
    def __init__(self, A: float, B: float):
        super().__init__("Exponential")
        self.params['A'] = A
        self.params['B'] = B

    def value(self, phi: float) -> float:
        return self.params['A'] * np.exp(self.params['B'] * phi)

    def derivative(self, phi: float) -> float:
        return self.params['A'] * self.params['B'] * np.exp(self.params['B'] * phi)

    def to_latex(self) -> str:
        a, b = self.params['A'], self.params['B']
        return fr"$V(\phi) = A e^{{B\phi}}$" + "\n" + fr"($A \approx {a:.2e}, B \approx {b:.2e}$)"
    
class PowerLawPotential(ScalarPotential):
    def __init__(self, A: float, B: float):
        super().__init__("Power Law")
        self.params['A'] = A
        self.params['B'] = B

    def value(self, phi: float) -> float:
        return self.params['A'] * (phi**self.params['B'])

    def derivative(self, phi: float) -> float:
        return self.params['A'] * self.params['B'] * (phi**(self.params['B'] - 1))

    def to_latex(self) -> str:
        a, b = self.params['A'], self.params['B']
        return fr"$V(\phi) = A \phi^{{B}}$" + "\n" + fr"($A \approx {a:.2e}, B \approx {b:.2e}$)"
    
class GaussianPotential(ScalarPotential):
    def __init__(self, A: float, B: float, C: float):
        super().__init__("Gaussian")
        self.params['A'] = A
        self.params['B'] = B
        self.params['C'] = C

    def value(self, phi: float) -> float:
        A, B, C = self.params['A'], self.params['B'], self.params['C']
        return A * (np.exp(-(B**2) * (phi - C)**2))

    def derivative(self, phi: float) -> float:
        A, B, C = self.params['A'], self.params['B'], self.params['C']
        return -2 * (B**2) * (phi - C) * A * np.exp(-(B**2) * (phi - C)**2)

    def to_latex(self) -> str:
        a, b, c = self.params['A'], self.params['B'], self.params['C']
        return fr"$V(\phi) = A e^{{-B^2(\phi-C)^2}}$" + "\n" + fr"($A \approx {a:.2e}, B \approx {b:.2e}, C \approx {c:.2e}$)"

class InversePowerLawPotential(ScalarPotential):
    def __init__(self, A: float, B: float):
        super().__init__("Inverse Power Law")
        self.params['A'] = A
        self.params['B'] = B

    def value(self, phi: float) -> float:
        return np.where(phi != 0, self.params['A'] * phi**(-self.params['B']), -np.inf)
    
    def derivative(self, phi: float) -> float:
        A, B = self.params['A'], self.params['B']
        return np.where(phi != 0, -A * B * phi**(-B - 1), -np.inf)

    def to_latex(self) -> str:
        a, b = self.params['A'], self.params['B']
        return fr"$V(\phi) = A \phi^{{-B}}$" + "\n" + fr"($A \approx {a:.2e}, B \approx {b:.2e}$)"

class SquareWellPotential(ScalarPotential):
    def __init__(self, V0: float, A: float, B: float):
        super().__init__("Square Well")
        self.params['V0'] = V0
        self.params['A'] = A
        self.params['B'] = B

    def value(self, phi: np.ndarray) -> np.ndarray:
        V0, A, B = self.params['V0'], self.params['A'], self.params['B']
        condition = (phi >= A) & (phi <= B)
        return np.where(condition, -0.0001 * V0, V0)
        
    def derivative(self, phi: float) -> float:
        return 0.0

    def to_latex(self) -> str:
        return r"$V = V_{well}$ (Square Well)"

class HyperTanPotential(ScalarPotential):
    def __init__(self, V0: float, A: float, B: float, C: float):
        super().__init__("Hyperbolic Tangent")
        self.params['V0'] = V0
        self.params['A'] = A
        self.params['B'] = B
        self.params['C'] = C

    def value(self, phi: np.ndarray) -> np.ndarray:
        V0, A, B, C = self.params['V0'], self.params['A'], self.params['B'], self.params['C']
        k = C / (B - A)
        return V0 * (1 - (1 / np.tanh(k * (B - A))) * (np.tanh(2 * k * (phi - A)) - np.tanh(2 * k * (phi - B))))
        
    def derivative(self, phi: float) -> float:
        V0, A, B, C = self.params['V0'], self.params['A'], self.params['B'], self.params['C']
        k = C / (B - A)
        return 2 * k * V0 * (1 / np.tanh(k * (B - A))) * (-(1 / np.cosh(2 * k * (phi - A))**2) + (1 / np.cosh(2 * k * (phi - B))**2))

    def to_latex(self) -> str:
        v0, a, b, c = self.params['V0'], self.params['A'], self.params['B'], self.params['C']
        return fr"$V(\phi) = V_{{tanh}}(\phi)$" + "\n" + fr"($V_0 \approx {v0:.2e}, A \approx {a:.2f}, B \approx {b:.2f}, C \approx {c:.2f}$)"

class AxionPotential(ScalarPotential):
    def __init__(self, A: float, B: float):
        super().__init__("Axion")
        self.params['A'] = A
        self.params['B'] = B

    def value(self, phi: float) -> float:
        return self.params['A'] * (1 + np.cos(self.params['B'] * phi))

    def derivative(self, phi: float) -> float:
        return (-1) * self.params['A'] * self.params['B'] * np.sin(self.params['B'] * phi)

    def to_latex(self) -> str:
        a, b = self.params['A'], self.params['B']
        return fr"$V(\phi) = A(1 + \cos(B\phi))$" + "\n" + fr"($A \approx {a:.2e}, B \approx {b:.2e}$)"

potential_factory: Dict[int, Type[ScalarPotential]] = {
        1: ConstantPotential,
        2: ExponentialPotential,
        3: PowerLawPotential,
        4: GaussianPotential,
        5: InversePowerLawPotential,
        6: SquareWellPotential,
        7: HyperTanPotential,
        8: AxionPotential
    }
