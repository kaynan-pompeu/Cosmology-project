import abc
from typing import Any, Dict, Type
import numpy as np

# =============================================================================
# Base abstract class for potentials
# =============================================================================

class ScalarPotential(abc.ABC):
    """
    An abstract base class that defines the interface for all potential models.

    Every potential implemented MUST inherit from this class and implement 
    its abstract methods.
    """
    def __init__(self, name: str):
        self.name = name
        # Each potential will store its unique parameters in this dictionary.
        self.params: Dict[str, Any] = {}

    @abc.abstractmethod
    def value(self, phi: float) -> float:
        """
        Calculates the value of the potential, V(phi), at a given phi.
        This method MUST be implemented by any child class.
        """
        pass

    @abc.abstractmethod
    def derivative(self, phi: float) -> float:
        """
        Calculates the derivative of the potential, dV/dphi, at a given phi.
        This method MUST be implemented by any child class.
        """
        pass

    def __repr__(self) -> str:
        """Provides a clean string representation of the potential object."""
        return f"{self.name} Potential with params: {self.params}"

# =============================================================================
# 2. Specific Potential Implementations
# =============================================================================

class ConstantPotential(ScalarPotential):
    """
    Constant potential, V(phi) = V0.
    """
    def __init__(self, V0: float):
        super().__init__("Constant")
        self.params['V0'] = V0

    def value(self, phi: float) -> float:
        """Returns the constant value V0, regardless of phi."""
        return self.params['V0']

    def derivative(self, phi: float) -> float:
        """The derivative of a constant is always zero."""
        return 0.0

class ExponentialPotential(ScalarPotential):
    """
    An exponential potential of the form V(phi) = A * exp(B * phi).
    """
    def __init__(self, A: float, B: float):
        super().__init__("Exponential")
        self.params['A'] = A
        self.params['B'] = B

    def value(self, phi: float) -> float:
        """Calculates V(phi) = A * exp(B * phi)."""
        A = self.params['A']
        B = self.params['B']
        return A * np.exp(B * phi)

    def derivative(self, phi: float) -> float:
        """Calculates dV/dphi = A * B * exp(B * phi)."""
        A = self.params['A']
        B = self.params['B']
        return A * B * np.exp(B * phi)
    
class PowerLawPotential(ScalarPotential):
    """
    A power law potential of the form V(phi) = A + B*\phi + C*\phi**2.
    """
    def __init__(self, A: float, B: float, C: float):
        super().__init__("Power Law")
        self.params['A'] = A
        self.params['B'] = B
        self.params['C'] = C

    def value(self, phi: float) -> float:
        """Calculates V(phi) = A + B*phi + C*phi**2."""
        A = self.params['A']
        B = self.params['B']
        C = self.params['C']
        return A + B*phi + C*phi**2

    def derivative(self, phi: float) -> float:
        """Calculates dV/dphi = B + 2 * C * phi."""
        B = self.params['B']
        C = self.params['C']
        return B + 2 * C * phi
    
class GaussianPotential(ScalarPotential):
    """
    A Gaussian potential of the form V(phi) = A*exp{-B*(x-C)**2}.
    """
    def __init__(self, A: float, B: float, C: float):
        super().__init__("Gaussian")
        self.params['A'] = A
        self.params['B'] = B
        self.params['C'] = C

    def value(self, phi: float) -> float:
        """Calculates V(phi) = V(phi) = A*exp{-B*(x-C)**2}."""
        A = self.params['A']
        B = self.params['B']
        C = self.params['C']
        return A*(np.exp(-B*(phi-C)**2))

    def derivative(self, phi: float) -> float:
        """Calculates dV/dphi = -2*B*(phi-C)*V(phi)."""
        A = self.params['A']
        B = self.params['B']
        C = self.params['C']
        return -2*B*(phi-C)*self.value(phi)

class InversePowerLawPotential(ScalarPotential):
    """
    An inverse power law potential of the form V(phi) = A * phi^(-B).
    """
    def __init__(self, A: float, B: float):
        super().__init__("Inverse Power Law")
        self.params['A'] = A
        self.params['B'] = B

    def value(self, phi: float) -> float:
        """Calculates V(phi) = A * phi^(-B)."""
        A = self.params['A']
        B = self.params['B']
        return np.where(phi != 0, A * phi**(-B), -np.inf)
    
    def derivative(self, phi: float) -> float:
        """Calculates dV/dphi = -A * B * phi^(-B-1)."""
        A = self.params['A']
        B = self.params['B']
        return np.where(phi != 0, -A * B * phi**(-B - 1), -np.inf)

class SquareWellPotential(ScalarPotential):
    """
    A square well potential.
    V(phi) = -V0 if A <= phi <= B, and V0 otherwise.
    """
    def __init__(self, V0: float, A: float, B: float):
        super().__init__("Square Well")
        self.params['V0'] = V0
        self.params['A'] = A
        self.params['B'] = B

    def value(self, phi: np.ndarray) -> np.ndarray:
        """Calculates the potential value in a vectorized way."""
        V0 = self.params['V0']
        A = self.params['A']
        B = self.params['B']
        
        # Condition: True where phi is inside the well [a, b]
        condition = (phi >= A) & (phi <= B)
        return np.where(condition, -1*V0, V0)
        
    def derivative(self, phi: float) -> float:
        """
        The derivative is zero everywhere except at the discontinuities,
        which we handle by returning zero for numerical stability.
        """
        return 0.0

class HyperTanPotential(ScalarPotential):
    """
    A hyoerbolic tangent potential that sort of imitates a square-well like:
    V_{0}\left(1-\frac{1}{\tanh\left(k\left(b-a\right)\right)}\left(\tanh\left(2k\left(x\ -a\right)\right)-\tanh\left(2k\left(x-b\right)\right)\right)\right)
    """
    def __init__(self, V0: float, A: float, B: float, C: float):
        super().__init__("Hyperbolic Tangent")
        self.params['V0'] = V0
        self.params['A'] = A
        self.params['B'] = B
        self.params['C'] = C

    def value(self, phi: np.ndarray) -> np.ndarray:
        V0 = self.params['V0']
        A = self.params['A']
        B = self.params['B']
        C = self.params['C']
        
        k = C/(B - A)
        return V0*(1 - (1/np.tanh(k*(B - A)))*(np.tanh(2*k*(phi - A)) - np.tanh(2*k*(phi - B))))
        
    def derivative(self, phi: float) -> float:
        V0 = self.params['V0']
        A = self.params['A']
        B = self.params['B']
        C = self.params['C']
        
        k = C/(B - A)
        return 2 * k * V0 * (1 / np.tanh(k * (B - A))) * (-(1 / np.cosh(2 * k * (phi - A))**2) + (1 / np.cosh(2 * k * (phi - B))**2))

# This dictionary maps an integer flag to the corresponding potential class.
# To add a new potential, define its class and then add an
# entry to this dictionary.
potential_factory: Dict[int, Type[ScalarPotential]] = {
        1: ConstantPotential,
        2: ExponentialPotential,
        3: PowerLawPotential,
        4: GaussianPotential,
        5: InversePowerLawPotential,
        6: SquareWellPotential,
        7: HyperTanPotential
    }
