from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
import scipy as sp

class AbstractCriteria(ABC):

    @abstractmethod
    def __init__(self, P: np.ndarray, R: np.ndarray, lambdas: np.ndarray) -> None:
        raise NotImplementedError
        
    @abstractmethod
    def get_optimum(self) -> int:
        raise NotImplementedError
    
    @abstractmethod
    def compute_curve(self) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

class CRESO(AbstractCriteria):

    def __init__(self, P: np.ndarray, R: np.ndarray, lambdas: np.ndarray) -> None:
        self.P = P
        self.R = R
        self.lambdas = lambdas
        self.x, self.y = self.compute_curve()

    def compute_curve(self) -> np.ndarray:
        l2, r2, p2 = self.lambdas**2, self.R**2, self.P**2
        return np.log10(l2)/2., np.gradient(l2 * p2 - r2, l2, edge_order=2)

    def get_optimum(self):
        return CRESO.first_local_max(self.y)
    
    @staticmethod
    def first_local_max(x: np.ndarray, **kwargs) -> int:
        peaks, _ = sp.signal.find_peaks(x, **kwargs)
        return x.argmax() if len(peaks) == 0 else peaks[0]
    
class UCurve(AbstractCriteria):

    def __init__(self, P: np.ndarray, R: np.ndarray, lambdas: np.ndarray) -> None:
        self.P = P
        self.R = R
        self.lambdas = lambdas
        self.x, self.y = self.compute_curve()

    def compute_curve(self):
        return np.log10(self.lambdas), np.log10(1./self.R**2 + 1./self.P**2)

    def get_optimum(self):
        return self.y.argmin()
    
class LCurve(AbstractCriteria):

    def __init__(self, P: np.ndarray, R: np.ndarray, lambdas: np.ndarray) -> None:

        self.P = P
        self.R = R
        self.lambdas = lambdas
        self.x, self.y = self.compute_curve()

    def compute_curve(self):
        return np.log10(self.R), np.log10(self.P)
    
    def get_optimum(self):
        _, r = LCurve.osculating_circle(self.x, self.y, self.lambdas)
        return r.argmin()
    
    @staticmethod
    def osculating_circle(x: np.ndarray, y:np.ndarray, t:np.ndarray) -> np.ndarray:
        dx, dy = np.gradient(x, t), np.gradient(y, t)
        d2x, d2y = np.gradient(dx, t), np.gradient(dy, t)
        dn = np.sqrt(dx**2 + dy**2)
        r = np.abs( dn**3 / (dx*d2y - dy*d2x) )
        c = (x - dy*r/dn, y + dx*r/dn )
        return c, r