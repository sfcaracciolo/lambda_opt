from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
import scipy as sp

class AbstractCriteria(ABC):

    def __init__(self, P: np.ndarray, R: np.ndarray, lambdas: np.ndarray) -> None:          
        self.P = P
        self.P[P==0.] = np.nan

        self.R = R
        self.R[P==0.] = np.nan

        self.lambdas = lambdas
        self.lambdas[P==0.] = np.nan

        self.x, self.y = self.compute_curve()
        
    @abstractmethod
    def get_optimum(self) -> int:
        raise NotImplementedError
    
    @abstractmethod
    def compute_curve(self) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

class CRESO(AbstractCriteria):

    def __init__(self, P: np.ndarray, R: np.ndarray, lambdas: np.ndarray) -> None:
        super().__init__(P, R, lambdas)

    def compute_curve(self) -> np.ndarray:
        l, r2, p2 = self.lambdas, self.R**2, self.P**2
        c = np.gradient(l * p2 - r2, l, edge_order=1) # argument sholud be always ascending
        return np.log10(l), np.log10(c)

    def get_optimum(self):
        return CRESO.first_local_max(self.y)
    
    @staticmethod
    def first_local_max(x: np.ndarray, **kwargs) -> int:
        peaks, _ = sp.signal.find_peaks(x, **kwargs)
        return np.nanargmax(x) if len(peaks) == 0 else peaks[0]
    
class UCurve(AbstractCriteria):

    def __init__(self, P: np.ndarray, R: np.ndarray, lambdas: np.ndarray) -> None:
        super().__init__(P, R, lambdas)

    def compute_curve(self):
        return np.log10(self.lambdas), np.log10(1./self.R**2 + 1./self.P**2)

    def get_optimum(self):
        return np.nanargmin(self.y)
    
class LCurve(AbstractCriteria):

    def __init__(self, P: np.ndarray, R: np.ndarray, lambdas: np.ndarray) -> None:
        super().__init__(P, R, lambdas)

    def compute_curve(self):
        return np.log10(self.R), np.log10(self.P)
    
    def get_optimum(self):
        _, r = LCurve.osculating_circle(self.x, self.y, self.lambdas)
        return np.nanargmin(r)
    
    @staticmethod
    def osculating_circle(x: np.ndarray, y:np.ndarray, t:np.ndarray) -> np.ndarray:
        dx, dy = np.gradient(x, t), np.gradient(y, t)
        d2x, d2y = np.gradient(dx, t), np.gradient(dy, t)
        dn = np.sqrt(dx**2 + dy**2)
        num, den = dn**3, dx*d2y - dy*d2x
        r = np.full_like(x, np.nan)
        np.divide(num, den, where=den!=0, out=r)
        np.abs(r, out=r)
        c = (x - dy*r/dn, y + dx*r/dn )
        return c, r