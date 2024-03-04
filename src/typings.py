import picos
import numpy as np
import sympy as sym

Unitary = np.ndarray
Unitaries = list[Unitary]
Circuit = list[Unitary]
Barrier = sym.Poly
BarrierCertificate = list[tuple[Unitary, Barrier]]
Chunk = tuple[np.ndarray,int,int]
LamPoly = sym.Poly
LamVector = list[LamPoly]
LamList = list[LamVector]
Idx = int
SemiAlgebraic = list[sym.Poly]
SemiAlgebraicDict = dict[str, SemiAlgebraic]
Timings = dict[str, float]