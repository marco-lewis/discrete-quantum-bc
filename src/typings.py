import numpy as np
import sympy as sym

type Unitary = np.ndarray
type Unitaries = list[Unitary]
type Circuit = list[Unitary]
type Barrier = sym.Poly
type BarrierCertificate = list[tuple[Unitary, Barrier]]
type Chunk = tuple[np.ndarray,int,int]
type LamPoly = sym.Poly
type LamVector = list[LamPoly]
type LamList = list[LamVector]
type Idx = int
type SemiAlgebraic = list[sym.Poly]
type SemiAlgebraicDict = dict[str, SemiAlgebraic]
type Timings = dict[str, float]