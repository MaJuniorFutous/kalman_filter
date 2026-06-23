from typing import Any

import numpy as np
from numpy._typing._array_like import NDArray 

def np_arr(x: list) -> NDArray[Any]: return np.array(x, dtype=float)

def np_ident(x: int) -> NDArray[Any]: return np.identity(x, dtype=float)

def np_zeros(x: tuple) -> NDArray[Any]: return np.zeros(x, dtype=float)