from numpy.typing import NDArray
from typing import Any, Union, Optional
import datetime

import numpy as np, pandas as pd

from .utils import np_arr, np_ident, np_zeros


class KalmanFilter:

    STATIC_MATRICES = ['H', 'B', 'R', 'q', 'A']
    RUNTIME_MATRICES = ['x', 'P', 'K']
    MODEL_PARAMS = ['n_state_var', 'n_measurement_inputs']

    def __init__(
            self,
            n_state_var: int,
            n_measurement_inputs: int,
            A: np.array = None,
            H: np.array = None,
            B: np.array = None,
            R: np.array = None,
            q: np.array = None,
            save_file: str = 'kfilter_save.npz'):

        self.n_state_var = n_state_var
        self.n_measurement_inputs = n_measurement_inputs

        self.x = np_zeros((n_state_var, 1))
        self.P = np_ident(n_state_var)  # PCM
        self.H = H if H is not None else np_zeros((n_measurement_inputs, n_state_var))  # observation matrix/measurement function
        self.A = A if A is not None else np_ident(n_state_var)  # state transition matrix
        self.B = B  # control matrix
        self.q = q if q is not None else np_ident(n_state_var)  # process noise covariance
        self.R = R if R is not None else np_ident(n_measurement_inputs)  # measurement/observation noise covariance (PCM)
        self._I = np_ident(n_state_var)
        self.K = np_zeros((n_state_var, n_measurement_inputs)) # Kalman gain

        self.save_file = save_file

        assert self.H.shape == (n_measurement_inputs, n_state_var), "H shape does not match n_measurement_inputs and n_state_var"

        #* A matrix needs to be a 2x2 if state is 2x1, 3x3 if state is 3x1, etc
    
    def _predict(
            self, 
            A: np.array = None, 
            B: np.array = None,
            q: np.array = None,  #eg. nutrition facts label error
            u = None):
        if B is None:
            B = self.B
        if A is None:
            A = self.A
        if q is None:
            q = self.q
            
        # state prediction
        if B is not None and (u is not None and u != 0):
            self.x = A@self.x + B@u
        else: self.x = A@self.x

        #* Main equation X_k = X_k-1 + V_k-1*deltaT
        # Process covariance prediction
        self.P = A@self.P@A.T + q  # q should match PCM and A

        #TODO: Hold on to prior PCM (PCM_k-1) and state prior to update

    def _update(self, z: Union[np.ndarray, float], R = None):
        if R is None: R = self.R
        elif np.isscalar(R):
            R = np_ident(self.n_measurement_inputs) * R

        z = np.atleast_2d(z)
        if z.shape[1] == self.n_measurement_inputs: z = z.T

        if z.shape != (self.n_measurement_inputs, 1):
            raise ValueError(
                f"Cant reshape z correctly, need: {(self.n_measurement_inputs, 1)}"
            )

        # Kalman Gain
        # equation (deprecated, we dont actually need to calculate the inverse)
        # S = H@predicted_pcm@H.T + R # inovation covariance
        # K = predicted_pcm@H.T@np.linalg.inv(S)

        '''
        Note: np.linalg.solve(A, B) only solves equation in the form: AX = B
        '''
        S = self.H@self.P@self.H.T + R #inovation covariance
        self.K = np.linalg.solve(S.T, (self.H@self.P.T)).T

        self.x = self.x + self.K@(z - self.H@self.x)

        # P = (self._I - self.K@self.H)@self.P  # faster (asymetrical)
        self.P = (self._I - self.K@self.H)@self.P@(self._I - self.K@self.H).T + self.K@R@self.K.T  # Joseph form (symetrical) 
    
    @staticmethod
    def _to_numpy(x, transpose: bool = False):
        if x is None: return None

        if isinstance(x, (pd.Series, pd.DataFrame)):
            assert not x.empty, "Pandas object cannot be empty"
            return x.to_numpy().T if transpose else x.to_numpy()

        assert isinstance(x, np.ndarray), (
            f"Expected None, numpy.ndarray, pandas.Series, or pandas.DataFrame, got {type(x)}"
        )
        assert x.size != 0, "NumPy array cannot be empty"
        return x.T if transpose else x
    
    def _construct_state(self, z, H: Optional[np.array] = None):
        if H is None: H = self.H
        return np.linalg.pinv(np.asarray(H, dtype=float))@np.asarray(z, dtype=float).reshape(-1, 1)
    
    def _bootstrap_filter(
            self, 
            data: np.array, 
            init_n: int,
            negate_cv: bool = False, 
            unobserved_variance: float = 0.0, 
            initialize_pcm: bool = False,
            transpose: bool = False):
        assert init_n > 0, "Must pass larger than 0 integer for 'init_n'"
        if initialize_pcm:
            if init_n <= 1:
                print("Can't initialize PCM. init_n needs to be at least 2")
            else:
                temp_pcm = np.atleast_2d(np.cov(m=data[:init_n], rowvar=False if transpose else True, dtype=float))
                observed_idx = np.where(np.any(self.H != 0, axis=0))[0]

                assert temp_pcm.shape == (len(observed_idx), len(observed_idx)), \
                    "Temp PCM shape mismatch with number of observation variables."

                self.P = np.full((self.n_state_var, self.n_state_var), unobserved_variance, dtype=float)
                
                #! Deprecated np_zeros for np.full now that we have unobserved_variance param
                # self.P = np_zeros((self.n_state_var, self.n_state_var))

                self.P[np.ix_(observed_idx, observed_idx)] = temp_pcm
                if negate_cv: self.P = np.diag(np.diag(self.P))

        self.x = self._construct_state(z=data[init_n - 1])
        return data[init_n:]

    def forward(
            self, 
            data: Optional[Union[np.ndarray, pd.DataFrame, pd.Series]], 
            R: Optional[Union[np.ndarray, pd.DataFrame, pd.Series]] = None,
            q: Optional[Union[np.ndarray, pd.DataFrame, pd.Series]] = None,
            A: Optional[Union[np.ndarray, pd.DataFrame, pd.Series]] = None,
            B: Optional[Union[np.ndarray, pd.DataFrame, pd.Series]] = None,
            u: Optional[Union[np.ndarray, pd.DataFrame, pd.Series]] = None,
            return_history: Optional[bool] = False,
            batch_init_n: Optional[int] = None,
            batch_init_negate_cv: bool = False,
            initialize_pcm: bool = False,
            unobserved_variance: Optional[float] = None,
            bootstrap_transpose: bool = True):
        if not isinstance(data, (np.ndarray, pd.DataFrame, pd.Series)):
            raise TypeError("data must be a numpy array or pandas DataFrame")
        '''Sort ascending by date first, custom_deltaT is the index of the custom deltaT column'''
        if batch_init_negate_cv and batch_init_n is None:
            raise ValueError("batch_init must be set if batch_init_negate_cv is provided")
        # if ignore_1st and (batch_init_negate_cv or batch_init_negate_cv):
        #     print("Prioritizing batch initialization over initial state setting using 1st row.")
        
        data, R, q, A, B, u = self._to_numpy(data), self._to_numpy(R), self._to_numpy(q), self._to_numpy(A), self._to_numpy(B), self._to_numpy(u)
        if batch_init_n:
            data = self._bootstrap_filter(
                data=data, 
                init_n=batch_init_n, 
                negate_cv=batch_init_negate_cv,
                unobserved_variance=unobserved_variance,
                initialize_pcm=initialize_pcm,
                transpose=bootstrap_transpose)
        records = [
            {
                'data': data[i],
                'R': None if R is None else R[i],
                'q': None if q is None else q[i],
                'A': None if A is None else A[i],
                'B': None if B is None else B[i],
                'u': None if u is None else u[i]
            } for i in range(len(data))
        ]

        if return_history: estimations = []
        for record in records:
            self._predict(A=record['A'], q=record['q'], B=record['B'], u=record['u'])
            self._update(z=record['data'], R=record['R'])
            if return_history:
                estimations.append(tuple(i for i in self.x.T[0]))
        
        if not return_history: return [tuple(i for i in self.x.T[0])]
        else: return estimations

    def save_state(
            self, 
            file: Optional[str] = None, 
            matrices: Optional[Union[list, str]] = None,
            mdata: dict = None):
        if matrices is None: matrices = self.STATIC_MATRICES
        if file is None: file = self.save_file
        
        if isinstance(matrices, str): matrices = list(matrices)
        data = {i: getattr(self, i) for i in matrices + self.RUNTIME_MATRICES + self.MODEL_PARAMS if hasattr(self, i)}
        if mdata:
            np.savez(file,
                    **data,
                    metadata=mdata,
                    allow_pickle=True)
        else:
            np.savez(file,
                    **data,
                    allow_pickle=True)

    @classmethod
    def extract_checkpoint(cls, file) -> Any: 
        return np.load(file, allow_pickle=True)

    @classmethod
    def from_file(cls, file):
        data = cls.extract_checkpoint(file)
        return cls(
            n_state_var=data['n_state_var'],
            n_measurement_inputs=data['n_measurement_inputs'],
            A=data.get('A'),
            H=data.get('H'),
            B=data.get('B'),
            R=data.get('R'),
            q=data.get('q')
        )