from typing import Any, Union, Literal
import datetime

import numpy as np, pandas as pd


def np_arr(x: list): return np.array(x, dtype=float)

class KalmanFilter:

    #TODO: Initialize these to zeros or identities based on shape/structure of X
    def __init__(
            self,
            n_state_var: int,
            n_measurement_inputs: int,
            A: np.array = None,
            H: np.array = None,
            B: np.array = None,
            R: np.array = None,
            q: np.array = None):

        self.n_state_var = n_state_var
        self.n_measurement_inputs = n_measurement_inputs

        self.x = np.zeros((n_state_var, 1))
        self.P = np.identity(n_state_var)  # PCM
        self.H = H if H.size!=0 else np.zeros((n_measurement_inputs, n_state_var))  # observation matrix/measurement function
        self.A = A if A else np.identity(n_state_var)  # state transition matrix
        self.B = B  # control matrix
        self.q = q if q else np.identity(n_state_var)  # process noise covariance
        self.R = R if R else np.identity(n_measurement_inputs)  # measurement/observation noise covariance (PCM)
        self._I = np.identity(n_state_var)
        self.K = np.zeros((n_state_var, n_measurement_inputs)) # Kalman gain

        assert self.H.shape == (n_measurement_inputs, n_state_var), "H shape does not match n_measurement_inputs and n_state_var"

        #* A matrix needs to be a 2x2 if state is 2x1, 3x3 if state is 3x1, etc
    
    #TODO: Determine correct typing enforcement
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
        else:
            self.x = A@self.x

        #* Main equation X_k = X_k-1 + V_k-1*deltaT
        # Process covariance prediction
        self.P = A@self.P@A.T + q  # q should match PCM and A

        #TODO: Hold on to prior PCM (PCM_k-1) and state prior to update

    def _update(self, z: Union[np.ndarray, float], R = None):
        if R is None:
            R = self.R
        elif np.isscalar(R):
            R = np.identity(self.n_measurement_inputs) * R

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
    def _to_numpy(x):
        if x is None: return None

        if isinstance(x, (pd.Series, pd.DataFrame)):
            assert not x.empty, "Pandas object cannot be empty"
            return x.to_numpy()

        assert isinstance(x, np.ndarray), (
            f"Expected None, numpy.ndarray, pandas.Series, or pandas.DataFrame, got {type(x)}"
        )
        assert x.size != 0, "NumPy array cannot be empty"
        return x
    
    @staticmethod
    def _construct_state(H, z):
        H = np.asarray(H, dtype=float)
        z = np.asarray(z, dtype=float).reshape(-1, 1)
        return np.linalg.pinv(H)@z
    
    def _initialize_pcm(self, data: np.array, init_n: int, negate_cv: bool = False, transpose: bool = False):
        subset = data[:init_n]

        if transpose: data_n_vars, rowvar = data.shape[1], False
        else: data_n_vars, rowvar = data.shape[0], True

        assert data_n_vars == self.P.shape[1], f"Incorrect number of data vars {data_n_vars}, need to match {self.P.shape[1]}"
        self.P = np.cov(m=subset, rowvar=rowvar, dtype=float)
        if negate_cv: self.P = np.diag(np.diag(self.P))
        return data[init_n:]

    def forward(
            self, 
            data: Union[np.ndarray, pd.DataFrame, pd.Series], 
            R: Union[np.ndarray, pd.DataFrame, pd.Series] | None = None,
            q: Union[np.ndarray, pd.DataFrame, pd.Series] | None = None,
            A: Union[np.ndarray, pd.DataFrame, pd.Series] | None = None,
            B: Union[np.ndarray, pd.DataFrame, pd.Series] | None = None,
            u: Union[np.ndarray, pd.DataFrame, pd.Series] | None = None,
            return_history: bool = False,
            batch_init_n: int | None = None,
            batch_init_negate_cv: bool | None = None):
        if not isinstance(data, (np.ndarray, pd.DataFrame, pd.Series)):
            raise TypeError("data must be a numpy array or pandas DataFrame")
        '''Sort ascending by date first, custom_deltaT is the index of the custom deltaT column'''
        if batch_init_negate_cv is not None and batch_init_n is None:
            raise ValueError("batch_init must be set if batch_init_negate_cv is provided")
        # if ignore_1st and (batch_init_negate_cv or batch_init_negate_cv):
        #     print("Prioritizing batch initialization over initial state setting using 1st row.")
        
        data, R, q, A, B, u = self._to_numpy(data), self._to_numpy(R), self._to_numpy(q), self._to_numpy(A), self._to_numpy(B), self._to_numpy(u)
        if batch_init_n:
            data = self._initialize_pcm(
                data=data, 
                init_n=batch_init_n, 
                negate_cv=batch_init_negate_cv,
                transpose=True)
        records = [
            {
                'record': i,
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
            #! deprecated ignore_first
            # if ignore_1st and record['record'] == 0:
            #     self.x = self._construct_state(H=self.H, z=record['data'])
            #     continue

            self._predict(A=record['A'], q=record['q'], B=record['B'], u=record['u'])
            self._update(z=record['data'], R=record['R'])
            # if return_history:
            #     estimations.append()

        

if __name__ == '__main__':
    # For R matrix
    err_obs_pos = 0.0025 # default, standard bathroom scale error += 1% or 2% of current body weight
    default_t, n_state_var, n_measurement_var = 1, 2, 1
    # For Q matrix percentages
    bw_perc, vel_perc = 0.005, 0.0005
    
    data = {
        "body_weight": [
            184.5, 183.5, 184.5, 184.5, 186.5,
            184.7, 184.7, 185.2, 186.6, 188.7,
            186.6, 187.0, 185.3, 186.5
        ],
        "delete": [
            185.1, 182.8, 185.0, 183.9, 187.2,
            183.8, 185.4, 184.6, 187.3, 187.9,
            185.7, 188.1, 184.5, 187.0
        ],
        "datetime": pd.to_datetime([
            "2026-05-13",
            "2026-05-10",
            "2026-05-08",
            "2026-05-07",
            "2026-05-06",
            "2026-05-03",
            "2026-05-02",
            "2026-05-01",
            "2026-04-30",
            "2026-04-29",
            "2026-04-28",
            "2026-04-27",
            "2026-04-26",
            "2026-04-25"
        ])
    }

    df = pd.DataFrame(data)
    df.sort_values('datetime', ascending=True, inplace=True)
    batch_init_n = int((df['datetime'].dt.date > (pd.Timestamp.now().date() - pd.Timedelta(days=30))).sum())
    # get dynamic delta T
    df["delta_v"] = (
        (df["datetime"] - df["datetime"].shift(1)).dt.days
    )
    df['A'] = [np_arr([[1, dt if not np.isnan(dt) else 0],[0, 1]]) for dt in df['delta_v']]

    # create Q and R matrices
    df['R'] = [np.identity(n_measurement_var, dtype=float) * ((bw*err_obs_pos)**2) for bw in df['body_weight']]
    df['q'] = [np_arr([[(bw*bw_perc)**2, 0],[0, (bw*vel_perc)**2]]) for bw in df['body_weight']]

    #TODO: create dynamic R based on scale error as % bodyweight
    filter = KalmanFilter(
        n_state_var=n_state_var,
        n_measurement_inputs=n_measurement_var,
        H=np_arr([[1, 0]])
    )
    #* pass dynamic Q (how “non-constant” your weight trend is) and R based on scale error as % bodyweight
    filter.forward(
        data=df[['body_weight', 'delete']],
        R=df['R'],
        q=df['q'],
        A=df['A'],
        # B=df['B'],
        # u=df['u'],
        return_history=True,
        batch_init_n=batch_init_n,
        batch_init_negate_cv=True
    )