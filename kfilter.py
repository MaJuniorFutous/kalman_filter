from typing import Any, Union

import numpy as np, pandas as pd


def np_arr(x: list): return np.array(x, dtype=float)

class KalmanFilter:

    #TODO: Initialize these to zeros or identoties based on shape/structure of X
    def __init__(
            self,
            n_state_var: int,
            n_measurement_inputs: int,
            B: np.array = None):

        self.n_state_var = n_state_var
        self.n_measurement_inputs = n_measurement_inputs

        self.x = np.zeros((n_state_var, 1))
        self.P = np.identity(n_state_var)  # PCM
        self.H = np.zeros((n_measurement_inputs, n_state_var))  # observation matrix/measurement function
        self.A = np.identity(n_state_var)  # state transition matrix
        self.B = B  # control matrix
        self.q = np.identity(n_state_var)  # process noise covariance
        self.R = np.identity(n_measurement_inputs)  # measurement/observation noise covariance (PCM)
        self._I = np.identity(n_state_var)
        self.K = np.zeros((n_state_var, n_measurement_inputs)) # Kalman gain

        #* A matrix needs to be a 2x2 if state is 2x1, 3x3 if state is 3x1, etc
    
    #TODO: Determine correct typing enforcement
    def _predict(
            self, 
            data: np.array, 
            A: np.array = None, 
            B: np.array = None,
            q: np.array = None,  #eg. calories
            u = None):
        if B is None:
            B = self.B
        if A is None:
            A = self.A
        if q is None:
            q = self.q

        #TODO: Check all values that can be scalars are and if so, set their shape
        # reshape array
        self.x = np.array([[i] for i in data])

        # state prediction
        if B is not None and u is not None:
            self.x = A@self.x + B@u
        else:
            self.x = A@self.x

        #* Main equation X_k = X_k-1 + V_k-1*deltaT
        # Process covariance prediction
        self.P = A@self.P@A.T + q  # q should match PCM and A

        #TODO: Hold on to prior PCM (PCM_k-1) and state prior to update

    def _update(self, z: np.array, R = None):
        if R is None:
            R = self.R
        elif np.isscalar(R):
            R = np.identity(self.n_measurement_inputs) * R

        # Kalman Gain
        # equation (deprecated, we dont actually need to calculate the inverse)
        # S = H@predicted_pcm@H.T + R # inovation covariance
        # K = predicted_pcm@H.T@np.linalg.inv(S)

        '''
        Note: np.linalg.solve(A, B) only solves equation in the form: AX = B
        '''
        S = self.H@self.P@self.H.T + R #inovation covariance
        self.K = np.linalg.solve(S.T, (self.H@self.P.T)).T

        #TODO: Check equation video on this phase!!!!
        self.x = self.x + self.K@(z - self.H@self.x)

        # equation
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

    def forward(
            self, 
            data: Union[np.ndarray, pd.DataFrame, pd.Series], 
            R: Union[np.ndarray, pd.DataFrame, pd.Series] = None,
            Q: Union[np.ndarray, pd.DataFrame, pd.Series] = None):
        if not isinstance(data, (np.ndarray, pd.DataFrame, pd.Series)):
            raise TypeError("data must be a numpy array or pandas DataFrame")
        '''Sort ascending by date first, custom_deltaT is the index of the custom deltaT column'''
        data, R, Q = self._to_numpy(data), self._to_numpy(R), self._to_numpy(Q)
        for obs, r, q in zip(data, R, Q):
            ...
        

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
    # get dynamic delta T
    df["delta_v"] = (
        (df["datetime"] - df["datetime"].shift(1)).dt.days
    )
    df['A'] = [np_arr([[1, dt],[0, 1]]) for dt in df['delta_v']]

    # create Q and R matrices
    df['R'] = [np.identity(n_measurement_var, dtype=float) * ((bw*err_obs_pos)**2) for bw in df['body_weight']]
    df['Q'] = [np_arr([[(bw*bw_perc)**2, 0],[0, (bw*vel_perc)**2]]) for bw in df['body_weight']]

    #TODO: create dynamic R based on scale error as % bodyweight
    filter = KalmanFilter(
        n_state_var=n_state_var,
        n_measurement_inputs=n_measurement_var,
    )
    #* pass dynamic Q (how “non-constant” your weight trend is) and R based on scale error as % bodyweight
    filter.forward(
        data=df['body_weight'],
        R=df['R'],
        Q=df['Q']
    )