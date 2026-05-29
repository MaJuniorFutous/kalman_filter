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

    def forward(
            self, 
            data: Union[np.ndarray, pd.DataFrame, pd.Series], 
            R: np.array = None,
            Q: np.array = None):
        if not isinstance(data, (np.ndarray, pd.DataFrame, pd.Series)):
            raise TypeError("data must be a numpy array or pandas DataFrame")
        '''Sort ascending by date first, custom_deltaT is the index of the custom deltaT column'''
        if isinstance(data, pd.Series):
            data = data.to_frame()
        if isinstance(data, pd.DataFrame):
            xs = []
            for row in data.itertuples(index=False):
                z = np_arr(row)
                #TODO: arrange row in np array and 2x1 without A, and also pass A to self.predict()
                self._predict(data=...)
                #!! Note: Set R manually when calling self._update() based on R param
        else:
            #TODO: Need to create this for numpy arrays
            return self._fit_predict()
        

if __name__ == '__main__':
    err_obs_pos = 2 # default, standard bathroom scale error += 1% or 2% of current body weight
    default_t, n_state_var = 1, 2
    
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
    df['A'] = df['delta_v'].apply(lambda x: np_arr([[1, x],[0, 1]]))
    #TODO: create dynamic R based on scale error as % bodyweight
    filter = KalmanFilter(
        n_state_var=n_state_var,
        n_measurement_inputs=1,
    )
    #* pass dynamic Q (how “non-constant” your weight trend is) and R based on scale error as % bodyweight
    filter.forward(
        data=df['body_weight'],
        R=np.identity(n_state_var, dtype=float) * err_obs_pos**2,
        Q=np.array([
            [0.1, 0.0],
            [0.0, 0.01]
        ])

    )