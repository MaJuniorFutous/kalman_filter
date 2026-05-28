from typing import Any, Union

import numpy as np, pandas as pd


def np_arr(x: list): return np.array(x, dtype=float)

class KalmanFilter:

    #TODO: Initialize these to zeros or identoties based on shape/structure of X
    def __init__(
            self,
            C: np.array,
            H: np.array,
            u: np.array,
            q: np.array,
            R: np.array,
            P: np.array,
            z: np.array,
            A: np.array = None,
            B: np.array = None):

        self.C = C  # observation matrix
        self.A = A  # state transition matrix
        self.H = H  # observation matrix
        self.B = B  # control matrix
        self.u = u # control vector
        self.q = q  # process noise covariance
        self.R = R  # measurement/observation noise covariance
        self.P = P
        self.z = z

        #* A matrix needs to be a 2x2 if state is 2x1, 3x3 if state is 3x1, etc
    
    #TODO: Determine correct typing enforcement
    def _predict(
            self, 
            data: np.array, 
            A: np.array = None, 
            B: np.array = None,
            q: np.array = None,
            u = None,
            w = None):
        if B is None:
            B = self.B
        if A is None:
            A = self.A
        if q is None:
            q = self.q

        #TODO: Check all values that can be scalars are and if so, set their shape
        # reshape array
        self.X = np.array([[i] for i in data])

        # state prediction
        if B is not None and u is not None:
            self.x = A@self.x + B@u + w  # w should match state
        else:
            self.x = A@self.x + w

        #* Main equation X_k = X_k-1 + V_k-1*deltaT
        # Process covariance prediction
        self.P = A@self.P@A.T + q  # q should match PCM and A

        #TODO: Hold on to prior PCM (PCM_k-1) and state prior to update

    def _update(self):
        # Kalman Gain
        # equation (deprecated, we dont actually need to calculate the inverse)
        # S = H@predicted_pcm@H.T + R # inovation covariance
        # K = predicted_pcm@H.T@np.linalg.inv(S)

        '''
        Note: np.linalg.solve(A, B) only solves equation in the form: AX = B
        '''
        S = self.H@self.P@self.H.T + self.R #inovation covariance
        K = np.linalg.solve(S.T, (self.H@self.P.T)).T

        # Observation/measurement update
        updated_observation = self.C@np_arr([[i] for i in data]) + np_arr([[0], [0]])

    def forward(self, data: Union[np.ndarray, pd.DataFrame], custom_deltaT_idx: int = None):
        if not isinstance(data, (np.ndarray, pd.DataFrame)):
            raise TypeError("data must be a numpy array or pandas DataFrame")
        '''Sort ascending by date first, custom_deltaT is the index of the custom deltaT column'''
        if isinstance(data, pd.DataFrame):
            if custom_deltaT_idx:
                df['A'] = df['delta_v'].apply(lambda x: np_arr([[1, x],[0, 1]]))
            df.drop(columns=df.columns[custom_deltaT_idx], inplace=True)
            xs = []
            for row in data.itertuples(index=False):
                z = np_arr(row)
                #TODO: arrange row in np array and 2x1 without A, and also pass A to self.predict()
                self._predict(data=...)
        else:
            #TODO: Need to create this for numpy arrays
            return self._fit_predict()
        

if __name__ == '__main__':
    err_obs_pos = 2 # default, standard bathroom scale error += 1% or 2% of current body weight
    default_t = 1
    
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
    filter = KalmanFilter(
        C=np_arr([[1, 0]]),
        A=np_arr([
            [1, default_t],
            [0, 1]
        ]),
        H=np_arr([[1,0]]),
        B=np_arr([
            [0],
            [0]
        ]),
        u=np_arr([[0]]),
        q=np.zeros((2,2), dtype=float),
        R=np_arr([[err_obs_pos**2]]),
        P=np_arr([
            [2, 0],
            [0, 0.5]
        ]),
        z=np_arr([
            [0],
            [0]
        ])
    )
    filter.forward(
        data=df.drop('datetime', axis=1),
        custom_deltaT_idx=1  # index of delta value
    )