import numpy as np, pandas as pd

from src.kalman_filter.kfilter import KalmanFilter
from src.kalman_filter.utils import np_arr


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

    filter = KalmanFilter(
        n_state_var=n_state_var,
        n_measurement_inputs=n_measurement_var,
        H=np_arr([[1, 0]])
    )

    #* pass dynamic Q (how “non-constant” your weight trend is) and R based on scale error as % bodyweight
    results = filter.forward(
        # data=df[['body_weight', 'delete']],
        data=df['body_weight'],
        R=df['R'],
        q=df['q'],
        A=df['A'],
        # B=df['B'],
        # u=df['u'],
        return_history=True,
        batch_init_n=batch_init_n,
        batch_init_negate_cv=True
    )

    filter.save_state(matrices='H', mdata={'date': df['datetime'].iloc[-1].strftime("%Y-%m-%d")})
    prev_filter = KalmanFilter.extract_checkpoint(file='kfilter_save.npz')
    newk_filter = KalmanFilter.from_file(file='kfilter_save.npz')