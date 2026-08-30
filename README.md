# Kalman Filter — NumPy / Pandas Implementation

A lightweight, general-purpose Kalman filter implementation for Python using **NumPy** and **Pandas**. The implementation supports configurable state-transition, observation, control, process-noise, and measurement-noise matrices, as well as batch initialization, time-varying matrices, state persistence, and historical state estimates.

## Features

* Linear Kalman filtering with arbitrary state and measurement dimensions
* NumPy and Pandas input support
* Optional control input (`B` and `u`)
* Time-varying:

  * Measurement noise covariance `R`
  * Process noise covariance `q`
  * State-transition matrix `A`
  * Control matrix `B`
  * Control input `u`
* Joseph-form covariance update for improved numerical stability
* Batch/bootstrap initialization from historical observations
* Optional initialization of the prior covariance matrix (`P`) from sample covariance
* Optional removal of cross-covariance terms during initialization
* Save and restore filter checkpoints using `.npz` files
* Optional filtering history
* Automatic state construction from observations using the pseudoinverse of `H`

---

## Installation

The implementation requires:

```bash
pip install numpy pandas
```

The project does not require SciPy.

---

## Mathematical Model

The filter follows the standard discrete linear Kalman filter.

### Prediction

The state prediction is:

$$
x_k^- = A_k x_{k-1} + B_k u_k
$$

When no control input is supplied:

$$
x_k^- = A_k x_{k-1}
$$

The predicted covariance is:

$$
P_k^- = A_k P_{k-1} A_k^T + Q_k
$$

In this implementation, `q` represents the process-noise covariance \(Q\).

### Measurement Update

The innovation covariance is:

$$
S_k = H_k P_k^- H_k^T + R_k
$$

where `R` is the measurement-noise covariance.

The Kalman gain is computed by solving the corresponding linear system rather than explicitly calculating \(S^{-1}\):

$$
K_k = P_k^- H_k^T S_k^{-1}
$$

The state is then updated:

$$
x_k = x_k^- + K_k(z_k - H_k x_k^-)
$$

### Covariance Update

The implementation uses the Joseph form:

$$
P_k =
(I-K_kH_k)P_k^-(I-K_kH_k)^T
+
K_kR_kK_k^T
$$

This is generally preferable to the simpler:

$$
P_k = (I-K_kH_k)P_k^-
$$

because the Joseph form provides better numerical behavior and helps preserve covariance symmetry and positive semidefiniteness in finite-precision arithmetic.

---

## State and Measurement Dimensions

The filter is initialized with:

```python
KalmanFilter(
    n_state_var=...,
    n_measurement_inputs=...
)
```

For example, a two-dimensional state with a single measurement:

```python
kf = KalmanFilter(
    n_state_var=2,
    n_measurement_inputs=1
)
```

This means:

```text
x = [state_1]
    [state_2]
```

and the measurement is:

```text
z = [measurement]
```

The observation matrix `H` therefore has shape:

```text
(1, 2)
```

In general:

| Matrix | Shape                                          |
| ------ | ---------------------------------------------- |
| `x`    | `(n_state_var, 1)`                             |
| `P`    | `(n_state_var, n_state_var)`                   |
| `A`    | `(n_state_var, n_state_var)`                   |
| `H`    | `(n_measurement_inputs, n_state_var)`          |
| `B`    | model-dependent                                |
| `q`    | `(n_state_var, n_state_var)`                   |
| `R`    | `(n_measurement_inputs, n_measurement_inputs)` |
| `K`    | `(n_state_var, n_measurement_inputs)`          |

---

## Basic Example

Suppose the state is:

$$
x =
\begin{bmatrix}
position \\
velocity
\end{bmatrix}
$$

and the only available measurement is position.

A simple constant-velocity model can be constructed as:

```python
import numpy as np

dt = 1.0

A = np.array([
    [1, dt],
    [0, 1]
], dtype=float)

H = np.array([
    [1, 0]
], dtype=float)

q = np.array([
    [0.01, 0],
    [0, 0.01]
], dtype=float)

R = np.array([
    [1.0]
], dtype=float)

kf = KalmanFilter(
    n_state_var=2,
    n_measurement_inputs=1,
    A=A,
    H=H,
    q=q,
    R=R
)
```

Measurements can then be passed to `forward()`:

```python
measurements = np.array([
    [10.0],
    [11.2],
    [12.1],
    [13.3],
    [14.0],
])

estimates = kf.forward(
    measurements,
    return_history=True
)
```

`estimates` contains the estimated state after each measurement update.

For example, conceptually:

```python
[
    (position_1, velocity_1),
    (position_2, velocity_2),
    (position_3, velocity_3),
    ...
]
```

---

## Constructor

```python
KalmanFilter(
    n_state_var,
    n_measurement_inputs,
    A=None,
    H=None,
    B=None,
    R=None,
    q=None,
    save_file="kfilter_save.npz"
)
```

### Parameters

#### `n_state_var`

Number of variables in the state vector.

Example:

```python
n_state_var=2
```

for:

```text
[position, velocity]
```

#### `n_measurement_inputs`

Number of observed/measured variables.

Example:

```python
n_measurement_inputs=1
```

when only position is measured.

#### `A`

State-transition matrix.

Defaults to the identity matrix:

```python
np.identity(n_state_var)
```

#### `H`

Observation/measurement matrix.

Defaults to a zero matrix of shape:

```text
(n_measurement_inputs, n_state_var)
```

A meaningful observation matrix should normally be supplied for an actual model.

#### `B`

Optional control-input matrix.

If supplied together with `u`, the prediction becomes:

```python
x = A @ x + B @ u
```

#### `R`

Measurement-noise covariance.

Defaults to an identity matrix of size `n_measurement_inputs`.

#### `q`

Process-noise covariance.

Defaults to an identity matrix of size `n_state_var`.

#### `save_file`

Default checkpoint filename:

```text
kfilter_save.npz
```

---

# `forward()`

The main public filtering method is:

```python
kf.forward(
    data,
    R=None,
    q=None,
    A=None,
    B=None,
    u=None,
    return_history=False,
    batch_init_n=None,
    batch_init_negate_cv=False,
    initialize_pcm=False,
    unobserved_variance=None,
    bootstrap_transpose=True
)
```

The method performs:

```text
prediction → measurement update
```

for every observation.

---

## Input Data

`data` can be a:

* `numpy.ndarray`
* `pandas.DataFrame`
* `pandas.Series`

For example:

```python
data = np.array([
    [10.0],
    [11.0],
    [12.0],
    [13.0]
])
```

or:

```python
data = pd.DataFrame({
    "measurement": [10.0, 11.0, 12.0, 13.0]
})
```

---

## Returning Only the Final Estimate

By default:

```python
result = kf.forward(data)
```

returns a list containing the final state:

```python
[
    (state_1, state_2, ...)
]
```

---

## Returning the Complete History

Use:

```python
history = kf.forward(
    data,
    return_history=True
)
```

This returns one state estimate for every processed observation:

```python
[
    (x1, x2, ...),
    (x1, x2, ...),
    (x1, x2, ...),
]
```

This is useful for plotting or analyzing the filter trajectory.

Example:

```python
history = kf.forward(data, return_history=True)

positions = [state[0] for state in history]
velocities = [state[1] for state in history]
```

---

# Time-Varying Matrices

`forward()` allows `R`, `q`, `A`, `B`, and `u` to vary for each observation.

For example, a time-varying measurement covariance:

```python
R = np.array([
    [[1.0]],
    [[2.0]],
    [[1.5]],
    [[0.8]]
])
```

can be passed as:

```python
kf.forward(
    data,
    R=R,
    return_history=True
)
```

For each row of `data`, the corresponding matrix is passed to the prediction/update step.

Conceptually:

```text
measurement 1 → R[0]
measurement 2 → R[1]
measurement 3 → R[2]
...
```

The same mechanism is available for:

```text
A
B
q
u
```

This is useful for models where dynamics, uncertainty, or control inputs change over time.

---

# Measurement Noise `R`

The measurement-noise covariance can be specified at construction:

```python
R = np.array([
    [0.5]
])

kf = KalmanFilter(
    n_state_var=2,
    n_measurement_inputs=1,
    H=H,
    A=A,
    R=R
)
```

It can also be overridden during filtering:

```python
kf.forward(
    data,
    R=R
)
```

Inside `_update()`, a scalar `R` is automatically converted into an identity-scaled covariance matrix:

```python
R = np.eye(n_measurement_inputs) * R
```

Thus, for a one-dimensional measurement:

```python
kf._update(z, R=2.0)
```

is equivalent to using:

```python
R = np.array([[2.0]])
```

---

# Process Noise `q`

`q` represents uncertainty in the state-transition model.

For a two-state model:

```python
q = np.array([
    [0.01, 0.0],
    [0.0, 0.1]
])
```

The prediction covariance becomes:

```python
self.P = A @ self.P @ A.T + q
```

A larger `q` generally allows the filter to adapt more quickly to unexpected changes in the underlying state.

A smaller `q` places greater confidence in the state-transition model.

---

# Control Inputs

The implementation optionally supports a control model:

$$
x_k^- = A x_{k-1} + B u_k
$$

Example:

```python
A = np.array([
    [1, 1],
    [0, 1]
], dtype=float)

B = np.array([
    [0.5],
    [1.0]
], dtype=float)

kf = KalmanFilter(
    n_state_var=2,
    n_measurement_inputs=1,
    A=A,
    B=B,
    H=H
)
```

Then:

```python
u = np.array([
    [1.0],
    [1.0],
    [0.5],
    [0.0]
])
```

can be passed to:

```python
kf.forward(
    data,
    u=u
)
```

If both `B` and `u` are present and `u` is nonzero, the control term is included in prediction.

---

# Batch / Bootstrap Initialization

The filter supports initializing its state from an initial batch of observations:

```python
kf.forward(
    data,
    batch_init_n=10
)
```

The first `10` observations are used to construct the initial state, and filtering begins with the remaining observations.

The initial state is constructed using:

```python
pinv(H) @ z
```

or, mathematically:

$$
x_0 = H^+z_0
$$

where \(H^+\) is the Moore-Penrose pseudoinverse.

This allows the implementation to construct an estimate of the full state even when the measurements do not directly observe every state variable.

---

# Initializing `P` From the Data

Set:

```python
initialize_pcm=True
```

when using batch initialization:

```python
kf.forward(
    data,
    batch_init_n=10,
    initialize_pcm=True
)
```

The covariance of the initial observations is calculated and used to initialize the observed portion of `P`.

For example:

```python
temp_pcm = np.cov(...)
```

The implementation identifies observed state variables using:

```python
observed_idx = np.where(
    np.any(self.H != 0, axis=0)
)[0]
```

The resulting covariance is then inserted into the corresponding state-space positions.

---

# Unobserved State Variance

When initializing `P` from observations, state variables that are not directly observed can be assigned a specified variance:

```python
kf.forward(
    data,
    batch_init_n=10,
    initialize_pcm=True,
    unobserved_variance=100.0
)
```

This initializes the covariance matrix approximately as:

```text
             observed     unobserved
observed       sample CV      0
unobserved        0          specified variance
```

The exact observed block is determined by `H`.

---

# Removing Initial Cross-Covariance

Set:

```python
batch_init_negate_cv=True
```

to replace the initialized covariance with its diagonal:

```python
self.P = np.diag(np.diag(self.P))
```

This removes covariance/cross-covariance terms between state variables.

Example:

```python
kf.forward(
    data,
    batch_init_n=10,
    initialize_pcm=True,
    batch_init_negate_cv=True
)
```

`batch_init_negate_cv=True` requires `batch_init_n` to be provided.

---

# Transpose Behavior During Bootstrap

The `bootstrap_transpose` argument controls how `np.cov()` interprets the initial batch:

```python
bootstrap_transpose=True
```

is the default.

This is particularly relevant when observations are organized differently, for example:

```text
rows    = measurements
columns = variables
```

versus:

```text
rows    = variables
columns = measurements
```

For most standard tabular time-series data, the default behavior should be appropriate, but it should be verified against the shape and orientation of the input data.

---

# Internal Methods

Although the following methods are primarily internal implementation details, they describe the filter's architecture.

## `_predict()`

Performs the prediction step:

```python
self._predict()
```

It updates:

```python
self.x
self.P
```

using `A`, `B`, `q`, and optionally `u`.

---

## `_update()`

Performs the measurement update:

```python
self._update(z)
```

It calculates:

```text
innovation covariance
Kalman gain
state update
covariance update
```

The Kalman gain is calculated using `np.linalg.solve()` rather than explicitly computing a matrix inverse.

This is preferable to:

```python
np.linalg.inv(S)
```

for numerical and computational reasons.

---

## `_construct_state()`

Constructs a state estimate from a measurement:

```python
self._construct_state(z)
```

using:

```python
np.linalg.pinv(H) @ z
```

This is particularly useful when the measurement vector does not directly contain every state variable.

---

## `_to_numpy()`

Normalizes input types.

It accepts:

```text
numpy.ndarray
pandas.Series
pandas.DataFrame
None
```

and converts Pandas objects to NumPy arrays.

---

# Saving Filter State

The filter can persist its state to an `.npz` file:

```python
kf.save_state()
```

By default this uses:

```text
kfilter_save.npz
```

The saved checkpoint contains the configured matrices and runtime state, including:

```text
H
B
R
q
A
x
P
K
n_state_var
n_measurement_inputs
```

---

# Custom Save Location

A custom filename can be provided:

```python
kf.save_state(
    file="checkpoints/model_001.npz"
)
```

---

# Saving Selected Matrices

The `matrices` argument controls which static matrices are persisted:

```python
kf.save_state(
    matrices=["A", "H", "R", "q"]
)
```

The runtime matrices and model parameters are included automatically.

---

# Metadata

Optional metadata can be stored with the checkpoint:

```python
metadata = {
    "model": "constant_velocity",
    "version": "1.0",
    "description": "Position tracking model"
}

kf.save_state(
    file="model.npz",
    mdata=metadata
)
```

The metadata is stored in the `.npz` archive under:

```text
metadata
```

---

# Loading a Checkpoint

A checkpoint can be inspected with:

```python
checkpoint = KalmanFilter.extract_checkpoint(
    "model.npz"
)
```

or reconstructed directly as a `KalmanFilter`:

```python
kf = KalmanFilter.from_file(
    "model.npz"
)
```

The restored filter contains the model configuration and static matrices.

---

# Important Note About Runtime State

`save_state()` stores the runtime matrices:

```text
x
P
K
```

However, `from_file()` currently reconstructs the filter using only the model configuration and static matrices:

```text
n_state_var
n_measurement_inputs
A
H
B
R
q
```

Therefore, if the intention is to **resume filtering exactly from the saved point**, `from_file()` should also restore:

```python
x
P
K
```

rather than only reconstructing the filter configuration.

A possible implementation is:

```python
@classmethod
def from_file(cls, file):
    data = cls.extract_checkpoint(file)

    kf = cls(
        n_state_var=int(data["n_state_var"]),
        n_measurement_inputs=int(data["n_measurement_inputs"]),
        A=data.get("A"),
        H=data.get("H"),
        B=data.get("B"),
        R=data.get("R"),
        q=data.get("q")
    )

    if "x" in data:
        kf.x = data["x"]

    if "P" in data:
        kf.P = data["P"]

    if "K" in data:
        kf.K = data["K"]

    return kf
```

This makes checkpoint restoration behave more like a true resume operation.

---

# Example With Pandas

The filter accepts Pandas objects directly:

```python
import pandas as pd

measurements = pd.DataFrame({
    "position": [
        10.2,
        10.9,
        12.1,
        13.0,
        14.2,
        15.1
    ]
})

history = kf.forward(
    measurements,
    return_history=True
)
```

The internal `_to_numpy()` method converts the DataFrame to a NumPy array before filtering.

---

# Complete Example

```python
import numpy as np

dt = 1.0

A = np.array([
    [1.0, dt],
    [0.0, 1.0]
])

H = np.array([
    [1.0, 0.0]
])

q = np.array([
    [0.01, 0.0],
    [0.0, 0.01]
])

R = np.array([
    [1.0]
])

measurements = np.array([
    [10.0],
    [11.0],
    [12.4],
    [13.1],
    [14.2],
    [15.0],
    [16.1],
])

kf = KalmanFilter(
    n_state_var=2,
    n_measurement_inputs=1,
    A=A,
    H=H,
    q=q,
    R=R
)

history = kf.forward(
    measurements,
    return_history=True
)

for state in history:
    position, velocity = state
    print(
        f"position={position:.3f}, "
        f"velocity={velocity:.3f}"
    )
```

---

# Typical Workflow

A typical application follows this pattern:

```text
1. Define the state vector
       ↓
2. Define A, H, Q, and R
       ↓
3. Create KalmanFilter
       ↓
4. Provide measurements to forward()
       ↓
5. Prediction
       ↓
6. Measurement update
       ↓
7. Retrieve estimated state
       ↓
8. Optionally save checkpoint
```

For each observation, the filter performs:

```text
Previous x, P
     │
     ▼
 Prediction
     │
     ├── x = A x + B u
     └── P = A P Aᵀ + Q
     │
     ▼
 Measurement Update
     │
     ├── S = H P Hᵀ + R
     ├── K = P Hᵀ S⁻¹
     ├── x = x + K(z - Hx)
     └── Joseph covariance update
     │
     ▼
Current x, P
```

---

# Choosing `Q` and `R`

The relative values of `q` and `R` strongly influence filter behavior.

### Larger `R`

The filter considers measurements less reliable.

Result:

```text
model prediction ↑
measurement influence ↓
```

### Smaller `R`

The filter considers measurements more reliable.

Result:

```text
model prediction ↓
measurement influence ↑
```

### Larger `q`

The filter considers the state-transition model less certain.

Result:

```text
model flexibility ↑
adaptation to measurements ↑
```

### Smaller `q`

The filter trusts the state-transition model more strongly.

Result:

```text
model stability ↑
adaptation to unexpected changes ↓
```

These parameters generally need to be tuned for the specific application.

---

# Numerical Considerations

The implementation intentionally avoids explicitly calculating:

```python
np.linalg.inv(S)
```

Instead, it uses:

```python
np.linalg.solve(
    S.T,
    self.H @ self.P.T
).T
```

This computes the equivalent Kalman gain without explicitly constructing the inverse.

The covariance update uses the Joseph form:

```python
self.P = (
    (self._I - self.K @ self.H)
    @ self.P
    @ (self._I - self.K @ self.H).T
    + self.K @ R @ self.K.T
)
```

This is more numerically robust than the simplified covariance equation.

---

# Current Limitations and Considerations

### Linear model only

This implementation assumes a linear state-space model:

$$
x_k = A x_{k-1} + B u_k + w_k
$$

and:

$$
z_k = H x_k + v_k
$$

It does not implement an Extended Kalman Filter (EKF) or Unscented Kalman Filter (UKF).

### No automatic parameter estimation

The filter does not automatically estimate optimal `q` or `R`. These must be supplied or tuned externally.

### Input lengths

When time-varying `R`, `q`, `A`, `B`, or `u` are supplied, their first dimension is expected to correspond to the number of observations.

### Stateful operation

`forward()` modifies:

```python
self.x
self.P
self.K
```

Consequently, calling `forward()` multiple times continues from the filter's current state rather than automatically resetting it.

To start a completely fresh filter, create a new `KalmanFilter` instance or explicitly reset its state.

### Matrix dimensions

The implementation validates the shape of `H`, but users should ensure that `A`, `B`, `q`, `R`, and `u` are dimensionally compatible with their model.

---

# API Summary

| Method                 | Purpose                                  |
| ---------------------- | ---------------------------------------- |
| `__init__()`           | Configure the Kalman filter              |
| `forward()`            | Run filtering over observations          |
| `_predict()`           | Perform prediction step                  |
| `_update()`            | Perform measurement update               |
| `_construct_state()`   | Construct state from measurement         |
| `_bootstrap_filter()`  | Initialize state/covariance from a batch |
| `_to_numpy()`          | Convert NumPy/Pandas inputs              |
| `save_state()`         | Save filter/checkpoint state             |
| `extract_checkpoint()` | Open a saved `.npz` checkpoint           |
| `from_file()`          | Reconstruct a filter from a checkpoint   |

---

# License

Add the project's applicable license here, for example:

```text
MIT License
```

if this implementation is intended to be released under the MIT License.

---

# Summary

This implementation provides a compact Kalman filtering framework centered around NumPy arrays while remaining convenient to use with Pandas time-series data.

Its main extension points are:

* configurable state-space models,
* time-varying model/noise matrices,
* control inputs,
* batch initialization,
* covariance initialization,
* historical estimates,
* and persistent model checkpoints.

For a standard linear tracking problem, the core workflow is simply:

```python
kf = KalmanFilter(
    n_state_var=...,
    n_measurement_inputs=...,
    A=A,
    H=H,
    q=q,
    R=R
)

estimates = kf.forward(
    measurements,
    return_history=True
)
```
