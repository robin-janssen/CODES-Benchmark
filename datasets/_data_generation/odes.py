from typing import Any, Dict

import numpy as np


def lotka_volterra(n: np.ndarray) -> np.ndarray:
    """
    Defines the Lotka-Volterra predator-prey model with three predators and three prey.

    Parameters
    ----------
    n : np.ndarray
        Current populations [p1, p2, p3, q1, q2, q3].

    Returns
    -------
    np.ndarray
        Derivatives [dp1/dt, dp2/dt, dp3/dt, dq1/dt, dq2/dt, dq3/dt].
    """
    p1, p2, p3, q1, q2, q3 = n
    derivatives = np.array(
        [
            0.5 * p1 - 0.02 * p1 * q1 - 0.01 * p1 * q2,
            0.6 * p2 - 0.03 * p2 * q1 - 0.015 * p2 * q3,
            0.4 * p3 - 0.01 * p3 * q2 - 0.025 * p3 * q3,
            -0.1 * q1 + 0.005 * p1 * q1 + 0.007 * p2 * q1,
            -0.08 * q2 + 0.006 * p1 * q2 + 0.009 * p3 * q2,
            -0.12 * q3 + 0.008 * p2 * q3 + 0.01 * p3 * q3,
        ]
    )
    return derivatives


def reaction_backup(n: np.ndarray) -> np.ndarray:
    """
    Defines a simple chemical reaction system.

    Parameters
    ----------
    n : np.ndarray
        Current concentrations [s1, s2, s3, s4, s5].

    Returns
    -------
    np.ndarray
        Derivatives [ds1/dt, ds2/dt, ds3/dt, ds4/dt, ds5/dt].
    """
    s1, s2, s3, s4, s5 = n
    derivatives = np.array(
        [
            -0.1 * s1 + 0.1 * s2,
            0.1 * s1 - 0.15 * s2 + 0.05 * s3,
            0.15 * s2 - 0.1 * s3 + 0.03 * s4,
            0.1 * s3 - 0.07 * s4 + 0.01 * s5,
            0.07 * s4 - 0.05 * s5,
        ]
    )
    return derivatives


import numpy as np


def reaction(t: float, n: np.ndarray) -> np.ndarray:
    """
    Defines a chemically motivated reaction system with five species.

    Reactions:
        1. A → B                 (k1)
        2. B + C ↔ D            (k2, k_minus2)
        3. 2D → E               (k3)
        4. E → A + C            (k4)
        5. C → A                (k5)

    Parameters
    ----------
    t : float
        Current time.
    n : np.ndarray
        Current concentrations [A, B, C, D, E].

    Returns
    -------
    np.ndarray
        Derivatives [dA/dt, dB/dt, dC/dt, dD/dt, dE/dt].
    """
    # Unpack concentrations
    s1, s2, s3, s4, s5 = n

    # Define rate constants
    k1 = 0.25  # Rate constant for A → B
    k2 = 0.5  # Forward rate constant for B + C → D
    k_minus2 = 0.25  # Reverse rate constant for D → B + C
    k3 = 0.1  # Rate constant for 2D → E
    k4 = 0.15  # Rate constant for E → A + C
    k5 = 0.05  # Rate constant for C → A

    # Calculate reaction rates
    rate1 = k1 * s1
    rate2_forward = k2 * s2 * s3
    rate2_reverse = k_minus2 * s4
    rate2_net = rate2_forward - rate2_reverse
    rate3 = k3 * s4**2
    rate4 = k4 * s5
    rate5 = k5 * s3

    # Differential equations
    ds1_dt = -rate1 + rate4 + rate5
    ds2_dt = rate1 - rate2_net
    ds3_dt = -rate2_net - rate5 + rate4
    ds4_dt = rate2_net - 2 * rate3
    ds5_dt = rate3 - rate4

    # Return derivatives
    return np.array([ds1_dt, ds2_dt, ds3_dt, ds4_dt, ds5_dt])


import numpy as np


def simple_ode(t: float, n: np.ndarray) -> np.ndarray:
    """
    Defines a numerically stable, non-conservative ODE system with interesting but controlled dynamics.

    Parameters
    ----------
    n : np.ndarray
        Current state [s1, s2, s3, s4, s5].

    Returns
    -------
    np.ndarray
        Derivatives [ds1/dt, ds2/dt, ds3/dt, ds4/dt, ds5/dt].
    """
    s1, s2, s3, s4, s5 = n

    # Rate constants
    k1, k2, k3, k4 = 0.7, 0.3, 0.4, 0.2
    k5, k6 = 0.25, 0.15  # Decay terms
    influx = 0.1  # Constant mass influx

    # Interaction terms (simplified)
    rate1 = k1 * s1 * (1 - s1)  # Logistic-like self-regulation
    rate2 = k2 * s2 * s3  # Simple interaction between s2 and s3
    rate3 = k3 * s4 * (1 - 0.1 * s4)  # Mild quadratic feedback for saturation
    rate4 = k4 * s5 * s2  # Interaction between s5 and s2
    rate5 = k3 * s4

    # Loss terms
    decay1 = k5 * s1
    decay2 = k6 * s3

    # System equations (mass is not conserved)
    ds1_dt = influx + rate1 - decay1  # s1 has an external source and self-regulation
    ds2_dt = rate2 - 0.5 * s2  # s2 grows via interaction with s3 and has linear damping
    ds3_dt = -rate2 + rate5 - decay2  # s3 is consumed by s2 but replenished by s4
    ds4_dt = rate3 - 0.1 * s4  # Simple growth and decay for stability
    ds5_dt = rate4 - 0.2 * s5  # Simple linear decay

    return np.array([ds1_dt, ds2_dt, ds3_dt, ds4_dt, ds5_dt])


def coupled_nonlinear_oscillators(t, state):
    """
    Defines a system of 5 coupled nonlinear harmonic oscillators.

    State Variables:
        state = [x1, v1, x2, v2, x3, v3, x4, v4, x5, v5]

    Parameters:
        t : float
            Current time
        state : np.ndarray
            Current state of the system [x1, v1, x2, v2, x3, v3, x4, v4, x5, v5]

    Returns:
        derivatives : list
            Derivatives [dx1/dt, dv1/dt, ..., dx5/dt, dv5/dt]
    """
    # Unpack the state variables
    x1, x2, x3, x4, x5, v1, v2, v3, v4, v5 = state

    # Define parameters
    # Masses
    m1 = m2 = m3 = m4 = m5 = 1.0

    # Linear spring constants
    k1 = 2.0
    k2 = 2.0
    k3 = 2.0
    k4 = 2.0
    k5 = 2.0

    # Nonlinear spring coefficients (cubic terms)
    alpha1 = 0.5
    alpha2 = 0.5
    alpha3 = 0.5
    alpha4 = 0.5
    alpha5 = 0.5

    # Damping coefficients
    c1 = 0.5
    c2 = 0.5
    c3 = 0.5
    c4 = 0.5
    c5 = 0.5

    # Nonlinear coupling coefficients
    beta13 = 0.05
    beta24 = 0.05
    beta35 = 0.05
    beta41 = 0.05
    beta52 = 0.05

    # Compute derivatives
    # Position derivatives are velocities
    dx1_dt = v1
    dx2_dt = v2
    dx3_dt = v3
    dx4_dt = v4
    dx5_dt = v5

    # Velocity derivatives
    dv1_dt = (
        -c1 * v1 - k1 * (x1 - x2) - alpha1 * (x1 - x2) ** 3 + beta13 * (x3 - x1) ** 2
    ) / m1

    dv2_dt = (
        -c2 * v2
        - k2 * (x2 - x1)
        - alpha2 * (x2 - x1) ** 3
        - k3 * (x2 - x3)
        - alpha3 * (x2 - x3) ** 3
        + beta24 * (x4 - x2) ** 2
    ) / m2

    dv3_dt = (
        -c3 * v3
        - k3 * (x3 - x2)
        - alpha3 * (x3 - x2) ** 3
        - k4 * (x3 - x4)
        - alpha4 * (x3 - x4) ** 3
        + beta35 * (x5 - x3) ** 2
    ) / m3

    dv4_dt = (
        -c4 * v4
        - k4 * (x4 - x3)
        - alpha4 * (x4 - x3) ** 3
        - k5 * (x4 - x5)
        - alpha5 * (x4 - x5) ** 3
        + beta41 * (x1 - x4) ** 2
    ) / m4

    dv5_dt = (
        -c5 * v5 - k5 * (x5 - x4) - alpha5 * (x5 - x4) ** 3 + beta52 * (x2 - x5) ** 2
    ) / m5

    return np.array(
        [dx1_dt, dx2_dt, dx3_dt, dx4_dt, dx5_dt, dv1_dt, dv2_dt, dv3_dt, dv4_dt, dv5_dt]
    )


FUNCS: Dict[str, Dict[str, Any]] = {
    "lotka_volterra": {
        "func": lotka_volterra,
        "tsteps": np.linspace(0, 100, 101),
        "ndim": 6,
        "labels": ["Predator1", "Predator2", "Predator3", "Prey1", "Prey2", "Prey3"],
        "sampling": {
            "space": "log",  # Options: 'linear', 'log'
            "bounds": [
                (1, 10.0),  # Predator1
                (1, 10.0),  # Predator2
                (1, 10.0),  # Predator3
                (1, 10.0),  # Prey1
                (1, 10.0),  # Prey2
                (1, 10.0),  # Prey3
            ],
        },
    },
    "reaction": {
        "func": reaction,
        "tsteps": np.linspace(0, 3, 101),
        "ndim": 5,
        "labels": ["S1", "S2", "S3", "S4", "S5"],
        "sampling": {
            "space": "log",  # Options: 'linear', 'log'
            "bounds": [
                (0.1, 20.0),  # S1
                (0.1, 20.0),  # S2
                (0.1, 20.0),  # S3
                (1, 30.0),  # S4
                (1, 30.0),  # S5
            ],
        },
    },
    "nonlinear_oscillators": {
        "func": coupled_nonlinear_oscillators,
        "tsteps": np.linspace(0, 3, 101),
        "ndim": 10,
        "labels": ["x1", "x2", "x3", "x4", "x5", "v1", "v2", "v3", "v4", "v5"],
        "sampling": {
            "space": "linear",  # Options: 'linear', 'log'
            "bounds": [
                (-1, 1),  # x1
                (-1, 1),  # x2
                (-1, 1),  # x3
                (-1, 1),  # x4
                (-1, 1),  # x5
                (-1, 1),  # v1
                (-1, 1),  # v2
                (-1, 1),  # v3
                (-1, 1),  # v4
                (-1, 1),  # v5
            ],
        },
    },
    "simple_ode": {
        "func": simple_ode,
        "tsteps": np.linspace(0, 10, 101),
        "ndim": 5,
        "labels": ["S1", "S2", "S3", "S4", "S5"],
        "sampling": {
            "space": "log",  # Options: 'linear', 'log'
            "bounds": [
                (0.1, 10.0),  # S1
                (0.1, 10.0),  # S2
                (0.1, 10.0),  # S3
                (0.1, 10.0),  # S4
                (0.1, 10.0),  # S5
            ],
        },
    },
}
