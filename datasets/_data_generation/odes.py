from typing import Any, Dict

import numpy as np


def lotka_volterra_backup(t: float, n: np.ndarray) -> np.ndarray:
    """
    Defines the Lotka-Volterra predator-prey model with three predators and three prey.

    Parameters
    ----------
    t : float
        Current time.
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


def lotka_volterra(t: float, n: np.ndarray) -> np.ndarray:
    """
    Defines the Lotka-Volterra predator-prey model with three predators and three prey.

    Parameters
    ----------
    t : float
        Current time.
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


def reaction(t: float, n: np.ndarray) -> np.ndarray:
    """
    Defines a simple chemical reaction system.

    Parameters
    ----------
    t : float
        Current time.
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
