#!/usr/bin/env python3
"""
Visualize unit balls for p-norms in R^2 (p = 1, 2, ∞).

* L1  → diamond
* L2  → circle
* L∞  → square
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_unit_balls():
    fig, ax = plt.subplots(figsize=(6, 6))

    # --- L2 (Euclidean) ------------------------------------------------------
    theta = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), label=r"$p=2$ (Euclidean)")

    # --- L1 (Manhattan) ------------------------------------------------------
    t = np.linspace(-1, 1, 200)
    diamond_x = np.concatenate([t,  t[::-1], -t, -t[::-1]])
    diamond_y = np.concatenate([1 - np.abs(t),  np.abs(t) - 1,
                                -1 + np.abs(t), -np.abs(t) + 1])
    ax.plot(diamond_x, diamond_y, label=r"$p=1$ (Manhattan)")

    # --- L∞ (Chebyshev) ------------------------------------------------------
    square = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]])
    ax.plot(square[:, 0], square[:, 1], label=r"$p=\infty$ (Chebyshev)")

    # --- Decorations ---------------------------------------------------------
    ax.set_aspect("equal", adjustable="box")
    ax.axhline(0, lw=0.5, color="black")
    ax.axvline(0, lw=0.5, color="black")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_title("Unit balls for different p-norms (radius = 1)")
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_unit_balls()
