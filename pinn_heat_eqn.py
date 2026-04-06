import numpy as np
import matplotlib.pyplot as plt

# Parameters
nx, ny = 50, 50          # number of grid points in x and y
lx, ly = 1.0, 1.0        # domain size
dx = lx / (nx - 1)
dy = ly / (ny - 1)

alpha = 0.01             # thermal diffusivity
dt = 0.0005              # time step
nt = 200                 # number of time steps

# Stability condition for explicit scheme
stability_limit = dx**2 * dy**2 / (2 * alpha * (dx**2 + dy**2))
if dt > stability_limit:
    raise ValueError(
        f"dt is too large for stability. Choose dt <= {stability_limit:.6e}"
    )

# Grid
x = np.linspace(0, lx, nx)
y = np.linspace(0, ly, ny)

# Initial condition
u = np.zeros((nx, ny))

# Hot square in the center
u[nx//4:3*nx//4, ny//4:3*ny//4] = 100.0

# Time-stepping
for n in range(nt):
    u_new = u.copy()

    # Update interior points
    u_new[1:-1, 1:-1] = (
        u[1:-1, 1:-1]
        + alpha * dt * (
            (u[2:, 1:-1] - 2*u[1:-1, 1:-1] + u[:-2, 1:-1]) / dx**2
            + (u[1:-1, 2:] - 2*u[1:-1, 1:-1] + u[1:-1, :-2]) / dy**2
        )
    )

    # Boundary conditions: u = 0 on all edges
    u_new[0, :] = 0
    u_new[-1, :] = 0
    u_new[:, 0] = 0
    u_new[:, -1] = 0

    u = u_new

# Plot final temperature
plt.figure(figsize=(6, 5))
plt.imshow(u.T, origin="lower", extent=[0, lx, 0, ly], cmap="hot", aspect="auto")
plt.colorbar(label="Temperature")
plt.title("2D Heat Equation Solution")
plt.xlabel("x")
plt.ylabel("y")
plt.show()