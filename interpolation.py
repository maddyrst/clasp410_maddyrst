# Climate 410: Interpolation, Cloning and Forking Code

import numpy as np

# number of grid points
nx = 50
ny = 50
# dimensions of the grid (meters)
Lx = 100
Ly = 50
# make the grid
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)

# define hydropotential
phi = np.zeros((ny, nx))
# left boundary (x=0)
y_left = np.array([0, 10, 25, 50]) # known y
phi_left = np.array([7, 14, 10, 19]) # chose random values
# right boundary (x=Lx)
y_right = np.array([0, 10, 25, 50])
phi_right = np.array([8, 21, 11, 5])

# linear interpolation
phi[:, 0] = np.interp(y, y_left, phi_left)
phi[:, -1] = np.interp(y, y_right, phi_right)