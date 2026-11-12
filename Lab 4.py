# Lab 4: Snowball Earth

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as LA
from snowball_functions import temp_warm, insolation

# copying down the values from Table 1 in lab manual
S0 = 1370 # solar flux (W/m2)
C = 4.2e6 # heat capacity of seawater (J/m3*K)
rho = 1020 # density of seawater (kg/m3)
r = 6378e3 # earth's radius (m)
lam = 100 # thermal diffusivity (m2/s)
dz = 50 # mixed-layer depth of ocean (m)
sigma = 5.67e-8 # stefan-boltzmann constant (J/m2*s*K4)
emiss = 1.0 # emissivity (range 0-1)
alpha_ice = 0.6 # albedo for ice and snow
alpha_ground = 0.3 # albedo for water and ground

# ----------------------------------------------------Task 1-----------------------------------------------------

# set up the grid
npoints = 18 # 180º divided into 10º slices
dlat = 10 # 10º slices
lats = np.linspace(dlat / 2.0, 180 - dlat / 2, npoints) # place grid points in the middle of lat blocks
dtheta = np.radians(dlat) # convert from degrees to radians
dy = r * dtheta # turns grid spacing to distance in meters (s = r*theta aka arc length)

# set up time
dt = 365 * 24 * 3600 # seconds in a year
years_tot = 10000 # given in lab manual to run to steady state
steps = int(years_tot)

# set up matrix operators
# initialize K and B
K = np.zeros((npoints, npoints)) # second derivative, how the radiation diffuses between latitude blocks
B = np.zeros((npoints, npoints)) # first derivative, gradients for spherical correction (describes how 
    # temp and cross-sectional area change with time)

for i in range(npoints):
    if i==0: # this is the South Pole boundary; implement Neumann boundary condition
        K[i, i] = -2 / (dy**2)
        K[i, i+1] = 2 / (dy**2)
    elif i == npoints - 1: # this is the North Pole boundary; implement Neumann boundary condition
        K[i, i-1] = 2 / (dy**2)
        K[i, i] = -2 / (dy**2)
    else: # all the other points between the two edge cases
        K[i, i-1] = 1 / (dy**2)
        K[i, i] = -2 / (dy**2)
        K[i, i+1] = 1 / (dy**2)

for i in range(npoints): 
    if 0 < i < npoints - 1: # no boundary rows because their fluxes are 0
        B[i, i-1] = -1 / (2*dy)
        B[i, i+1] = 1 / (2*dy)

# spherical area (A_xz)
theta = np.radians(lats)
A_xz = np.pi * ((r + dz)**2 - r**2) * np.sin(theta)
dA_dy = B @ A_xz # the @ symbol multiplies matrices

# implicit solver matrix L
I = np.eye(npoints) # identity matrix
L = I - lam * dt * K
L_inv = LA.inv(L)

# center the latitudes to match the figure in the lab manual
lats_centered = lats - 90 # center the latitudes around the equator

# check to see graph for most basic condition
plt.figure(figsize=(9, 6))
plt.plot(lats_centered, temp_warm(lats))
plt.xlabel('Latitude (º)')
plt.ylabel('Temperature (ºC)')
plt.title('Initial Condition Test')
plt.grid(True)
plt.show()

# basic diffusion (red line in figure 1)
T_basic = temp_warm(lats) # this is the initial warm earth condition
for _ in range(steps):
    T_basic = L_inv @ T_basic

# check to see graph for basic diffusion condition
plt.figure(figsize=(9, 6))
plt.plot(lats_centered, T_basic)
plt.xlabel('Latitude (º)')
plt.ylabel('Temperature (ºC)')
plt.title('Basic Diffusion Test')
plt.grid(True)
plt.show()

# basic diffusion with spherical correction (yellow line in figure 1)
T_sphere = temp_warm(lats)
for _ in range(steps):
    f_term = (lam * dt / A_xz) * (B @ T_sphere) * (B @ A_xz)
    T_sphere = L_inv @ (T_sphere + f_term)

# check to see graph for basic diffusion plus spherical correction condition
plt.figure(figsize=(9, 6))
plt.plot(lats_centered, T_sphere)
plt.xlabel('Latitude (º)')
plt.ylabel('Temperature (ºC)')
plt.title('Diff. + Spherical Correction Test')
plt.grid(True)
plt.show()

# basic diffusion with spherical correction and radiative forcing (green line in figure 1)
T_rad = temp_warm(lats)
albedo = np.full(npoints, alpha_ground)
Sy = insolation(S0, lats) 

# loop through the system year by year
for _ in range(steps):
    # spherical correction factor
    f_term = (lam * dt / A_xz) * (B @ T_rad) * (B @ A_xz)
    # radiative forcing factor
    rad_forcing = (dt / (rho * C * dz)) * (Sy * (1 - albedo) - emiss * sigma * (T_rad + 273.15)**4)
    # implicit time step solver (solve for temp profile at each time step)
    T_rad = L_inv @ (T_rad + f_term + rad_forcing)

# check to see graph for basic diffusion plus spherical correction plus radiative forcing condition
plt.figure(figsize=(9, 6))
plt.plot(lats_centered, T_rad)
plt.xlabel('Latitude (º)')
plt.ylabel('Temperature (ºC)')
plt.title('Diff. + SphCorr + Radiative Test')
plt.grid(True)
plt.show()

# plot all lines together (it should match figure 1)

plt.figure(figsize=(9,6))
plt.plot(lats_centered, temp_warm(lats), label = 'Initial Condition', color='dodgerblue')
plt.plot(lats_centered, T_basic, label='Basic Diffusion', color='crimson')
plt.plot(lats_centered, T_sphere, label='Diff. + Spherical Correction', color='orange')
plt.plot(lats_centered, T_rad, label='Diff. + SphCorr + Radiative', color='forestgreen')
plt.xlabel('Latitude (º)')
plt.ylabel('temperature (ºC)')
plt.title('Temps and Lats for Snowball Earth with Different Corrections')
plt.grid(True)
plt.legend()
plt.show()

# ----------------------------------------------------Task 2-----------------------------------------------------
# find lam and emiss that looks like the curve produced by temp_warm(lats)
lam_test = 55 # trying range 0-150
emiss_test = 0.7 # trying range 0-1
# these above values are the ones that matched temp_warm() the best, so they'll be used for the rest of the lab

# rewrite implicit solver L but using lam_test
I = np.eye(npoints) # identity matrix
tuned_L = I - lam_test * dt * K

# warm earth initial condition
tuned_temp = temp_warm(lats)
albedo = np.full(npoints, alpha_ground) # keep albedo constant
Sy = insolation(S0, lats)

# loop through time to steady state (10,000 years)
for _ in range(steps):
    f_term = (lam_test * dt / A_xz) * (B @ tuned_temp) * (B @ A_xz)
    rad_forcing = (dt / (rho * C * dz)) * (Sy * (1 - albedo) - emiss_test * sigma * (tuned_temp + 273.15)**4)
    rhs = tuned_temp + f_term + rad_forcing # right hand side
    tuned_temp = LA.solve(tuned_L, rhs)

# plot both tuned result and temp_warm curves to compare
lats_centered = lats - 90 # center around equator

plt.figure(figsize=(9, 6))
plt.plot(lats_centered, temp_warm(lats), label='temp_warm() result', color='dodgerblue')
plt.plot(lats_centered, tuned_temp, label=f'Model ($\lambda$={lam_test}, $\epsilon$={emiss_test})', 
         color='purple')
plt.xlabel('Latitude (º)')
plt.ylabel('Temperature (ºC)')
plt.title('Tuned Model vs temp_warm() Result')
plt.grid(True)
plt.legend()
plt.show()

# ----------------------------------------------------Task 3-----------------------------------------------------
# look at how initial conditions affect equilibrium solution (nonconstant albedo, different earth temps)

# update with values from Task 2
lam_tuned = 55
emiss_tuned = 0.7

L_tuned = I - lam_tuned * dt * K
Sy = insolation(S0, lats)

def dynamic_albedo(initial_temp, initial_albedo):
    '''
    Parameters
        initial_temp: an array of values that represents the temperatures of each latitude block
        initial_albedo: an array of values that represents the albedos of each latitude block

    This function calculates the temperatures of each latitude block after 10000 years given an array of initial
    temperatures and an array of initial albedos. It loops through each time step and determines the albedo for
    each block as well as the temperature, using all the conditions (diffusion, spherical correction, and 
    radiative forcing). It returns two arrays, one of temperatures for each latitude and one of albedos for each
    latitude.
    '''
    T = initial_temp.copy()
    albedo = np.full(npoints, initial_albedo)

    for _ in range(steps): # steps is still 10000 years
       # update albedo based on conditions
       loc_ice = T <= -10
       albedo[loc_ice] = alpha_ice
       albedo[~loc_ice] = alpha_ground 

       # now go through same steps as before
       f_term = (lam_tuned * dt / A_xz) * (B @ T) * (B @ A_xz)
       rad_forcing = (dt / (rho * C * dz)) * (Sy * (1 - albedo) - emiss_tuned * sigma * (T+273.15)**4)
       rhs = T + f_term + rad_forcing
       T = LA.solve(tuned_L, rhs)

    return T, albedo

# First: hot earth, everywhere is 60º
T_hot_initial = np.full(npoints, 60)
T_hot_equil, _ = dynamic_albedo(T_hot_initial, alpha_ground)

# Second: cold earth, everywhere is -60º
T_cold_initial = np.full(npoints, -60)
T_cold_equil, _ = dynamic_albedo(T_cold_initial, alpha_ice)

# Third: flash freeze the earth, use warm earth from Tasks 1 and 2 and set albedo to 0.6
T_warm_equil = T_rad
T_flash_equil, _ = dynamic_albedo(T_warm_equil, alpha_ice)

# plot the results
lats_centered = lats - 90

plt.figure(figsize=(9, 6))
plt.plot(lats_centered, T_hot_equil, label='Hot Earth Equilibrium', color='skyblue')
plt.plot(lats_centered, T_cold_equil, label='Cold Earth Equilibrium', color='navy')
plt.plot(lats_centered, T_flash_equil, label='Flash Freeze Equilibrium', color='blue')
plt.xlabel('Latitude (º)')
plt.ylabel('Temperature (ºC)')
plt.title('Initial Conditions with Dynamic Albedo')
plt.grid(True)
plt.legend()
plt.show()

# ----------------------------------------------------Task 4-----------------------------------------------------
# looking at impact of solar forcing

# define gamma values to use later
gamma_up = np.arange(0.4, 1.41, 0.05) # stepping up
gamma_down = np.arange(1.35, 0.39, -0.05) # stepping down

def diff_gammas(initial_temp, gamma, nsteps):
    '''
    Parameters
        initial_temp: an array of values that represents the temperatures of each latitude block
        gamma: an integer or float that represents the solar forcing multiplier (gamma)
        nsteps: an integere representing the number of steps the function will loop through

    This function uses initial temperatures at each latitiude and the given gamma value to calculate the 
    temperatures of each latitude after nsteps. It returns an array of temperatures corresponding to each 
    latitude block. 
    '''
    T = initial_temp.copy()
    albedo = np.full(npoints, alpha_ground)
    Sy_gamma = gamma * insolation(S0, lats) 

    for _ in range(nsteps):
        loc_ice = (T <= -10.0)
        albedo[loc_ice] = alpha_ice
        albedo[~loc_ice] = alpha_ground

        f_term = (lam_tuned * dt / A_xz) * (B @ T) * (B @ A_xz)
        rad_forcing = (dt / (rho * C * dz)) * (Sy_gamma * (1 - albedo) - emiss_tuned * sigma * (T + 2273.15)**4)
        rhs = T + f_term + rad_forcing
        T = LA.solve(tuned_L, rhs)

    return T

def global_avg_T(T_profile):
    '''
    Parameters
        T_profile: an array that represents the temperatures at each latitude
    
    This function calculates the global average temperature, taking each latitude block's area into account. It 
    returns a float that represents the average global temperature given an array of all the latitude blocks' 
    temperatures. 
    '''
    weight = np.cos(np.radians(lats-90))
    avg_T = np.sum(T_profile * weight) / np.sum(weight)
    return avg_T

# forward steps upward to 1.4
T_current = np.full(npoints, -60)
gamma_forward = []
T_forward = []

for g in gamma_up:
    T_current = diff_gammas(T_current, g, 1000)
    avg_T = global_avg_T(T_current)
    gamma_forward.append(g)
    T_forward.append(avg_T)

# backward steps downward back to 0.4
gamma_backward = []
T_backward = []

for g in gamma_down:
    T_current = diff_gammas(T_current, g, 1000)
    avg_T = global_avg_T(T_current)
    gamma_backward.append(g)
    T_backward.append(avg_T)

# plot the results
plt.figure(figsize=(10, 6))
plt.plot(gamma_forward, T_forward, label='Increasing gamma (Deglaciation Branch)', color='crimson')
plt.plot(gamma_backward, T_backward, label='Decreasing gamma (Glaciation Branch)', color='dodgerblue')
plt.xlabel('Solar Multiplier (gamma)', fontsize=12)
plt.ylabel('Global Average Temperature (ºC)', fontsize=12)
plt.title('Impacts of Solar Forcing Variations', fontsize=14)
plt.grid(True)
plt.legend(fontsize=11)
plt.show()