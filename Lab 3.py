# Lab 3: Earth's Energy Balance

import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------- Task 1 ---------------------------------------------------
def atm_flux(nlayers, F_solar, debug=False):
    '''
    Parameters
        nlayers: an integer that specifies the number of layers of the atmosphere
        F_solar: an integer or float that specifies the amount of incoming solar flux
        debug: an added feature that, when set to True, prints lines to help in debugging the function

    This function returns a dictionary of values containing the fluxes for each atmospheric layer
    and earth's surface in W/m^2. These values can then be plugged into the Stefan-Boltzmann equation to 
    solve for temperature. 
    '''
    # start by initializing A and b arrays (Ax=b)
    A = np.zeros([nlayers+1, nlayers+1])
    b = np.zeros(nlayers+1)

    # set the diagonal to -2, according to the pattern found in lecture
    np.fill_diagonal(A, -2)
    # set the surrounding diagonals to 1, according to the pattern found in lecture
    np.fill_diagonal(A[0:, 1:], 1)
    np.fill_diagonal(A[1:, 0:], 1)
    # the first row of A is different, so set the first element to its value of -1
    A[0,0] = -1
    # the first element of b is the solar flux
    b[0] = -F_solar

    # solve the matrix to get the fluxes
    fluxes = np.linalg.solve(A, b)

    # checking for bugs (when debug=True)
    N = nlayers+1
    if debug:
        for i in range(N):
            for j in range(N):
                print(f'A[i={i},j={j}] = {A[i, j]}')

    # organizing the results so they're easier to understand
    result = {"F_E": fluxes[0]}
    for i in range(1, nlayers + 1):
        result[f"F_a{i}"] = fluxes[i]
    
    return result

# run the function and test it
test = atm_flux(3, 300) # note: I set debug to True, but later set back to False to stay organized.
print(test)

# convert these fluxes to temperatures using the Stefan-Boltzmann equation
def flux2temp(flux):
    '''
    Parameters
        flux: a dictionary of fluxes from using atm_flux()

    This function uses the Stefan-Boltzmann equation to convert a given dictionary or array of flux 
    values to temperatures for their corresponding layers and returns a dictionary of temperature values.
    '''
    sigma = 5.67e-8 # W/m2/K−4
    temp_results = {}
    for key, flux_val in flux.items():
        temp = (flux_val / sigma) ** (1/4)
        # change the key names
        new_key = key.replace('F', 'T')
        # assign the temp values to the new dictionary
        temp_results[new_key] = temp
    return temp_results

# --------------------------------------------- Task 2 ---------------------------------------------------
def atm_flux_temp(nlayers, F_solar, emissivity, albedo, debug=False):
    '''
    Parameters
        nlayers: an integer that specifies the number of layers of the atmosphere
        F_solar: an integer or float that specifies the amount of incoming solar flux
        debug: an added feature that, when set to True, prints lines to help in debugging the function
        emissivity: an integer or float that represents the emissivity of the atmosphere
        albedo: an integer or float that represents the albedo of the planet/atmosphere

    This function returns a dictionary of values representing the temperatures for each atmospheric layer
    and earth's surface in Kelvin.
    '''
    # start by initializing A and b arrays (Ax=b)
    A = np.zeros([nlayers+1, nlayers+1])
    b = np.zeros(nlayers+1)

    # set the diagonal to -2, according to the pattern found in lecture
    np.fill_diagonal(A, -2)

    # the first element of b is the solar flux
    b[0] = -(F_solar / 4) * (1 - albedo)
    
    # set the initial values
    A[0,0] = -1
    for j in range(1, nlayers+1):
        A[0,j] = (1 - emissivity) ** (j - 1)

    # account for the atmospheric absorbance with emissivity
    for i in range(1, nlayers+1):
        A[i, i] = -2 # setting diagonal pattern
        A[i, 0] = emissivity * ((1 - emissivity) ** (i - 1)) # flux from surface
    # flux from all the other atm layers
        for j in range(1, nlayers+1):
            if i == j:
                continue
            num_between = abs(i - j) - 1
            A[i, j] = emissivity * ((1 - emissivity) ** num_between)

    # invert A and multiply by b
    Ainv = np.linalg.inv(A)
    fluxes = np.matmul(Ainv, b)

    # checking for bugs (when debug=True)
    N = nlayers+1
    if debug:
        for i in range(N):
            for j in range(N):
                print(f'A[i={i},j={j}] = {A[i, j]}')

    sigma = 5.67e-8

    # put results for all the layers in a dictionary
    result = {}
    # results for earth's surface (which doesn't involve emissivity)
    result["T_E"] = (fluxes[0] / sigma) ** (1/4)
    # results for atmosphere layers (which use emissivity)
    for i in range(1, nlayers + 1):
        result[f"T_a{i}"] = (fluxes[i] / (emissivity * sigma)) ** (1/4)

    return result

# validate this model with values from class
test = atm_flux_temp(0, 1350, 1.0, 0.33)
print("This is the surface temp with a 0-layer atmosphere:", test["T_E"])
# the answer should be 251.3K

# --------------------------------------------- Task 3 ---------------------------------------------------
# Experiment 1: 1-layer atmosphere with different emissivities
 
emissivities = np.linspace(0.0, 1.0) # initialize emissivity values
surface_temps = [] # make an empty array to store the values

# get values for emissivity and corresponding surface temps
for ems in emissivities:
    one_layer = atm_flux_temp(1, 1350, ems, 0.33)
    surface_temps.append(one_layer["T_E"]) 

# create the plot of this data
plt.figure(figsize=(8,5))
plt.plot(emissivities, surface_temps)
plt.xlabel('Atmospheric Emissivity ($\epsilon$)')
plt.ylabel('Surface Temperature (K)')
plt.title('Surface Temperature vs Atmospheric Emissivity for a 1-Layer Atm')
plt.grid(True)
plt.show()

# find what emissivity value yeilds a surface temp of 288K
emiss288 = np.interp(288, surface_temps, emissivities)
print(f"The emissivity where earth's surface temp is 288K is {emiss288}")

# Experiment 2: use this emissivity value and vary the number of atmospheric layers
# (same process as Experiment 1)

layers = np.linspace(0, 11, dtype=int)
surface_temp_layers = []

for layer in layers:
    emiss84 = atm_flux_temp(layer, 1350, 0.255, 0.33)
    surface_temp_layers.append(emiss84["T_E"])

plt.figure(figsize=(8,5))
plt.plot(layers, surface_temp_layers)
plt.xlabel('Number of Atmospheric Layers')
plt.ylabel('Surface Temperature (K)')
plt.title('Surface Temperature vs Number of Atmospheric Layers with $\epsilon$ = 0.255')
plt.grid(True)
plt.show()

num_layers = np.interp(288, surface_temp_layers, layers)
print(f"The number of layers where earth's surface temp is 288K is {num_layers}")

# plot altitude vs temperature
number_layers = 4
temp_profile = atm_flux_temp(number_layers, 1350, 0.255, 0.33)
altitude = np.arange(0, number_layers+1) # surface is 0, layer 1 is 1, etc
temperatures = [temp_profile["T_E"]]
for i in range(1, number_layers+1):
    temperatures.append(temp_profile[f"T_a{i}"])

plt.figure(figsize=(6,7))
plt.plot(temperatures, altitude)
plt.xlabel('Temperature (K)')
plt.ylabel('Altitude (Atmospheric Layer Number)')
plt.title('Vertical Temperature Profile for a 4-Layer Atmosphere Model')
plt.yticks(altitude)
plt.grid(True)
plt.show()

# --------------------------------------------- Task 4 ---------------------------------------------------
# Find how many perfectly absorbing layers is needed to match Venus's surface temperature
# similar procedure as in Task 3
venus_layers = np.arange(1, 101, dtype=int)
venus_surface_temps = []

for layer in venus_layers:
    model = atm_flux_temp(layer, 2600, 1.0, 0.71)
    venus_surface_temps.append(model["T_E"])

layer_match = np.interp(700, venus_surface_temps, venus_layers)
print(f"To reach 700K, Venus needs {layer_match} atmospheric layers")

# --------------------------------------------- Task 5 ---------------------------------------------------
# Find the surface temperature in a nuclear winter scenario
nlayers_winter = 5
F_solar_winter = 1350
emissivity_winter = 0.5
albedo_winter = 0.33

# initialize A and b
A_winter = np.zeros([nlayers_winter+1, nlayers_winter+1])
b_winter = np.zeros(nlayers_winter+1)

# set A's conditions just like in atm_flux_temp()
A_winter[0, 0] = -1
for j in range(1, nlayers_winter+1):
    A_winter[0, j] = (1 - emissivity_winter) ** (j - 1)

for i in range(1, nlayers_winter+1):
    A_winter[i, i] = -2
    A_winter[i, 0] = emissivity_winter * ((1 - emissivity_winter) ** (i - 1))
    for j in range(1, nlayers_winter+1):
        if i == j:
            continue
        num_between = abs(i - j) - 1
        A_winter[i, j] = emissivity_winter * ((1 - emissivity_winter) ** num_between)

b_winter[0] = 0.0 # no solar radiation reaches the surface
b_winter[-1] = -(F_solar_winter / 4) * (1 - albedo_winter) # solar flux absorbed at top of atm

# solve this new system
fluxes_winter = np.linalg.solve(A_winter,b_winter)

# put the fluxes into a dictionary so they can go into flux2temp
fluxes_winter_dict = {"F_E": fluxes_winter[0]}
for i in range(1, nlayers_winter+1):
    fluxes_winter_dict[f"F_a{i}"] = fluxes_winter[i]

# convert to temperature
surface_temp_winter = flux2temp(fluxes_winter_dict)

print(f"Earth's surface temperature would be {surface_temp_winter['T_E']}K in a nuclear winter scenario")

# plot altitude vs temperature in the nuclear winter scenario
alt_winter = np.arange(0, nlayers_winter+1)
sigma = 5.67e-8
temps_winter = [(fluxes_winter[0] / sigma) ** (1/4)]

for i in range(1, nlayers_winter+1):
    t_layer = (fluxes_winter[i] / (emissivity_winter * sigma)) ** (1/4)
    temps_winter.append(t_layer)

plt.figure(figsize=(6,7))
plt.plot(temps_winter, alt_winter)
plt.xlabel('Temperature (K)')
plt.ylabel('Altitude (Atmospheric Layer Number)')
plt.title('Vertical Temperature Profile (Nuclear Winter Scenario)')
plt.yticks(alt_winter)
plt.grid(True)
plt.show()