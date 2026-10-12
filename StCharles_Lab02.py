# Lab 2: Ordinary Differential Equations

# To reproduce plots, uncomment the lines creating and displaying the plots.
# (They were commented out for simplicity in testing)

# Note on how I varied the different values: rather than repeatedly call my functions to test the 
# different values, I manually went in and changed the corresponding variables. All results for 
# each test can be found in my lab report (Tasks 1-4).

# import relevant libraries
from scipy.integrate import ode
import numpy as np
import matplotlib.pyplot as plt

# Function from HW3 amended to use Euler's method to solve the Lotka-Volterra equations
def lv_euler(N, a, b, c, d, delta_t, t_max):
    '''
    Parameters
        N: a list or array that contains the two species' current populations 
        capacity: a list or array that contains the capacities of each of the populations 
        a, b, c, d: integers or floats that represent the scale factors in the Lotka-Volterra equations.
        delta_t: an integer or float that represents the size of the time step between each iteration.
        t_max: the total time period for the simulation to iterate through.

    This function uses Euler's method to estimate the behavior of two species populations as they 
    compete for resources. It returns three items: an integer or float that represents the total time
    it calculated over, an array that contains the time at each step in the operation, and an array that
    contains the population values at each time in the operation. 
    '''
    pop1, pop2 = N[0], N[1]
    t = 0 # set initial value to be updated with each iteration

    # create arrays to store the data
    t_list = [t]
    N_list = [[pop1, pop2]]

    while t < t_max: # the loop continues through the specified time range
        dPop1dt = a * pop1 * (1 - pop1) - b * pop1 * pop2
        dPop2dt = c * pop2 * (1 - pop2) - d * pop2 * pop1

        # update the values using Euler's formula
        pop1 = pop1 + (dPop1dt * delta_t)
        pop2 = pop2 + (dPop2dt * delta_t)

        t = t + delta_t # finding the total amount of time

        t_list.append(t)
        N_list.append([pop1, pop2])

    return t, t_list, N_list

N = [0.3, 0.6]
a, b, c, d = 1, 2, 1, 3
t, t_list, N_list = lv_euler(N, a, b, c, d, 1, 100) 
    # Note: I changed delta_t or a, b, c, d while testing different values here. 
    # See my lab report for the results for each.
pop1 = [item[0] for item in N_list]
pop2 = [item[1] for item in N_list]
# Test lv_euler() by plotting and investigating the behavior
# plt.figure(figsize=(10, 5))
# plt.plot(t_list, pop1, label='Species Population 1', color='mediumorchid')
# plt.plot(t_list, pop2, label='Species Population 2', color='cornflowerblue')
# plt.xlabel('Time')
# plt.ylabel('Species Population')
# plt.title("Species Populations Over Time (Euler's Method)")
# plt.legend()
# plt.show()

# rk45 method for the LV equations
def lotvolt1(t, N, a=1, b=2, c=1, d=3):
    '''
    Parameters
        t: the current time step
        N: a list or array that represents the populations of two species
        a, b, c, d: integers or floats representing the scale factors of the LV equation
    This function computes the Lotka-Volterra rates for two species with populations N at time t.
    '''
    # initialize time rate of change
    dNdT = np.zeros(2)

    # RHS of LV model
    dNdT[0] = a * N[0] * (1 - N[0]) - b * N[0] * N[1]
    dNdT[1] = c * N[1] * (1 - N[1]) - d * N[1] * N[0]

    return dNdT

# Set up times to integrate to and initial conditions for rk45
dt = 1 # years
    # Note: I changed delta_t or a, b, c, d while testing different values here. 
    # See my lab report for the results for each.
t1 = 100 # years
Ninitial = np.array([0.3, 0.6])
# create integrator object
r = ode(lotvolt1).set_integrator('dopri5')
r.set_initial_value(Ninitial,0).set_f_params(1, 2, 1, 3)
# set up output
ts = [0]
Ns = [Ninitial]
while r.successful() and r.t < t1:
    r.integrate(r.t+dt)
    ts.append(r.t)
    Ns.append(r.y)

# set up the rk45 results so they can be plotted
Ns_array = np.array(Ns)
pop1_rk = Ns_array[:, 0]
pop2_rk = Ns_array[:, 1]

# Plot the results from Euler's method and rk45 together to compare for the competitive species
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,5), sharey=True)

# ax1.plot(t_list, pop1, label='Species Population 1', color='mediumorchid')
# ax1.plot(t_list, pop2, label='Species Population 2', color='cornflowerblue')

# ax2.plot(ts, pop1_rk, label='Species Population 1', color='mediumorchid')
# ax2.plot(ts, pop2_rk, label='Species Population 2', color='cornflowerblue')

# fig.supxlabel('Time (years)', fontsize=14)
# fig.supylabel('Species Population', fontsize=14)
# plt.suptitle('Model for Competing Species', fontsize=17)
# ax1.set_title("Species Populations Over Time (Euler's Method)")
# ax2.set_title("Species Populations Over Time (rk45 Method)")
# ax1.legend()
# ax2.legend()
# ax1.grid()
# ax2.grid()
# fig.tight_layout()
# plt.show()

# ------------------------------------- Same thing for predator and prey model -----------------------------
def pred_prey(N, a, b, c, d, delta_t, t_max):
    '''
    Parameters
        N: a list or array that contains the two species' current populations 
        capacity: a list or array that contains the capacities of each of the populations 
        a, b, c, d: integers or floats that represent the scale factors in the Lotka-Volterra equations.
        delta_t: an integer or float that represents the size of the time step between each iteration.
        t_max: the total time period for the simulation to iterate through.

    This function uses Euler's method to estimate the behavior of two species populations, one being prey
    and the other being predators. It returns three items: an integer or float that represents the total time
    it calculated over, an array that contains the time at each step in the operation, and an array that
    contains the population values at each time in the operation.  
    '''
    prey, pred = N[0], N[1]
    t = 0 # set initial value to be updated with each iteration

    # create arrays to store the data
    t_list = [t]
    N_list = [[prey, pred]]

    while t < t_max: # the loop continues through the specified time range
        dPreydt = a * prey * (1 - prey) - b * prey * pred
        dPreddt = c * pred * (1 - pred) + d * pred * prey

        # update the values using Euler's formula
        prey = prey + (dPreydt * delta_t)
        pred = pred + (dPreddt * delta_t)

        t = t + delta_t # finding the total amount of time

        t_list.append(t)
        N_list.append([prey, pred])

    return t, t_list, N_list

N = [0.3, 0.6]
a, b, c, d = 1, 2, 1, 3
t2, t_list2, N_list2 = pred_prey(N, a, b, c, d, 0.05, 100)
pop1_pred = [item[0] for item in N_list2]
pop2_pred = [item[1] for item in N_list2]

# rk45 method for the predator and prey equations
def lotvolt2(t, N, a=1, b=2, c=1, d=3):
    '''
    Parameters
        t: an integer or float representing the current time step
        N: a list or array that represents the populations of two species
        a, b, c, d: integers or floats representing the scale factors of the LV equation
    This function computes the Lotka-Volterra rates for two species with populations N at time t.
    '''
    # initialize time rate of change
    dNdT2 = np.zeros(2)

    # RHS of LV model
    dNdT2[0] = a * N[0] - b * N[0] * N[1]
    dNdT2[1] = -c * N[1] + d * N[0] * N[1]

    return dNdT2


# Set up times to integrate to and initial conditions for rk45
dt = 0.05 # years
t1 = 100 # years
Ninitial = np.array([0.3, 0.6])
# create integrator object
r = ode(lotvolt2).set_integrator('dopri5')
r.set_initial_value(Ninitial,0).set_f_params(1, 2, 1, 3)
# set up output
ts = [0]
Ns = [Ninitial]
while r.successful() and r.t < t1:
    r.integrate(r.t+dt)
    ts.append(r.t)
    Ns.append(r.y)

# set up the rk45 results so they can be plotted
Ns_array = np.array(Ns)
pop1_rk = Ns_array[:, 0]
pop2_rk = Ns_array[:, 1]

# Plot the results from Euler's method and rk45 together to compare
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,5), sharey=True)

# ax1.plot(t_list2, pop1_pred, label='Species Population 1', color='mediumorchid')
# ax1.plot(t_list2, pop2_pred, label='Species Population 2', color='cornflowerblue')

# ax2.plot(ts, pop1_rk, label='Species Population 1', color='mediumorchid')
# ax2.plot(ts, pop2_rk, label='Species Population 2', color='cornflowerblue')

# fig.supxlabel('Time (years)')
# fig.supylabel('Species Population')
# plt.suptitle('Predator vs Prey Model, Initial Populations 0.9 and 0.1')
# ax1.set_title("Species Populations Over Time (Euler's Method)")
# ax2.set_title("Species Populations Over Time (rk45 Method)")
# ax1.legend()
# ax2.legend()
# ax1.grid()
# ax2.grid()
# fig.tight_layout()
# plt.show()

# Plot the phase diagram with prey on the x-axis and predator on the y-axis
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,5), sharey=True)

# ax1.plot(pop1_pred, pop2_pred, color='mediumorchid')
# ax2.plot(pop1_rk, pop2_rk, color='mediumorchid')

# fig.supxlabel('Species 1 Population (Prey)')
# fig.supylabel('Species 2 Population (Predator)')
# plt.suptitle('Phase Diagrams for Prey vs Predator Model')
# ax1.set_title("Species Population using Euler's Method")
# ax2.set_title("Species Population using the rk45 Method")
# ax1.grid()
# ax2.grid()
# fig.tight_layout()
# plt.show()

# ------------------------------------- Solving Lorenz Equations with rk45 ---------------------------------

# Note: when testing initial conditions, I went and changed the values right in this equation. All results
# are found in my lab writeup. 


def lorenz(t, fluid, sigma=1, r=23, b=8/3):
    '''
    Parameters
        t: an integer or float representing the current time step
        fluid: an array containing the variables [x, y, z], which represent the fluid's convection rate, 
            horizontal temperature variation, and vertical temperature variation respectively. 
        sigma: an integer or float representing a constant in the Lorenz equations
        r: an integer or float representing a constant in the Lorenz equations
        b: an integer or float representing a constant in the Lorenz equations
    This function uses the rk45 method to estimate the behavior behavior and predictabiliy of weather 
    systems using the Lorenz equations. It returns an array that contains the rates of change for the
    three fluid variables in the fluid array. 
    '''
    # initialize array with the rates of change for the three variables
    dNdT = np.zeros(3)
    x, y, z = fluid[0], fluid[1], fluid[2]

    # RHS of Lorenz model
    dNdT[0] = -sigma * x + sigma * y
    dNdT[1] = -x * z + r * x - y
    dNdT[2] = x * y - b * z

    return dNdT


# Set up times to integrate to and initial conditions for rk45
dt = 0.05 # years
t1 = 100 # years

fluid_init = np.array([1, 1, 1])
fluid_init_pert = np.array([1+1e-5, 1+1e-5, 1+1e-5])

# r = 23
# create integrator object
r23 = ode(lorenz).set_integrator('dopri5')
r23.set_initial_value(fluid_init,0).set_f_params(10, 23, 8/3)
# set up output
ts23 = [0]
fluid_s23 = [fluid_init]
while r23.successful() and r23.t < t1:
    r23.integrate(r23.t+dt)
    ts23.append(r23.t)
    fluid_s23.append(r23.y)

# set up the rk45 results so they can be plotted
fluid_s23_array = np.array(fluid_s23)
x23_rk = fluid_s23_array[:, 0]
y23_rk = fluid_s23_array[:, 1]
z23_rk = fluid_s23_array[:, 2]

# r = 23, perturbed
# create integrator object
r23_pert = ode(lorenz).set_integrator('dopri5')
r23_pert.set_initial_value(fluid_init_pert,0).set_f_params(10, 23, 8/3)
# set up output
ts23_pert = [0]
fluid_s23_pert = [fluid_init_pert]
while r23_pert.successful() and r23_pert.t < t1:
    r23_pert.integrate(r23_pert.t+dt)
    ts23_pert.append(r23_pert.t)
    fluid_s23_pert.append(r23_pert.y)

# set up the rk45 results so they can be plotted
fluid_s23_pert_array = np.array(fluid_s23_pert)
x23_pert_rk = fluid_s23_pert_array[:, 0]
y23_pert_rk = fluid_s23_pert_array[:, 1]
z23_pert_rk = fluid_s23_pert_array[:, 2]

# r = 25
# create integrator object
r25 = ode(lorenz).set_integrator('dopri5')
r25.set_initial_value(fluid_init,0).set_f_params(10, 25, 8/3)
# set up output
ts25 = [0]
fluid_s25 = [fluid_init]
while r25.successful() and r25.t < t1:
    r25.integrate(r25.t+dt)
    ts25.append(r25.t)
    fluid_s25.append(r25.y)

# set up the rk45 results so they can be plotted
fluid_s25_array = np.array(fluid_s25)
x25_rk = fluid_s25_array[:, 0]
y25_rk = fluid_s25_array[:, 1]
z25_rk = fluid_s25_array[:, 2]

# r = 25, perturbed
# create integrator object
r25_pert = ode(lorenz).set_integrator('dopri5')
r25_pert.set_initial_value(fluid_init_pert,0).set_f_params(10, 25, 8/3)
# set up output
ts25_pert = [0]
fluid_s25_pert = [fluid_init_pert]
while r25_pert.successful() and r25_pert.t < t1:
    r25_pert.integrate(r25_pert.t+dt)
    ts25_pert.append(r25_pert.t)
    fluid_s25_pert.append(r25_pert.y)

# set up the rk45 results so they can be plotted
fluid_s25_pert_array = np.array(fluid_s25_pert)
x25_pert_rk = fluid_s25_pert_array[:, 0]
y25_pert_rk = fluid_s25_pert_array[:, 1]
z25_pert_rk = fluid_s25_pert_array[:, 2]


# Plot the results
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

# ax1.plot(ts23, x23_rk, label='Original Trajectory', color='mediumorchid')
# ax1.plot(ts23_pert, x23_pert_rk, label='Perturbed Trajectory', color='cornflowerblue')
# ax1.set_title('Lorenz Trajectories Over Time (r=23)')
# ax1.set_xlabel('Time')
# ax1.set_ylabel('X Value (Convection Rate)')
# ax1.legend()
# ax1.grid()

# ax2.plot(ts25, x25_rk, label='Original Trajectory', color='mediumorchid')
# ax2.plot(ts25_pert, x25_pert_rk, label='Perturbed Trajectory', color='cornflowerblue')
# ax2.set_title('Lorenz Trajectories Over Time (r=25)')
# ax2.set_xlabel('Time')
# ax2.legend()
# ax2.grid()

# plt.suptitle('Lorenz Trajectories: Convection Rates')
# plt.tight_layout()
# plt.show()