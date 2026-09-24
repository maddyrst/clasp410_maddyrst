# Turning the pseudocode from HW2 into actual code

import numpy as np
import matplotlib.pyplot as plt

# my pseudocode pasted below:
# function that takes the initial coffee temp, the environmental temp, and a defined time step and 
# loops to find the time it takes for coffee temp to equal environment temp

# while the coffee temp is not yet equal to the environmental temperature:

       # T(t=0) = Ti  # initial temp of coffee
       # take a forward step using Euler's method: T(t+deltat)=T(t)-k*deltat(T(t)-Ts)   
            #  now you have the temp after one step
       # find the rate of change at the next step forward
       # calculate what time it is now (t + deltat)
       # use the new temp and rate of change found and take the next step with those values
       # continue this until the coffee temp equals the environmental temp

# return what the final time is (t + however many deltat)



def coffee_time(T_coffee, T_final, T_env, delta_t):
    '''
    Parameters
        T_coffee: an integer or float that represents the starting temperature of the coffee. This 
            is the value the loop with use to begin calculating and iterating.
        T_final: an integer or float that represents the drinkable temperature of the coffee. This
            is the value where the loop will stop, once the temperature of the coffee reaches this
            specified value. 
        T_env: an integer or float that represents the temperature of the environment the coffee is
            in. This value is used in the calculation that the loop performs. 
        delta_t: an integer or float that represents the size of the time step between each 
            iteration. 

    This function is used to calculate the amount of time it takes for a cup of coffee to cool down 
    to a drinkable temperature, given the initial temperature of the coffee, the desired drinking 
    temperature, and the temperature of the environment. This is calculated using Euler's method and
    an ordinary differential equation provided to us in class. Given a specified time step value 
    (delta_t), the function loops through the equation, calculating the new coffee temperature at each
    time step and determining the total amount of time that has passed. The total time is returned, 
    as well as an array of coffee temperatures and an array of time, so that the results can then 
    be plotted. 
    '''
    k = 0.1 # constant
    T_current = T_coffee # this will change with each iteration
    t = 0 # this will change with each iteration

    # create arrays to store the data so it can be plotted later
    t_list = [t]
    T_list = [T_current]

    while T_current > T_final: # the loop continues until the coffee temp = drinkable temp
        t = t + delta_t # finding the total amount of time
        T_new = T_current - (k * delta_t * (T_current - T_env)) # equation given in class
        T_current = T_new

        t_list.append(t)
        T_list.append(T_current)

    return t, t_list, T_list

class_values = coffee_time(145, 80, 18, 10)
#print(f"It took {class_values} minutes for the coffee temp to cool to the drinkable temperature.")


# Create a figure showing the results
# plot time versus coffee temp for each delta_t
T_start = 145
T_drink = 80
T_room = 18
delta_t_vals = [0.5, 2, 5, 10] # testing four different time step sizes

fig, axes = plt.subplots(2, 2, figsize=(7,5)) # four plots for four delta_t values
flat_axes = axes.flatten() # flattening so it can be looped through to plot

# looping through the delta t values to plot more efficiently
for i, dt in enumerate(delta_t_vals):
    ax = flat_axes[i]
    # use "_" to ignore the returned value of t, since it's not needed here
    _, times, temps = coffee_time(T_start, T_drink, T_room, dt)
    # plot the values from the function and label the graphic
    ax.plot(times, temps, color='mediumorchid', linestyle='-')
    ax.scatter(times, temps, color='mediumorchid')
    ax.set_title(f'Coffee Temp vs Time with Δt = {dt}', fontsize=12)
    ax.grid(True)

fig.supxlabel("Time (minutes)", fontsize=14)
fig.supylabel("Temperature (ºC)", fontsize=14)
fig.suptitle("Coffee Cooling over Time with Different Time Steps", fontsize=16)
plt.tight_layout()
plt.show()