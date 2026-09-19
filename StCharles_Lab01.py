## Lab 1: Forest Fires and Disease Spread
'''
This file contains tools and scripts for completing Lab 1 for Climate 410.

To reproduce the graphics at the bottom of this file, uncomment the plt.show() lines.
'''


# Import the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random


# Define variables
nx, ny = 3, 3          # number of cells in x and y directions
prob_spread = 1.0      # chance to spread to adjacent cells
prob_bare = 0.0        # chance of cell to start as bare patch
prob_ignite = 0.0      # chance of cell to start on fire

burnt = 1
forested = 2
burning = 3

## ------------------------------------ TASK 1 -----------------------------------------

def forestfire(forest, prob_spread):
    '''
    Parameters
        forest: a grid of integers valued 1 (burnt), 2 (forested), or 3 (burning). (Note that 
            this function can be used for different scenarios. So broadly, 1 can indicate a cell
            already impacted, 2 can be a cell unaffected, and 3 can be a cell actively 
            experiencing the condition.)
        prob_spread: a float that represents the probability of spread to the next cell

    This function spreads the imposed condition to the cells around it, given a grid, 
    probability of spread, and initial starting point(s). From each starting point given,
    it will investigate the cells around it and randomly decide if the condition spreads to 
    those cells, based on the probability given. As the condition spreads with each iteration, 
    the cells that have already experienced the condition will be set to value 1.  
    '''
    ny, nx = forest.shape

    old_forest = forest.copy()
    new_forest = forest.copy()

    for i in range(ny):
        for j in range(nx):
            # check for forested areas and spread the fire
            if old_forest[i, j] == 3:  
                # this checks if the neighbors are burning, and if conditions are valid               
                if (i>0) and (old_forest[i-1,j] == 2) and (random.random() < prob_spread):
                    new_forest[i-1,j] = 3 # sets that point we just checked to burning
                if (i<ny-1) and (old_forest[i+1,j] == 2) and (random.random() < prob_spread):
                    new_forest[i+1,j] = 3
                if (j>0) and (old_forest[i,j-1] == 2) and (random.random() < prob_spread):
                    new_forest[i,j-1] = 3
                if (j<nx-1) and (old_forest[i,j+1] == 2) and (random.random() < prob_spread):
                    new_forest[i,j+1] = 3
                new_forest[i, j] = 1

    return new_forest

# Set up a 3x3 array of integers to test the model
# set each value in the array to 2 (indicating that they're forested)
forest = np.zeros([nx, ny]) + 2
# set the center cell to 3 (burning)
forest[1, 1] = 3
iteration1 = forestfire(forest, prob_spread)
iteration2 = forestfire(iteration1, prob_spread)
#print(iteration1)
#print(iteration2)

# Test it with a grid wider than it is tall (3x5)
bigger_forest = np.zeros([3, 5]) + 2
bigger_forest[1, 2] = 3
test_big_forest = forestfire(bigger_forest, prob_spread) # iteration 1
test_big_forest2 = forestfire(test_big_forest, prob_spread) # iteration 2
test_big_forest3 = forestfire(test_big_forest2, prob_spread) # iteration 3
#print(test_big_forest)

# ------------------------------------ TASK 2 -----------------------------------------

# Experiment 1: Vary prob_spread to see how the prob. of spread affects how the fire spreads
prob_spread_2 = 0.25
prob_spread_3 = 0.5
prob_spread_4 = 0.75

exp_forest = forestfire(bigger_forest, prob_spread_2)
exp_forest2 = forestfire(exp_forest, prob_spread_2)
#print(exp_forest2)

exp_forest = forestfire(bigger_forest, prob_spread_3)
exp_forest2 = forestfire(exp_forest, prob_spread_3)
#print(exp_forest2)

exp_forest = forestfire(bigger_forest, prob_spread_4)
exp_forest2 = forestfire(exp_forest, prob_spread_4)
print(exp_forest2)

# Experiment 2: Vary prob_bare and prob_ignite to see how the forestation of an area affects 
# how the fire spreads
def assign_random(forest, prob_bare, prob_ignite):
    '''
    Parameters
        forest: a grid of integers valued 1 (burnt), 2 (forested), or 3 (burning). 
            (Note that this function can be used for different scenarios. So broadly, 1 can 
            indicate a cell already impacted but cannot experience the condition again, 2 can 
            be a cell unaffected, and 3 can be a cell actively experiencing the condition.)
        prob_bare: a float that determines the probability that a cell is already bare, so that 
            the fire does not spread to it in the simulation.
        prob_ignite: a float that determines the probability that a cell is burning and able to 
            spread the fire. This is where the fire will originate, and the simulation will 
            spread the condition from the point(s) determined here. 
    
    This function initializes the grid given to forestfire by randomly assigning cells 
    certain conditions that will affect how the fire spreads. It chooses cells to be bare 
    (preventing the fire from spreading to those cells) or burning (the origin points of the
    spreading fire). 
    '''
    for i in range(ny):
        for j in range(nx):
            if random.random() < prob_bare:
                random_i = np.random.randint(ny)
                random_j = np.random.randint(nx)
                forest[random_i, random_j] = 1
            if random.random() < prob_ignite:
                random_i = np.random.randint(ny)
                random_j = np.random.randint(nx)
                forest[random_i, random_j] = 3

    return forest

# testing assign_random
prob_ignite = 0.8
prob_bare = 0.8
test = assign_random(bigger_forest, prob_bare, prob_ignite)
print(test)

# ------------------------------------ TASK 3 -----------------------------------------

# Use our wildfire model for the spread of disease
dead = 0
immune = 1
healthy = 2
sick = 3

def assign_random_people(population, prob_immune, prob_sick):
    '''
    Parameters
        population: a grid of integers valued 0 (dead), 1 (immune), 2 (healthy), or 3 (sick). 
            (Note that this function can be used for different scenarios. So broadly, 0 can mean 
            a cell already impacted and cannot experience the conditon again, 1 can indicate a 
            cell already impacted but cannot experience the condition again, 2 can be a cell 
            unaffected, and 3 can be a cell actively experiencing the condition.)
        prob_immune: a float that determines the probability that a person is already immune 
            to the spreading disease, so that the disease does not spread to them in the 
            simulation.
        prob_sick: a float that determines the probability that a person is sick and able to 
        spread the disease. This is where the disease will originate, and the simulation will 
        spread the condition from the point(s) determined here. 

    This function initializes the grid given to disease_spread by randomly assigning cells 
    certain conditions that will affect how the disease spreads. It chooses cells to be immune 
    (preventing the disease from spreading to those cells) or sick (the origin points of the
    spreading disease). 
    '''

    ny, nx = population.shape
    initialized_pop = population.copy()

    for i in range(ny):
        for j in range(nx):
            if random.random() < prob_immune:
                random_i = np.random.randint(ny)
                random_j = np.random.randint(nx)
                initialized_pop[random_i, random_j] = 1
            if random.random() < prob_sick:
                random_i = np.random.randint(ny)
                random_j = np.random.randint(nx)
                initialized_pop[random_i, random_j] = 3

    return initialized_pop



def disease_spread(population, prob_spread, prob_survive):
    '''
    Parameters
        population: a grid of integers valued 0 (dead), 1 (immune), 2 (healthy), or 3 (sick). 
            (Note that this function can be used for different scenarios. So broadly, 0 can mean 
            a cell already impacted and cannot experience the conditon again, 1 can indicate a 
            cell already impacted but cannot experience the condition again, 2 can be a cell 
            unaffected, and 3 can be a cell actively experiencing the condition.)
        prob_spread: a float that represents the probability of spread to the next cell
        prob_survive: a float that represents the probability that an infected person survives
            and becomes immune

    This function spreads the imposed condition to the cells around it, given a grid, 
    probability of spread, and initial starting point(s). From each starting point given,
    it will investigate the cells around it and randomly decide if the condition spreads to 
    those cells, based on the probability given. As the condition spreads with each iteration, 
    the cells that have already experienced the condition will be set to either value 1 or 0, as
    determined by prob_spread.   
    '''
    ny, nx = population.shape

    old_pop = population.copy()
    new_pop = population.copy()

    for i in range(ny):
        for j in range(nx):
            # check for healthy people and spread the disease
            if old_pop[i, j] == 3:  
                # this checks if the neighbors are sick, and if conditions are valid               
                if (i>0) and (old_pop[i-1,j] == 2) and (random.random() < prob_spread):
                    new_pop[i-1,j] = 3 # sets that point we just checked to sick
                if (i<ny-1) and (old_pop[i+1,j] == 2) and (random.random() < prob_spread):
                    new_pop[i+1,j] = 3
                if (j>0) and (old_pop[i,j-1] == 2) and (random.random() < prob_spread):
                    new_pop[i,j-1] = 3
                if (j<nx-1) and (old_pop[i,j+1] == 2) and (random.random() < prob_spread):
                    new_pop[i,j+1] = 3
                if random.random() < prob_survive:
                    new_pop[i, j] = 1
                else:
                    new_pop[i, j] = 0

    return new_pop



# Below is where I created all my figures for testing and demonstration

# Looking at the 3x3 grid from Task 1
forest_cmap = ListedColormap(['tan', 'darkgreen', 'firebrick'])
#fig, ax = plt.subplots(1,1)
forest2 = forestfire(forest, prob_spread)
forest3 = forestfire(forest2, prob_spread)
forest4 = forestfire(forest3, prob_spread)
#ax.pcolor(forest4, cmap=forest_cmap, vmin=1, vmax=3)
#plt.show()

# Looking at the 3x5 grid from Task 1
forest_cmap = ListedColormap(['tan', 'darkgreen', 'firebrick'])
#fig, ax = plt.subplots(1,1)
bigger_forest2 = forestfire(bigger_forest, prob_spread)
bigger_forest3 = forestfire(bigger_forest2, prob_spread)
bigger_forest4 = forestfire(bigger_forest3, prob_spread)
bigger_forest5 = forestfire(bigger_forest4, prob_spread)
#ax.pcolor(bigger_forest5, cmap=forest_cmap, vmin=1, vmax=3)
#plt.show()

# Looking at assign_random
forest_cmap = ListedColormap(['tan', 'darkgreen', 'firebrick'])
#fig, ax = plt.subplots(1,1)
#ax.pcolor(test, cmap=forest_cmap, vmin=1, vmax=3)
#plt.show()

# Looking at controlled fires
controlled = np.zeros((5,5)) + 2
controlled[1:4, 1:4] = 1
controlled[2, 2] = 2
forest_cmap = ListedColormap(['tan', 'darkgreen', 'firebrick'])
#fig, ax = plt.subplots(1,1)
#ax.pcolor(controlled, cmap=forest_cmap, vmin=1, vmax=3)
#plt.show()


# Looking at disease spread
# high mortality rate, high chance of spread, no initial immunizations 
pop = np.zeros((5,5)) + 2
prob_spread = 1.0
prob_survive = 0.25
prob_sick = 0.50
prob_immune = 0.0
pop_init = assign_random_people(pop, prob_immune, prob_sick)
sick_pop = disease_spread(pop_init, prob_spread, prob_survive)
disease_cmap = ListedColormap(['gray', 'gold', 'dodgerblue', 'sienna'])
#fig, ax = plt.subplots(1,1)
#ax.pcolor(sick_pop, cmap=disease_cmap, vmin=0, vmax=3)
#plt.show()
# low mortality rate, high chance of spread, with initial immunizations 
pop = np.zeros((5,5)) + 2
prob_spread = 1.0
prob_survive = 0.90
prob_sick = 0.50
prob_immune = 0.80
pop_init = assign_random_people(pop, prob_immune, prob_sick)
sick_pop = disease_spread(pop_init, prob_spread, prob_survive)
disease_cmap = ListedColormap(['gray', 'gold', 'dodgerblue', 'sienna'])
#fig, ax = plt.subplots(1,1)
#ax.pcolor(pop_init, cmap=disease_cmap, vmin=0, vmax=3)
#plt.show()