import tensorflow
import deap
from tensorflow.examples.tutorials.mnist import input_data
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import numpy as np
import random
import math


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#import array [784, 55000], each 784 array is a single piece of 
#training data (there are 55,000) pieces of training data
matrixOfTestData = mnist.train.images
# 10 X 55000 array of desired outputs, index of label indicated the numerical value of the images
outputcheckarray=mnist.train.labels
# 5000 validation data
# 44000 tests, 11000 validation data


# Read your ANN structure from "config.py":
num_inputs = 784
num_hidden_nodes = 20
num_outputs = 10

creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
# Prepare your individuals below.
# Let's assume that you have a one-hidden layer neural network with 2 hidden nodes:
# You would need to define a list of floating numbers of size: 16 (10+6)
toolbox.register("attr_real", np.random.uniform, 0, 1)
# create hidden weigts
num_weigts = ((num_inputs+1) * num_hidden_nodes) + (num_outputs*(num_hidden_nodes+1))
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_real, num_weigts)
toolbox.register("population", tools.initRepeat, list, toolbox.individual, 100)

# Fitness Evaluation:
# 
count = 0
def evalANN(individual):
    # indivual here is a list of weight of ANN
    actualOutputArray = ANN(num_inputs, num_hidden_nodes, num_outputs, individual).evaluate(matrixOfTestData[count])
    expectedOutputArray = outputCheckArray[count]
    sumError = 0
    for i in range(0, len(outputarray)):
    	sumError = sumError + math.pow((actualOutputArray[i] - expectedOutputArray[i]), 2)
    return sumError, 
    #return 0,
    # comma at the end is necessary since DEAP stores fitness values as a tuple

toolbox.register("evaluate", evalANN)

# Define your selection, crossover and mutation operators below:
# selection
toolbox.register("select", tools.selTournament, tournsize=3)
# crossover
toolbox.register("mate", tools.cxTwoPoint)	#TODO check if the method is right
# mutation
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=.1, indpb=.5)
toolbox.register("map", map)


# Define EA parameters: n_gen, pop_size, prob_xover, prob_mut:
# You can define them in the "config.py" file too.
# ...

# pop = toolbox.population(?)
pop = toolbox.population()

# Create initial population (each individual represents an agent or ANN):
for ind in pop:
    # ind (individual) corresponds to the list of weights
    # ANN class is initialized with ANN parameters and the list of weights
    # TODO check
    ann = ANN(4, 2, 2, ind)
    my_game.add_agent(ann)

# Let's evaluate the fitness of each individual.
# First, simulation should be run!
my_game.game_loop(False) # Set it to "False" for headless mode;
#recommended for training, otherwise learning process will be very slow!
    
# Let's collect the fitness values from the simulation using
fitnesses = list(map(toolbox.evaluate, pop))
for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit

for g in range(1, config.game['n_gen']):
    my_game.generation += 1
    my_game.reset()
        
    # Start creating the children (or offspring)

    # First, Apply selection:
    offspring = toolbox.select(pop, k=len(pop))
    # Apply variations (xover and mutation), Ex: algorithms.varAnd(?, ?, ?, ?)
    offspring = algorithms.varAnd(offspring, toolbox, config.game['prob_xover'], config.game['prob_mut'])
    pop[:] = offspring
    # Repeat the process of fitness evaluation below. You need to put the recently
    # created offspring-ANN's into the game (Line 55-69) and extract their fitness values:
    for ind in offspring:
    	# TODO check
    	ann = ANN(4, 2, 2, ind)
    	my_game.add_agent(ann)
    my_game.game_loop(False)
    fits = toolbox.map(toolbox.evaluate, offspring)
    for ind, fit in zip(pop, fits):
	ind.fitness.values = fit
    # One way of implementing elitism is to combine parents and children to give them equal chance to compete:
    # For example: pop[:] = pop + offspring
    # Otherwise you can select the parents of the generation from the offspring population only: pop[:] = offspring

    # This is the end of the "for" loop (end of generations!)
	
data_gen.generate_cvs(my_game.data)
raw_input("Training is over!")
while True:
    my_game.game_loop(True)

pygame.quit()




