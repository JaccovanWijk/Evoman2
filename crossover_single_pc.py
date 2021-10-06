#from __future__ import print_function
import sys, os

import random
from numpy.lib.shape_base import _expand_dims_dispatcher
sys.path.insert(0, 'evoman') 
import neat       
import pickle   
import numpy as np
from evoman.environment import Environment
from demo_controller import player_controller
from plot import plot_fitness
from genomes import genomes

# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"
            
pop_size = 100
gen = 50
n_hidden = 5
N_runs = 10
enemies = [4, 5, 8]
keep_old = 0.2 # TODO: GEBRUIKEN?
mutation = 0.2 # TODO: DEZE AANPASSEN?

experiment_name = f"crossover_sigmax_enemy{enemies}"
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

env = Environment(experiment_name=experiment_name,
                  playermode="ai",
                  player_controller=player_controller(n_hidden),
                  enemies=enemies,
                  randomini="yes", 
                  multiplemode="yes")

def fitness(population, i):
    pop_fitness = []
    for individual in population:
        fitness = env.play(pcont=individual)[0]
        pop_fitness.append(fitness)
        print(f"\nFitness for all enemies = {fitness}")
    
    fitness_gens.append(np.mean(pop_fitness))       # adding mean fitness to list
    np.save(f"{experiment_name}/fitness_gens_{i}", fitness_gens)   # saving to numpy file, opening in test.py
    fitness_max.append(np.max(pop_fitness))         # adding max fitness to list
    np.save(f"{experiment_name}/fitness_max_{i}", fitness_max)     # saving to numpy file, opening in test.py
    # TODO: Save mean sigma 
    sigma_mean.append(np.mean([x[-1] for x in population]))
    np.save(f"{experiment_name}/sigma_gens_{i}", sigma_mean)
     
    return pop_fitness

def crossover(population, pop_fitness):
    new_population = np.zeros((len(pop),len(pop[0])))
    
    # Get top 20%
    # TODO: DO WE WANT TO USE THIS? Do we want to let only top parents breed or do we want to keep the top parents in the population
    # top_parents = int(len(population) * keep_old)
    # top_index = sorted(range(len(pop_fitness)), key=lambda i: pop_fitness[i])[-top_parents:]
    # top_pop = [population[x] for x in top_index]
    # top_fitness = [pop_fitness[x] for x in top_index]
    
    # get weights according to relative fitness
    if (min(pop_fitness) < 0):
        positive = [x + min(pop_fitness) for x in pop_fitness]
        pop_weights = [x/sum(positive) for x in positive]
    else:
        pop_weights = [x/sum(pop_fitness) for x in pop_fitness]
        
    
    # Make 100% new population with uniform crossover
    for c in range(len(population)): #new_children):        
        # Choose random parents with weights in mind
        # TODO: COULD ALSO CHOOSE TO TAKE MEAN OF TWO PARENTS, OR RANDOM VALUE p BETWEEN 0,1 AND GET p FROM ONE PARENT AND (1 - p) FROM THE OTHER
        parents = random.choices(population, weights=pop_weights, k=2)
        
        # Pick every gene of the parents randomly
        parent_length = len(parents[0])
        child = np.zeros(parent_length)
        for j in range(parent_length):
            gene = random.choice([parents[0][j], parents[1][j]])
            child[j] = gene
        
        # Mutate sigma and look if the genes of the children need to be mutated
        child[-1] *= np.e ** np.random.normal(0, 1) # TODO: Look into TAU, set as 1 now
        if child[-1] < 0.2: # TODO: WHAT THRESHOLD TO USE??
            child[-1] = 0.2 
        for j in range(parent_length - 1):
            if random.random() < mutation:
                # TODO: Make sigma part of genes, either one sigme per child (size is n + 1) or one sigma per gene (size is n * 2)
                child[j] = gaussian(child[j], child[-1])
            
        new_population[c] = child

    return new_population

def gaussian(value, sigma):
    value += sigma * np.random.normal(0, 1) # Same as np.random.normal(0,sigma)
    if value < -1:
        return -1
    elif value > 1:
        return 1
    else:
        return value

# number of weights for multilayer with hidden neurons
n_vars = (env.get_num_sensors()+1)*n_hidden + (n_hidden+1)*5

for i in range(N_runs):
    if not os.path.exists(f"{experiment_name}/winner_{i}.pkl"):
        fitness_gens = []
        fitness_max = []
        sigma_mean = []
    
        # create initial population and add to environment
        #all_genomes = genomes(pop_size, n_vars)
        # pop = all_genomes.get_population()
        sigmaless_pop = np.random.uniform(-1, 1, (pop_size, n_vars)) 
        
        # Add sigma parameter control
        # TODO: BETWEEN 0 and 3?
        pop = np.zeros((pop_size, n_vars + 1))
        for i in range(pop_size):
            pop[i] = np.append(sigmaless_pop[i], random.uniform(0, 3))
        # sigmas = np.zeros(pop_size)
        # for i in range(pop_size):
        #     sigmas[i] = random.uniform(0, 3)
            
        pop_fitness = fitness(sigmaless_pop, i)
        
        best_each_gen = [np.max(pop_fitness)]
        best = pop[np.argmax(pop_fitness)]
        mean_each_gen = [np.mean(pop_fitness)]
        std_each_gen = [np.std(pop_fitness)]
        sigma_each_gen = [np.mean([x[-1] for x in pop])]
        
        print("\n------------------------------------------------------------------")
        print(f"Generation 0. Mean {mean_each_gen[-1]}, best {best_each_gen[-1]}")
        print("------------------------------------------------------------------")
        
        # TODO: Moet dit eigenlijk?
        solutions = [sigmaless_pop, pop_fitness]
        env.update_solutions(solutions)
        
        for g in range(gen - 1):
            pop = crossover(pop, pop_fitness)#, keep_old)
            sigmaless_pop = [x[:-1] for x in pop]
            pop_fitness = fitness(sigmaless_pop, i)
            
            new_best = np.max(pop_fitness)
            # TODO: REEVALUATE THE OLD BEST ?
            if new_best > best_each_gen[-1]:
                best = pop[np.argmax(pop_fitness)]
            best_each_gen.append(new_best)
            
            mean_each_gen.append(np.mean(pop_fitness))
            std_each_gen.append(np.std(pop_fitness))
            sigma_each_gen.append(np.mean([x[-1] for x in pop]))
             
            print("\n------------------------------------------------------------------")
            print(f"Generation {g+1}. Mean {mean_each_gen[-1]}, best {best_each_gen[-1]}")
            print("------------------------------------------------------------------")
                
            # TODO: Moet dit eigenlijk?
            solutions = [sigmaless_pop, pop_fitness]
            env.update_solutions(solutions)
        print(best)
        # Stackoverflow on how to save the winning file and open it: https://stackoverflow.com/questions/61365668/applying-saved-neat-python-genome-to-test-environment-after-training
        with open(f"{experiment_name}/winner_{i}.pkl", "wb") as f:
            pickle.dump(best, f)
            f.close()