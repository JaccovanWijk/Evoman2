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
n_hidden = 10 # TODO: DIT MOET 10 VAN DE OPDRACHT
N_runs = 10
enemies = [4, 5, 8]
keep_old = 0.2 # TODO: GEBRUIKEN?
mutation = 0.2 # TODO: DEZE AANPASSEN?

experiment_name = f"crossover_sigma1_enemy{enemies[0]}{enemies[1]}{enemies[2]}"
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
        print(f"--Fitness for all enemies = {fitness}--")
    
    fitness_gens.append(np.mean(pop_fitness))       # adding mean fitness to list
    np.save(f"experiments/{experiment_name}/fitness_gens_{i}", fitness_gens)   # saving to numpy file, opening in test.py
    fitness_max.append(np.max(pop_fitness))         # adding max fitness to list
    np.save(f"experiments/{experiment_name}/fitness_max_{i}", fitness_max)     # saving to numpy file, opening in test.py
     
    return pop_fitness

def crossover(solutions): #, old):
    population, pop_fitness = solutions
    new_population = np.zeros((len(pop),len(pop[0])))
    
    # Get top 20%
    # TODO: DO WE WANT TO USE THIS? Do we want to let only top parents breed or do we want to keep the top parents in the population
    top_parents = int(len(population) * keep_old)
    top_index = sorted(range(len(pop_fitness)), key=lambda i: pop_fitness[i])[-top_parents:]
    top_pop = [population[x] for x in top_index]
    top_fitness = [pop_fitness[x] for x in top_index]
    
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
            
        for j in range(parent_length):
            if random.random() < mutation:
                sigma = 1
                child[j] = gaussian(child[j], sigma)
            
        new_population[c] = child

    return new_population

def gaussian(value, sigma):
    value += np.random.normal(0, sigma)
    if value < -1:
        return -1
    elif value > 1:
        return 1
    else:
        return value

# number of weights for multilayer with hidden neurons
n_vars = (env.get_num_sensors()+1)*n_hidden + (n_hidden+1)*5

for i in range(N_runs):
    if not os.path.exists(f"experiments/{experiment_name}/winner_{i}.pkl"):
        if os.path.exists(f"experiments/{experiment_name}/fitness_gens{i}.pkl"):
            os.remove(f"experiments/{experiment_name}/fitness_gens{i}.pkl")
            os.remove(f"experiments/{experiment_name}/fitness_max{i}.pkl")
            
        fitness_gens = []
        fitness_max = []
    
        # create initial population and add to environment
        #all_genomes = genomes(pop_size, n_vars)
        # pop = all_genomes.get_population()
        pop = np.random.uniform(-1, 1, (pop_size, n_vars)) 
        pop_fitness = fitness(pop, i)
        
        best_each_gen = [np.max(pop_fitness)]
        best = pop[np.argmax(pop_fitness)]
        mean_each_gen = [np.mean(pop_fitness)]
        std_each_gen = [np.std(pop_fitness)]
        
        print("\n------------------------------------------------------------------")
        print(f"Run:{i}. Generation 0. Mean {mean_each_gen[-1]}, best {best_each_gen[-1]}")
        print("------------------------------------------------------------------")
        
        solutions = [pop, pop_fitness]
        env.update_solutions(solutions)
        
        for g in range(gen - 1):
            pop = crossover(solutions)#, keep_old)
            pop_fitness = fitness(pop, i)
            
            new_best = np.max(pop_fitness)
            # TODO: REEVALUATE THE OLD BEST ?
            if new_best > best_each_gen[-1]:
                best = pop[np.argmax(pop_fitness)]
            best_each_gen.append(new_best)
            
            mean_each_gen.append(np.mean(pop_fitness))
            std_each_gen.append(np.std(pop_fitness))
             
            print("\n------------------------------------------------------------------")
            print(f"Run:{i}. Generation {g+1}. Mean {mean_each_gen[-1]}, best {best_each_gen[-1]}")
            print("------------------------------------------------------------------")
            
            solutions = [pop, pop_fitness]
            env.update_solutions(solutions)
        print(best)
        # Stackoverflow on how to save the winning file and open it: https://stackoverflow.com/questions/61365668/applying-saved-neat-python-genome-to-test-environment-after-training
        with open(f"experiments/{experiment_name}/winner_{i}.pkl", "wb") as f:
            pickle.dump(best, f)
            f.close()