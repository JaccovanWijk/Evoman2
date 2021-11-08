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

# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    
# Set parameters        
pop_size = 100
gen = 50
n_hidden = 10
N_runs = 10
enemies = [2,3,4,5,7,8]
keep_old = 0.1
mutation = 0.4

# Define experiment name
experiment_name = "extra2_long_crossover"
experiment_name += "_enemy"
for e in enemies:
    experiment_name += f"{e}"

# Create directory for experiment
if not os.path.exists(f"experiments/{experiment_name}"):
    os.makedirs(f"experiments/{experiment_name}")
    
# Create Environment class instance
env = Environment(experiment_name=experiment_name,
          playermode="ai",
          player_controller=player_controller(n_hidden),
          enemies=enemies,
          randomini="yes", 
          multiplemode="yes",
          logs="off")

def fitness(population, i):
    """    
    Looks at every genome of a population and executes env.play to get fitness and gain.
    Saves mean and max fitness and gain of each population to npy file.
    """
    pop_fitness = []
    pop_gain = []
    for individual in population:
        fitness, p, e, t  = env.play(pcont=individual)
        pop_fitness.append(fitness)
        pop_gain.append(p-e)
        print(f"\n-- Fitness for all enemies = {fitness}, gain = {p-e} player = {p}, enemy = {e}, time = {t} --")

    fitness_gens.append(np.mean(pop_fitness)) 
    np.save(f"experiments/{experiment_name}/fitness_gens_{i}", fitness_gens)  
    fitness_max.append(np.max(pop_fitness))     
    np.save(f"experiments/{experiment_name}/fitness_max_{i}", fitness_max)   
    gain_gens.append(np.mean(pop_gain))     
    np.save(f"experiments/{experiment_name}/gain_gens_{i}", gain_gens)   
    gain_max.append(np.max(pop_gain))         
    np.save(f"experiments/{experiment_name}/gain_max_{i}", gain_max)    
    
    return pop_fitness

def crossover(solutions): 
    """    
    Looks at every genome and corrisponding fitness of a population and 
    calculates the top of the population. The new population is found 
    by an uniform crossover combined with the top of the old population.
    """
    population, pop_fitness = solutions
    
    # Get top of the population
    top_parents = int(len(population) * keep_old)
    top_index = sorted(range(len(pop_fitness)), key=lambda i: pop_fitness[i])[-top_parents:]
    top_pop = [population[x] for x in top_index]
    
    # get weights according to relative fitness
    if (min(pop_fitness) < 0):
        positive = [x + abs(min(pop_fitness)) for x in pop_fitness]
        pop_weights = [x/sum(positive) for x in positive]
    else:
        pop_weights = [x/sum(pop_fitness) for x in pop_fitness]
        
        
    new_population = np.zeros((pop_size,len(pop[0])))
    
    # Make partly new population with uniform crossover
    for c in range(pop_size - len(top_pop)):       
        # Choose random parents with weights in mind
        parents = random.choices(population, weights=pop_weights, k=2)
        
        # Pick every gene of the parents randomly
        parent_length = len(parents[0])
        child = np.zeros(parent_length)
        for j in range(parent_length):
            gene = random.choice([parents[0][j], parents[1][j]])
            child[j] = gene
        
        # Check for mutation
        for j in range(parent_length):
            if random.random() < mutation:
                sigma = 1
                child[j] = gaussian(child[j], sigma)
        new_population[c] = child
    
    # Add top to new population
    for c in [x + pop_size - len(top_pop) for x in range(len(top_pop))]:
        new_population[c] = top_pop[c - pop_size + len(top_pop)]

    return new_population

def gaussian(value, sigma):
    """    
    Scales a value between -1 and 1 with normal distribution.
    """
    value += np.random.normal(0, sigma)
    if value < -1:
        return -1
    elif value > 1:
        return 1
    else:
        return value

# number of weights for multilayer with hidden neurons
n_vars = (env.get_num_sensors()+1)*n_hidden + (n_hidden+1)*5

# Run for N_runs times
for i in range(N_runs):
    if not os.path.exists(f"experiments/{experiment_name}/winner_{i}.pkl"):
        if os.path.exists(f"experiments/{experiment_name}/fitness_gens{i}.pkl"):
            os.remove(f"experiments/{experiment_name}/fitness_gens{i}.pkl")
            os.remove(f"experiments/{experiment_name}/fitness_max{i}.pkl")
          
        fitness_gens = []
        fitness_max = []
        gain_gens = []
        gain_max = []
    
        # create initial population
        pop = np.random.uniform(-1, 1, (pop_size, n_vars)) 
        
        # TODO: HAAL WEG
        genome_path = f"experiments/winners.pkl"
        # unpickle saved winner
        with open(genome_path, "rb") as f1:
            genomes = pickle.load(f1)[0] # [0] for best gain and [1] for best slain
            f1.close()
            
        for g,name in enumerate(genomes):
            print(name)
            with open(f"experiments/{name[:-9]}/winner_{name[-1]}.pkl", "rb") as f2:
                genome = pickle.load(f2)
                f2.close()
            for extra in range(10):
                pop[g*10 + extra] = genome
            
            # pop[g] = genome
            
        # with open("current_beast.pkl", "rb") as f:
        #         genome = pickle.load(f)
                
        # for p in range(len(pop)):
        #     pop[p] = genome

        pop_fitness = fitness(pop, i)

        # Keep track of mean and max values
        best_each_gen = [np.max(pop_fitness)]
        best = pop[np.argmax(pop_fitness)]
        mean_each_gen = [np.mean(pop_fitness)]
        std_each_gen = [np.std(pop_fitness)]
        
        print("\n------------------------------------------------------------------")
        print(f"Run:{i}. Generation 0. Mean {mean_each_gen[-1]}, best {best_each_gen[-1]}")
        print("------------------------------------------------------------------")
        
        # Update env
        solutions = [pop, pop_fitness]
        env.update_solutions(solutions)
        
        for g in range(gen - 1):
            pop = crossover(solutions)
            pop_fitness = fitness(pop, i)
            
            new_best = np.max(pop_fitness)
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
       
        # Write winner to winner file
        with open(f"experiments/{experiment_name}/winner_{i}.pkl", "wb") as f3:
            pickle.dump(best, f3)
            f3.close()