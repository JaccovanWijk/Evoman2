#from __future__ import print_function
import sys, os

sys.path.insert(0, 'evoman') 
import neat       
import pickle   
import numpy as np
from time import time
from evoman.environment import Environment
from player_controllers import player_controller
from plot import plot_fitness
 

def fitness_player(genomes, config):
    """
    Partly from https://neat-python.readthedocs.io/en/latest/xor_example.html
    
    Looks at every genome of a generation and executes env.play to get fitness.
    Saves mean and max of each generation to npy file.
    """
    
    # get fitness each genome
    f_g = []
    for genome_id, g in genomes:
        fi, player_life, enemy_life, playtime = env.play(pcont=g)
        # g.fitness = fi
        g.fitness = player_life - enemy_life
        f_g.append(fi)
        # print(f"\nrun {i}, fitness: {np.round(fi, 5)}, playerlife: {np.round(player_life, 3)}, enemylife: {np.round(enemy_life, 3)}, time: {np.round(playtime,1)} s\n")
        print(f"\nrun {i}, fitness: {np.round(fi, 5)}, gain: {np.round(g.fitness, 3)}, playerlife: {np.round(player_life, 3)}, enemylife: {np.round(enemy_life, 3)}, time: {np.round(playtime,1)} s\n")

    # # save mean and max each generation
    fitness_gens.append(np.mean(f_g))     
    np.save(f"experiments/{experiment_name}/fitness_gens_{i}", fitness_gens)
    fitness_max.append(np.max(f_g))       
    np.save(f"experiments/{experiment_name}/fitness_max_{i}", fitness_max)   

def fitness_sigma(genomes, config):
    """
    Partly from https://neat-python.readthedocs.io/en/latest/xor_example.html
    
    Looks at every genome of a generation and executes env.play to get the
    unscaled fitness. Scales these by applying sigma scaling. Saves the 
    unscaled mean and max fitnesses each generation.
    """
    
    # get unscaled fitness each genome
    unscaled = []
    f_g = []
    for genome_id, g in genomes:
        fi, player_life, enemy_life, playtime = env.play(pcont=g)
        unscaled.append(fi)
        # unscaled.append(env.play(pcont=g)[0])
        print(f"\nrun {i}, fitness: {np.round(fi, 5)}, playerlife: {np.round(player_life, 3)}, enemylife: {np.round(enemy_life, 3)}, time: {np.round(playtime,1)} s\n")

    j = 0
    for genome_id, g in genomes:
        # get mean and standard deviation
        mean = np.mean(unscaled)
        std = np.std(unscaled)
        
        # scale fitness of the genome with sigma scaling
        g.fitness = max(0, unscaled[j] - (mean - 2 * std))
        f_g.append(g.fitness)
        j += 1
    
    # # save mean and max each generation
    fitness_gens.append(np.mean(unscaled))       
    np.save(f"experiments/{experiment_name}/fitness_gens_{i}", fitness_gens)  
    fitness_max.append(np.max(unscaled))      
    np.save(f"experiments/{experiment_name}/fitness_max_{i}", fitness_max)   


def run():
    """
    Mostly from https://neat-python.readthedocs.io/en/latest/xor_example.html
    
    Create a population, start a run with or without sigma scaled fitness,
    print and save the result.
    """
    # create the population, which is the top-level object for a NEAT run
    p = neat.Population(config)

    # add a stdout reporter to show progress in the terminal
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    #p.add_reporter(neat.Checkpointer(1000))

    if sigma_scaling:
        # run for number of generations
        winner = p.run(fitness_sigma, generations)
    else:
        # run for number of generations
        winner = p.run(fitness_player, generations)
    
    # display the winning genome
    print('\nBest genome:\n{!s}'.format(winner))

    # saving winner as pickle file (copied from https://stackoverflow.com/questions/61365668/applying-saved-neat-python-genome-to-test-environment-after-training)
    with open(f"experiments/{experiment_name}/winner_{i}.pkl", "wb") as f:
        pickle.dump(winner, f)
        f.close()

# TODO: SLDFJSDLFNSLF Weg gooien 
# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"


if __name__ == '__main__':
    time0 = time()
    # set parameters
    N_runs = 10     
    generations = 50         
    enemies = [4,5,7,8]             
    sigma_scaling = False        

    # set the right directory path name
    if sigma_scaling:
        experiment_name = f"neat_sigma_gain_enemy{enemies[0]}{enemies[1]}{enemies[2]}{enemies[3]}"
    else:
        experiment_name = f"neat_gain_enemy{enemies[0]}{enemies[1]}{enemies[2]}{enemies[3]}"
    
    # create directory if it does not exist yet
    if not os.path.exists(f"experiments/{experiment_name}"):
        os.makedirs(f"experiments/{experiment_name}")
    
    # find config file and create a neat config
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'neat_config_file.txt')
    config = neat.Config(neat.DefaultGenome,neat.DefaultReproduction,
                          neat.DefaultSpeciesSet, neat.DefaultStagnation,
                          config_path)
    
    # create an Evoman environment
    env = Environment(experiment_name=experiment_name,
                      playermode="ai",
                      player_controller=player_controller(config),
                      enemies=enemies,
                      randomini="yes", 
                      multiplemode="yes",
                      logs="off")

    for i in range(N_runs):
        print(f"\n----------------------\nWelcome to run {i}\n----------------------\n")
        # continue after the last completed run
        if not os.path.exists(f"experiments/{experiment_name}/winner_{i}.pkl"):
            if os.path.exists(f"experiments/{experiment_name}/fitness_gens_{i}.npy"):
                print("fitness.npys exists, but no winner, deleting npys")
                os.remove(f"experiments/{experiment_name}/fitness_gens_{i}.npy")
                os.remove(f"experiments/{experiment_name}/fitness_max_{i}.npy")

            # global variables for saving mean and max each generation
            fitness_gens = []       
            fitness_max = []    
            
            run()
    time1 = time()
    print(f"#########\nTook {time1-time0} seconds.\n#########")
    # plot results
    plot_fitness(experiment_name, N_runs)