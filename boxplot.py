import neat
import pickle
import numpy as np
import sys, os
import matplotlib.pyplot as plt
from pygame.constants import SRCALPHA
sys.path.insert(0, 'evoman') 
from evoman.environment import Environment
from demo_controller import player_controller
import regex as re

# TODO: SLDFJSDLFNSLF Weg gooien 
# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

def replay_genome(config_path, run_i, experiment_name):
    """
    Picks a winner.pkl for run_i and experiment_name and reruns the genome to get the gain.
    """
    genome_path = f"experiments/{experiment_name}/winner_{run_i}.pkl"
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    # unpickle saved winner
    with open(genome_path, "rb") as f:
        genome = pickle.load(f)
        
    gain = 0
    for enemy in [1,2,3,4,5,6,7,8]:
        env = Environment(experiment_name=experiment_name,
                          playermode="ai",
                          player_controller=player_controller(n_hidden),
                          enemies=[enemy],
                          randomini="yes", 
                          logs="off")
        
        # playing genome and getting gain
        player_life, enemy_life = env.play(pcont=genome)[1:3]
        
        print(f"enemy {enemy}, player life {player_life}, enemy life {enemy_life}")
        
        gain += player_life - enemy_life

    return gain

def five_runs(run_i, experiment_name):
    """
    Runs replay_genome() five times and saves the fitnesses.
    """
    f_r = []
    for i in range(0,5):
        print(f"doing winner_{run_i} for {i+1}th time of {experiment_name}")
        gain = replay_genome(config_path, run_i, experiment_name)
        f_r.append(gain)
        print(f"gain: {gain}\n")
    return f_r


# used variables
N_runs = 10
n_hidden = 10
local_dir = os.path.dirname(__file__)
config_path = os.path.join (f"{local_dir}",'neat_config_file.txt')
config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
  
# reading all directories and saving all starting with experiment names with specified variables
directories = [name for name in os.listdir("experiments/") if os.path.isdir(f"experiments/{name}")]
enemies_names = []
experiment_names = []
for dir in directories:
    if re.match(r"crossover_enemy\d{4}$", dir):
        enemies_names.append(int(re.findall(r"enemy\d{4}$", f"experiments/{dir}")[0][5:]))
        experiment_names.append(dir)
    # if re.match("neat_sigma_nhidden5_gen50_enemy",f"experiments/{dir}"):    # can be crossover
    #     enemies.append(int(re.findall(r"enemy\d{3}", dir)[0][5:]))
    #     experiment_names.append(dir)

# sorting enemies (copied from https://stackoverflow.com/questions/6618515/sorting-list-based-on-values-from-another-list)
experiment_names = [x for _, x in sorted(zip(enemies_names, experiment_names))]
enemies_names.sort()

# print(str(enemies[0]))
# enemies = [int(enemy) for enemy in [str(enemieslist) for enemieslist in enemies]]
# enemies1 = []
# for i in range(len(enemies)):
#     enemies1.append([int(enemy) for enemy in str(enemies[i])])
# enemies = [int(enemy) for enemy in str(enemies[0])]
# enemies = [1,2,3,4,5,6,7,8]
print(experiment_names)

print(enemies_names)
# saving all data for the boxplots
boxplotdata = []


# saving the mean gain from every five runs for every experiment name
for i, experiment_name in enumerate(experiment_names):
    gains = []
    # env = Environment(experiment_name=experiment_name,
    #                   playermode="ai",
    #                   player_controller=player_controller(n_hidden),
    #                   enemies=enemies,
    #                   randomini="yes", 
    #                   multiplemode="yes",
    #                   logs="off")

    gains = [np.mean(five_runs(j, experiment_name)) for j in range(0, N_runs)]
    
    print(f"The best winner this run was winner_{np.argmax(gains)}")
    
    # saving the data in an array for plt.boxplot() in shape (enemy, gains)
    boxplotdata.append(gains)

# changing enemy names with sigma for xticks
# for i, enemy in enumerate(enemies):
#     if (i % 2) != 0:
#         enemies[i] = f"{enemy} Sigma"

plt.figure()
plt.boxplot(boxplotdata)
plt.xticks(enemies_names, len(enemies_names))
plt.ylabel("individual gain")
plt.xlabel('Enemy')
# plt.title('Individual Gain\nNormal vs Sigma Scaling')
plt.savefig(f"boxplotfigs/neat", dpi=400)
plt.show()
