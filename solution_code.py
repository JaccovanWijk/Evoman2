import pickle
import os
import regex as re

directories = [name for name in os.listdir("experiments/") if os.path.isdir(f"experiments/{name}")]
enemies_names = []
experiment_names = []
for dir in directories:
    if re.match(r"crossover_enemy\d{3,4}$", dir):
        experiment_names.append(dir)
    # if re.match(r"neat_enemy\d{3,4}$", dir):
    #     experiment_names.append(dir)

# experiment_name = "crossover_enemy265"
for experiment_name in experiment_names:
    for i in range(10):
        # unpickle saved winner
        with open(f"experiments/{experiment_name}/winner_{i}.pkl", "rb") as f:
            genome = pickle.load(f)
    
        if os.path.exists(f"solutions/{experiment_name}_winner_{i}_sol.txt"):
            os.remove(f"solutions/{experiment_name}_winner_{i}_sol.txt")
    
        with open(f"solutions/{experiment_name}_winner_{i}_sol.txt", "a") as f:
            for i, x in enumerate(genome):
                f.write(str(x)+"\n")

