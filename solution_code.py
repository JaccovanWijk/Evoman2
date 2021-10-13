import pickle
import os

experiment_name = "crossover_enemy78"

for i in range(10):
    # unpickle saved winner
    with open(f"experiments/{experiment_name}/winner_{i}.pkl", "rb") as f:
        genome = pickle.load(f)

    if os.path.exists(f"solutions/{experiment_name}_winner_{i}_sol.txt"):
        os.remove(f"solutions/{experiment_name}_winner_{i}_sol.txt")

    with open(f"solutions/{experiment_name}_winner_{i}_sol.txt", "a") as f:
        for i, x in enumerate(genome):
            f.write(str(x)+"\n")

