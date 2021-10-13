# from matplotlib.lines import _LineStyle
import numpy as np
import matplotlib.pyplot as plt
import sys, os
import regex as re

# from numpy.core.fromnumeric import mean, std
sys.path.insert(0, 'evoman') 

def plot_fitness(general_names, N_runs):
    """
    Plotting the fitness for all experiment names starting with general_names 
    (taking the sigma also), getting the saved mean fitnesses and plotting the 
    results. N_runs and gens are needed to specify respectively the amount of 
    winners and the size of the fitness array saved in the numpy file.
    """

    local_dir = os.path.dirname(__file__)
    
    # saving the enemy list and experiment names for specified general name
    directories = [name for name in os.listdir("experiments/") if os.path.isdir(f"experiments/{name}")]
    enemies = []
    experiment_names = []

    for dir in directories:
        for general_name in general_names:
            if re.match(general_name+r"\d{2,4}$", dir):
                enemies.append(int(re.findall(r"enemy\d{2,4}$", f"experiments/{dir}")[0][5:]))
                experiment_names.append(dir)
    print(experiment_names)
    print(enemies)
    

    # enemies = [4578]
    # one big array to save the loaded fitness numpy files, shape: (exp.names, runs, mean/max, generations)
    array_shape = np.load(f"experiments/{experiment_names[0]}/fitness_gens_0.npy")
    # print(len(array_shape))
    fitnesses = np.zeros((len(experiment_names), N_runs, 2, len(array_shape)))
    gains = np.zeros((len(experiment_names), N_runs, 2, len(array_shape)))

    # loading the numpy files and saving in right indices    
    for exp_id, experiment_name in enumerate(experiment_names):       
        # plt.figure()
        # plt.title(f"fitness {experiment_name}")
        for i in range(N_runs):
            f_mean = np.load(f"experiments/{experiment_name}/fitness_gens_{i}.npy")
            f_max = np.load(f"experiments/{experiment_name}/fitness_max_{i}.npy")
            fitnesses[exp_id, i, :, :] = np.array((f_mean, f_max))
            # lines = []
            # lines.append(plt.plot(f_mean, '-')[0])
            # lines.append(plt.plot(f_max, '--')[0])

    # plt.xlabel('generations')
    # plt.ylabel('fitness')
    # plt.legend(lines, ['mean', 'max'])
    
    # for exp_id, experiment_name in enumerate(experiment_names):       
    #     plt.figure()
    #     plt.title(f"gain {experiment_name}")
    #     for i in range(N_runs):
    #         g_mean = np.load(f"experiments/{experiment_name}/gain_gens_{i}.npy")
    #         g_max = np.load(f"experiments/{experiment_name}/gain_max_{i}.npy")
    #         gains[exp_id, i, :, :len(g_mean)] = np.array((g_mean, g_max))
    #         lines = []
    #         lines.append(plt.plot(g_mean, '-')[0])
    #         lines.append(plt.plot(g_max, '--')[0])

    # plt.xlabel('generations')
    # plt.ylabel('gain')
    # plt.legend(lines, ['mean', 'max'])
    
    # create a figure for every enemy and plot the non sigma and sigma fitnesses
    for i, experiment_name in enumerate(experiment_names):
        plt.figure(enemies[i])
        
        fitnesses[fitnesses==0] = np.nan
        mean_mean_fitness = np.nanmean(fitnesses[i,:,0,:], axis=0)
        stdev_mean_fitness = np.nanstd(fitnesses[i,:,0,:], axis=0)
        mean_mean_fitness = mean_mean_fitness[mean_mean_fitness != 0]
        max_fitness = np.max(fitnesses[i,:,1,:], axis=0)
        stdev_max_fitness = np.std(fitnesses[i,:,1,:], axis=0)

        # second half of indices are sigma scaled
        if experiment_name[:4] == "neat":
            color = 'blue'
            label = 'neat'
        else:
            color = 'red'
            label = 'crossover'
        plt.plot(mean_mean_fitness, '-', color=color, label=label)
        plt.plot(max_fitness, '--', color=color)
        plt.fill_between(np.arange(0, len(mean_mean_fitness)), mean_mean_fitness - stdev_mean_fitness, mean_mean_fitness + stdev_mean_fitness,
                        color=color, alpha=0.2)
        plt.fill_between(np.arange(0, len(max_fitness)), max_fitness - stdev_max_fitness, max_fitness + stdev_max_fitness,
                        color=color, alpha=0.2, hatch='/')
        # plt.title(f"normal fitness vs. sigma - enemy {experiment_name[-1]}")
        plt.xlabel("generations")
        plt.ylabel("fitness")
        plt.legend(loc=4)
        # plt.ylim(-9, 80)
        # plt.savefig(f"meanfigs/neat_mean_enemy{enemies[i]}", dpi=200)
        
    plt.show()


if __name__ == '__main__':
    # experiment names specified
    experiment_names = ['crossover_enemy', 'neat_enemy']
    N_runs = 10

    plot_fitness(experiment_names, N_runs)
