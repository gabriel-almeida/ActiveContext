import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import ttest_ind


def plot(results, base_algorithm='Cramer Deviation', ttest_p=0.05):
    possible_contexts = list(results.keys())
    possible_contexts.sort()
    n_possible_contexts = len(possible_contexts)

    algo_names = list(results[possible_contexts[0]].keys())
    algo_names.sort()
    n_algorithms = len(algo_names)

    avg = [np.mean(results[i][j]) for i in results for j in results[i]]
    std = np.std(avg)
    y_min = np.min(avg) - std  # calculate the min average
    x_individual_step = 1
    x_group_step = 2
    color_sequence = 'krbcymw'

    y_axis = []
    x_axis = []
    y_ttest = []
    x_ttest = []

    current_x = 0
    centered_x_label = [(x_individual_step*n_algorithms - 1)/2.0 + (i-1)*(x_group_step + x_individual_step*n_algorithms) for i in range(1,n_possible_contexts+1)]  # i have no idea what im doing

    for n_context in possible_contexts:
        results_by_algorithm = results[n_context]
        for algorithm in algo_names:
            current_y = np.average(results_by_algorithm[algorithm]) - y_min
            y_axis += [current_y]
            x_axis += [current_x]

            # if algorithm != base_algorithm:
            base = results_by_algorithm[base_algorithm]
            current = results_by_algorithm[algorithm]
            t, prob = ttest_ind(base, current)
            if prob < 1 - ttest_p:
                y_ttest += [current_y + y_min + std*0.1]
                x_ttest += [current_x]

            current_x += x_individual_step
        current_x += x_group_step

    plt.bar(x_axis, y_axis, bottom=y_min, align='center', color=color_sequence[:n_algorithms])
    plt.plot(x_ttest, y_ttest, 'k*')

    legend_meaning = [mpatches.Patch(color=color_sequence[i % len(color_sequence)], label=algo_names[i])
                      for i in range(n_algorithms)]
    plt.legend(handles=legend_meaning, loc=0, fontsize=9)

    plt.title('Results')
    plt.xticks(centered_x_label, possible_contexts)
    plt.xlabel('Number of contexts')
    plt.ylabel('MAE')
    plt.show()

if __name__ == "__main__":
    import json
    results = json.loads(open('results-cramer-tilde.json').readline())
    plot(results)