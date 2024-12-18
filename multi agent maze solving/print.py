import matplotlib.pyplot as plt
import numpy as np

# Data
sizes = [11, 51, 101, 201, 401, 801, 1601]

times = {
    'DFS': [0.000, 0.002, 0.014, 0.061, 0.318, 4.147, 9.520],
    'BFS': [0.000, 0.008, 0.032, 0.148, 0.673, 3.848, 51.548],
    'A*':  [0.000, 0.007, 0.017, 0.091, 0.069, 1.450, 5.543]
}

path_lengths = {
    'DFS': [17, 131, 505, 1873, 5305, 20691, 31381],
    'BFS': [17, 109, 205, 429, 803, 1673, 3337],
    'A*':  [17, 109, 205, 429, 803, 1673, 3337]
}

plt.figure(figsize=(15, 6))

# Time comparison plot
plt.subplot(1, 2, 1)
for algorithm in times:
    plt.plot(sizes, times[algorithm], marker='o', label=algorithm, linewidth=2)
plt.title('Algorithm Solving Times', fontsize=14)
plt.xlabel('Maze Size (NxN)', fontsize=12)
plt.ylabel('Time (seconds)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.xscale('log')
plt.yscale('log')

# Add grid lines
plt.grid(True, which="both", ls="-", alpha=0.2)

# Path length comparison plot
plt.subplot(1, 2, 2)
for algorithm in path_lengths:
    plt.plot(sizes, path_lengths[algorithm], marker='o', label=algorithm, linewidth=2)
plt.title('Path Lengths', fontsize=14)
plt.xlabel('Maze Size (NxN)', fontsize=12)
plt.ylabel('Number of Steps', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.xscale('log')
plt.yscale('log')

# Add grid lines
plt.grid(True, which="both", ls="-", alpha=0.2)

plt.tight_layout()
plt.savefig('algorithm_comparison_results.png', dpi=300, bbox_inches='tight')
plt.close()

print("Plots have been saved as 'algorithm_comparison_results.png'")