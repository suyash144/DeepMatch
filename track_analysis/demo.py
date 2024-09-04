import numpy as np

def sort_neurons_by_position(positions):
    # Sort first by the y-axis values, then by the z-axis values within each y-group
    sorted_indices = np.lexsort((positions[:, 1], positions[:, 0]))
    return sorted_indices

if __name__ == '__main__':
    positions = np.array([[400, 100], [432, 100], [400, 115], [432, 115], [200,100], [232, 100], [200,85], [232, 85]])
    sorted_indices = sort_neurons_by_position(positions)
    print(sorted_indices)