import math
import random
import matplotlib.pyplot as plt
from copy import deepcopy


def generation(n):
    gen = []
    while len(gen) != n:
        node = (random.randint(0, 200), random.randint(0, 200))
        if node not in gen:
            gen.append(node)
    return gen


def plot_start(path):
    x, y = zip(*path)
    plt.ion()
    plt.subplots(1, 2, figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(x, y)
    plt.plot(x, y, color='red')

    plt.subplot(1, 2, 2)
    plt.plot(x, y, color='red')

    plt.show()


def plot_update(path, data):
    plt.clf()

    x, y = zip(*path)
    plt.subplot(1, 2, 1)
    plt.scatter(x, y)
    plt.plot(x, y, color='red')

    x, y = zip(*data)
    plt.subplot(1, 2, 2)
    plt.plot(x, y, color='red')

    plt.pause(0.005)
    plt.draw()


def distance(path):
    dist = 0
    for j in range(1, len(path)):
        dist = dist + math.sqrt((path[j][0] - path[j - 1][0]) ** 2 + (path[j][1] - path[j - 1][1]) ** 2)
    return dist


def neighborhood(path):
    neighbors = []
    for i in range(1, len(path) - 1):
        for j in range(i + 1, len(path) - 1):
            inversion = deepcopy(path)
            inversion[i], inversion[j] = inversion[j], inversion[i]
            neighbors.append(inversion)
    return neighbors


def random_neighbor(path):
    new_path = deepcopy(path)
    x, y = 0, 0
    while x == y:
        x = random.randint(1, len(path)-2)
        y = random.randint(1, len(path)-2)

    new_path[x], new_path[y] = new_path[y], new_path[x]
    return new_path


def tabu(path, tabu_size, max_threshold):
    data = []
    tabu_list = [path]
    best_solution, best_candidate = path, path
    plot_start(path)

    threshold = 0
    while threshold < max_threshold:
        neighbors = neighborhood(best_candidate)
        best_candidate = neighbors[0]

        for neighbor in neighbors:
            if (neighbor not in tabu_list) and (distance(neighbor) < distance(best_candidate)):
                best_candidate = neighbor

        if distance(best_candidate) < distance(best_solution):
            best_solution = best_candidate
            threshold = 0
        else:
            threshold += 1
        tabu_list.append(best_candidate)
        if len(tabu_list) > tabu_size:
            tabu_list.pop(0)

        data.append((len(data) + 1, distance(best_solution)))
        plot_update(best_solution, data)
        print(f"Iteration: {len(data)} Distance: {distance(best_solution)} Tabu list occupation: {len(tabu_list)} Threshold: {threshold}")

    plt.pause(10)


def simulated_annealing(path, max_treshold, max_one_temp):
    threshold = 0
    temperature = 5
    data = []
    best_solution = path

    plot_start(path)
    while threshold < max_treshold:
        candidate = random_neighbor(best_solution)
        delta = distance(best_solution) - distance(candidate)

        for i in range(max_one_temp):
            if delta > 0:
                best_solution = candidate
                threshold = 0
            else:
                prob = math.exp(delta/temperature)
                if random.random() < prob:
                    best_solution = candidate
                    threshold = 0

        if len(data) > 0:
            if data[-1][1] == distance(best_solution):
                threshold += 1

        temperature *= 0.995
        data.append((len(data) + 1, distance(best_solution)))
        plot_update(best_solution, data)
        print(f"Iteration: {len(data)} Distance: {distance(best_solution)} Temperature: {temperature} Threshold: {threshold}")


start = generation(40)
# start = [(60, 200), (180, 200), (100, 180), (140, 180), (20, 160), (80, 160), (200, 160), (140, 140), (40, 120), (120, 120), (180, 100), (60, 80), (100, 80), (180, 60), (20, 40), (100, 40), (200, 40), (20, 20), (60, 20), (160, 20)]
tabu(start, 100, 100)
simulated_annealing(start, 100, 10000)
