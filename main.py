import math
import time
import random
import pandas
import matplotlib.pyplot as plt
from copy import deepcopy


def generation(n):
    gen = []
    while len(gen) != n:
        node = (random.randint(0, 200), random.randint(0, 200))
        if node not in gen:
            gen.append(node)
    return gen


def plot_start(x, y):
    plt.ion()
    plt.figure(figsize=(x, y))


def plot_update(path, data):
    plt.clf()

    x, y = zip(*path)
    plt.subplot(2, 2, 1)
    plt.scatter(x, y)
    plt.plot(x, y, color='red')

    x, y1, y2, y3 = zip(*data)
    plt.subplot(2, 2, 2)
    plt.plot(x, y1, color='red')

    plt.subplot(2, 2, 3)
    plt.plot(x, y2, color='red')

    plt.subplot(2, 2, 4)
    plt.plot(x, y3, color='red')
    plt.pause(0.01)
    plt.draw()


def distance(path):
    dist = 0
    for j in range(1, len(path)):
        dist = dist + math.sqrt((path[j][0] - path[j - 1][0]) ** 2 + (path[j][1] - path[j - 1][1]) ** 2)
    return dist


def tabu(path, max_threshold=25, tabu_size=25, plotting=True):
    def neighborhood_swap(entry):
        neighbor_list = []

        for i in range(1, len(entry) - 1):
            for j in range(i + 1, len(entry) - 1):
                swap = deepcopy(entry)
                swap[i], swap[j] = swap[j], swap[i]
                neighbor_list.append(swap)

        return neighbor_list

    def neighborhood_2opt(entry):
        neighbor_list = []

        for i in range(1, len(entry) - 2):
            for j in range(i + 2, len(entry)):
                opt = entry[:i] + entry[i:j][::-1] + entry[j:]
                neighbor_list.append(opt)

        return neighbor_list

    threshold = 0
    iteration = 0

    tabu_list = [path]
    best_solution, best_candidate = path, path
    data = []

    if plotting:
        plot_start(8, 8)
    while threshold < max_threshold:
        if random.random() > 0.5:
            neighbors = neighborhood_swap(best_candidate)
        else:
            neighbors = neighborhood_2opt(best_candidate)
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

        iteration += 1
        if plotting:
            data.append((len(data) + 1, distance(best_solution), len(tabu_list), threshold))
            plot_update(best_solution, data)
            print(f"Iteration: {iteration} Distance: {distance(best_solution)} Tabu list occupation: {len(tabu_list)} Threshold: {threshold}")

    return distance(best_solution)


def simulated_annealing(path, alpha=0.995, temperature=3, plotting=True):
    def random_neighbor_swap(entry):
        new_path = deepcopy(entry)
        x, y = 0, 0
        while x == y:
            x = random.randint(1, len(entry) - 2)
            y = random.randint(1, len(entry) - 2)

        new_path[x], new_path[y] = new_path[y], new_path[x]
        return new_path

    def random_neighbor_shuffle(entry):
        new_path = deepcopy(entry)
        x, y = 0, 0
        while x == y:
            x = random.randint(1, len(entry) - 2)
            y = random.randint(1, len(entry) - 2)

        shuffle = new_path[min(x, y):max(x, y) + 1]
        random.shuffle(shuffle)
        return new_path[:min(x, y)] + shuffle + new_path[max(x, y) + 1:]

    def random_neighbor_inverse(entry):
        new_path = deepcopy(entry)
        x, y = 0, 0
        while x == y:
            x = random.randint(1, len(entry) - 2)
            y = random.randint(1, len(entry) - 2)

        new_path[min(x, y):max(x, y) + 1] = new_path[min(x, y):max(x, y) + 1][::-1]
        return new_path

    def random_neighbor_insert(entry):
        new_path = deepcopy(entry)
        x, y = 0, 0
        while x == y:
            x = random.randint(1, len(entry) - 2)
            y = random.randint(1, len(entry) - 2)

        temp = new_path[x]
        new_path.pop(x)
        new_path.insert(y, temp)
        return new_path

    threshold = 0
    iteration = 0

    max_threshold = 8 * len(path)
    max_one_temp = 18 * len(path)

    best_solution = path
    data = []

    if plotting:
        plot_start(8, 8)

    while threshold < max_threshold:
        choice = random.choice([1, 2, 3, 4])

        if choice == 1:
            candidate = random_neighbor_swap(best_solution)
        elif choice == 2:
            candidate = random_neighbor_insert(best_solution)
        elif choice == 3:
            candidate = random_neighbor_shuffle(best_solution)
        else:
            candidate = random_neighbor_inverse(best_solution)

        delta = distance(best_solution) - distance(candidate)

        for i in range(max_one_temp):
            if delta > 0:
                best_solution = candidate
                threshold = 0
            elif delta < 0:
                prob = math.exp(delta/temperature)
                if random.random() < prob:
                    best_solution = candidate
                    threshold = 0

        if len(data) > 0:
            if data[-1][1] == distance(best_solution):
                threshold += 1

        temperature *= alpha
        iteration += 1
        data.append((len(data) + 1, distance(best_solution), temperature, threshold))

        if plotting:
            plot_update(best_solution, data)
            print(f"Iteration: {len(data)} Distance: {distance(best_solution)} Temperature: {temperature} Threshold: {threshold}")

    return distance(best_solution)


def test(n, algorithm):
    sum = 0
    for i in range(n):
        start = generation(20)
        time_start = time.time()
        algorithm(start, plotting=False)
        time_end = time.time()
        stop = time_end - time_start
        print(stop)
        sum += stop

    print(sum/n)


test(50, tabu)
