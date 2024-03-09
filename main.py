import math
import time
import random
import pandas
import matplotlib.pyplot as plt
from copy import deepcopy


def generation(n):  # Generate a list of random coordinates (nodes) with a specified length.
    gen = []
    while len(gen) != n:
        node = (random.randint(0, 200), random.randint(0, 200))
        if node not in gen:
            gen.append(node)

    gen.append(gen[0])
    return gen


def plot_start(name):   # Initialize a new plot for visualization.
    plt.ion()
    plt.figure(figsize=(8, 8), num=name)


def plot_update(data, specific):    # Update the plot with specified data.
    x2, y2, y3, y4, road = zip(*data)
    for i in range(len(road)):
        x1, y1 = zip(*(road[i]))
        plt.clf()
        plt.subplot(2, 2, 1)
        plt.scatter(x1, y1)
        plt.plot(x1, y1, color='red')

        plt.title(f'TOWNS: {len(x1) - 1}')

        plt.subplot(2, 2, 2)
        plt.plot(x2[:i+1], y2[:i+1], color='red')

        plt.title(f'DISTANCE: {y2[i]}')
        plt.xlabel('ITERATION')
        plt.ylabel('DISTANCE')

        plt.subplot(2, 2, 3)
        plt.plot(x2[:i+1], y3[:i+1], color='red')

        plt.title(f'{specific}: {y3[i]}')
        plt.xlabel('ITERATION')
        plt.ylabel(specific)

        plt.subplot(2, 2, 4)
        plt.plot(x2[:i+1], y4[:i+1], color='red')

        plt.title(f'THRESHOLD: {y4[i]}')
        plt.xlabel('ITERATION')
        plt.ylabel('THRESHOLD')

        plt.tight_layout()
        plt.pause(0.01)
        plt.draw()

    plt.pause(10)
    plt.close()


def distance(path):  # Calculate the total distance of a path connecting a series of points.
    dist = 0
    for j in range(1, len(path)):
        dist = dist + math.sqrt((path[j][0] - path[j - 1][0]) ** 2 + (path[j][1] - path[j - 1][1]) ** 2)
    return dist


def tabu(path, max_threshold=25, tabu_size=25):  # Solve the Traveling Salesman Problem using Tabu Search.
    def neighborhood_swap(entry):  # Inner function for generating list of all neighbors solutions by swapping cities
        neighbor_list = []

        for i in range(1, len(entry) - 1):
            for j in range(i + 1, len(entry) - 1):
                swap = deepcopy(entry)
                swap[i], swap[j] = swap[j], swap[i]
                neighbor_list.append(swap)

        return neighbor_list

    def neighborhood_2opt(entry):  # Inner function for generating list of all neighbors solutions using 2-opt technique
        neighbor_list = []

        for i in range(1, len(entry) - 2):
            for j in range(i + 2, len(entry)):
                opt = entry[:i] + entry[i:j][::-1] + entry[j:]
                neighbor_list.append(opt)

        return neighbor_list

    threshold = 0

    tabu_list = [path]
    best_solution, best_candidate = path, path
    data = [(1, round(distance(best_solution), 3), len(tabu_list), threshold, best_solution)]

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

        data.append((len(data) + 1, round(distance(best_solution), 3), len(tabu_list), threshold, best_solution))

    return data


def simulated_annealing(path, alpha=0.995, temperature=3):  # Solve the Traveling Salesman Problem using Simulated Annealing.
    def random_neighbor_swap(entry):  # Inner function for generating a new path by swapping two random cities.
        new_path = deepcopy(entry)
        x, y = 0, 0
        while x == y:
            x = random.randint(1, len(entry) - 2)
            y = random.randint(1, len(entry) - 2)

        new_path[x], new_path[y] = new_path[y], new_path[x]
        return new_path

    def random_neighbor_shuffle(entry):  # Inner function for generating a new path by shuffling a random segment of cities.
        new_path = deepcopy(entry)
        x, y = 0, 0
        while x == y:
            x = random.randint(1, len(entry) - 2)
            y = random.randint(1, len(entry) - 2)

        shuffle = new_path[min(x, y):max(x, y) + 1]
        random.shuffle(shuffle)
        return new_path[:min(x, y)] + shuffle + new_path[max(x, y) + 1:]

    def random_neighbor_inverse(entry):  # Inner function for generating a new path by inverting the order of a random segment of cities.
        new_path = deepcopy(entry)
        x, y = 0, 0
        while x == y:
            x = random.randint(1, len(entry) - 2)
            y = random.randint(1, len(entry) - 2)

        new_path[min(x, y):max(x, y) + 1] = new_path[min(x, y):max(x, y) + 1][::-1]
        return new_path

    def random_neighbor_insert(entry):   # Inner function for generating a new path by inserting a random city at a random position.
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

    max_threshold = 8 * len(path)
    max_one_temp = 18 * len(path)

    best_solution = path
    data = [(1, round(distance(best_solution), 3), round(temperature, 3), threshold, best_solution)]

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
            if data[-1][1] == round(distance(best_solution), 3):
                threshold += 1

        temperature *= alpha
        data.append((len(data) + 1, round(distance(best_solution), 3), round(temperature, 3), threshold, best_solution))

    return data


def ask_number(n):  # Prompt the user to input an iteration number.
    while True:
        try:
            decision_number = int(input("What iteration to choose: ")) - 1
        except ValueError:
            print("Enter integer")
            continue

        if n > decision_number >= 0:
            return decision_number
        else:
            print("Integer out of range")


def avg(lst, n):  # Calculate the average of a list of values.
    temp = 0
    for i in range(n):
        temp += lst[i][1]

    return temp/n


def test(n, towns, incremental=False):  # Perform multiple iterations of solving the Traveling Salesman Problem and provide analysis.
    results = []

    for i in range(n):
        start = generation(towns)
        time_start1 = time.time()
        data1 = tabu(start)
        time_end1 = time.time()

        time_start2 = time.time()
        data2 = simulated_annealing(start)
        time_end2 = time.time()

        results.append((data1, round(time_end1 - time_start1, 3), data2, round(time_end2 - time_start2, 3)))

        if incremental:
            print(towns)
            towns += 1

    df = pandas.DataFrame(results, columns=["tabu", "tabu_time", "sa", "sa_time"])
    df.to_csv('output/results.csv', index=False)

    tabu_df = []
    sa_df = []
    for i in range(0, n):
        tabu_df.append([pandas.DataFrame(df.tabu.at[i], columns=['iteration', 'distance', 'tabu_list', 'threshold', 'path']), df.tabu_time.at[i]])
        sa_df.append([pandas.DataFrame(df.sa.at[i], columns=['iteration', 'distance', 'temperature', 'threshold', 'path']), df.sa_time.at[i]])

    print("ANIMATION | STATS | HELP")
    while True:
        decision = input("Action: ").lower()
        if decision == "animation":
            iteration = ask_number(n)

            plot_start("Tabu Search algorithm")
            plot_update(tabu_df[iteration][0].values.tolist(), "TABU LIST SIZE")

            plot_start("Simulated Annealing algorithm")
            plot_update(sa_df[iteration][0].values.tolist(), "TEMPERATURE")

        elif decision == "stats":
            iteration = ask_number(n)
            print(f"Analysis of {iteration + 1}. iteration of Tabu search algorithm values\n")
            print(tabu_df[iteration][0].iloc[:, 1:].describe())

            print(f"\nTime of {iteration + 1}. iteration of Tabu search algorithm: {tabu_df[iteration][1]} s")
            print(f"Average time of Tabu search algorithm: {round(avg(tabu_df, n), 3)} s")

            print(f"\nAnalysis of {iteration + 1}. iteration of Simulated Annealing algorithm values\n")
            print(sa_df[iteration][0].iloc[:, 1:].describe())

            print(f"\nTime of {iteration + 1}. iteration of Simulated Annealing  algorithm: {sa_df[iteration][1]} s")
            print(f"Average time of Simulated Annealing algorithm: {round(avg(sa_df, n), 3)} s\n")

        elif decision == "help":
            print("animation returns visualization about certain solution")
            print("stats returns statistical values about certain solution")
            print("exit will end the program\n")

        elif decision == "exit":
            break

        else:
            print("Wrong input")


print("SAME | INCREMENTAL | HELP ")
while True:
    testik = input("Type test: ").lower()
    if testik == "incremental" or testik == "same":
        break
    elif testik == "help":
        print("same starts repeated test with constant of towns")
        print("incremental starts incremental test with incrementing towns")
    else:
        print("Wrong input")

while True:
    try:
        if testik == "same":
            town_number = int(input("How many towns to choose: "))
            times = int(input("How many iterations of problem to choose: "))
        else:
            town_number = 4
            times = int(input("Max number of towns to choose: ")) - 3

    except ValueError:
        print("Enter integer")
        continue

    if town_number > 0 and times > 0:
        if testik == "same":
            test(times, town_number)
        else:
            test(times, town_number, True)
        break
    else:
        print("Enter at least 4 towns and 1 iteration")
