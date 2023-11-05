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


def plot_update(data, specific):
    x2, y2, y3, y4, road = zip(*data)
    for i in range(len(road)):
        x1, y1 = zip(*(road[i]))
        plt.clf()
        plt.subplot(2, 2, 1)
        plt.scatter(x1, y1)
        plt.plot(x1, y1, color='red')
        plt.title(f'TOWNS: {len(x1)}')

        plt.subplot(2, 2, 2)
        plt.plot(x2[:i+1], y2[:i+1], color='red')
        plt.title(f'DISTANCE: {y2[i]}')

        plt.subplot(2, 2, 3)
        plt.plot(x2[:i+1], y3[:i+1], color='red')
        plt.title(f'{specific}: {y3[i]}')

        plt.subplot(2, 2, 4)
        plt.plot(x2[:i+1], y4[:i+1], color='red')
        plt.title(f'THRESHOLD: {y4[i]}')
        plt.pause(0.001)
        plt.draw()

    plt.pause(5)
    plt.close()


def distance(path):
    dist = 0
    for j in range(1, len(path)):
        dist = dist + math.sqrt((path[j][0] - path[j - 1][0]) ** 2 + (path[j][1] - path[j - 1][1]) ** 2)
    return dist


def tabu(path, max_threshold=25, tabu_size=25):
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

    tabu_list = [path]
    best_solution, best_candidate = path, path
    data = []

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


def simulated_annealing(path, alpha=0.995, temperature=3):
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

    max_threshold = 8 * len(path)
    max_one_temp = 18 * len(path)

    best_solution = path
    data = []

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


def ask_number(n):
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


def avg(lst, n):                               # Calculate average of a specific index in a list of lists.
    temp = 0
    for i in range(n):
        temp += lst[i][1]

    return temp/n


def test(n, towns):
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

    df = pandas.DataFrame(results, columns=["tabu", "tabu_time", "sa", "sa_time"])
    df.to_csv('output/results.csv', index=False)

    tabu_df = []
    sa_df = []
    for i in range(0, n):
        tabu_df.append([pandas.DataFrame(df.tabu.at[i], columns=['iteration', 'distance', 'tabu_list', 'threshold', 'path']), df.tabu_time.at[i]])
        sa_df.append([pandas.DataFrame(df.sa.at[i], columns=['iteration', 'distance', 'temperature', 'threshold', 'path']), df.sa_time.at[i]])

    print("Write help to get list of commands")

    while True:
        decision = input("Action: ").lower()
        if decision == "animation":
            iteration = ask_number(n)

            while True:
                decision = input("What algorithm to animate: ")
                if decision == "annealing":
                    plot_start(8, 8)
                    plot_update(sa_df[iteration][0].values.tolist(), "TEMPERATURE")
                    break

                elif decision == "tabu":
                    plot_start(8, 8)
                    plot_update(tabu_df[iteration][0].values.tolist(), "TABU LIST SIZE")
                    break
                else:
                    print("Wrong algorithm")

        elif decision == "statistic":
            iteration = ask_number(n)

            while True:
                decision = input("What algorithm to analyze: ")
                if decision == "annealing":
                    print(f"\nAnalysis of {iteration + 1}. iteration of Simulated Annealing algorithm values\n")
                    print(sa_df[iteration][0].describe())
                    print(f"\nAverage Simulated Annealing time: {round(avg(sa_df, n), 3)}")
                    break

                elif decision == "tabu":
                    print(f"\nAnalysis of {iteration + 1}. iteration of Tabu search algorithm values\n")
                    print(tabu_df[iteration][0].describe())
                    print(f"\nAverage Tabu search time: {round(avg(tabu_df, n), 3)}")
                    break
                else:
                    print("Wrong algorithm")

        elif decision == "help":
            print("animation returns visualization about certain solution")
            print("statistic returns statistical values about certain solution")
            print("exit will end the program")

        elif decision == "exit":
            break

        else:
            print("Wrong input")


test(10, 20)
