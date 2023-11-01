import math
import random
import matplotlib.pyplot as plt


def generation(n):
    gen = []
    while len(gen) != n:
        node = [random.randint(0, 200), random.randint(0, 200)]
        if node not in gen:
            gen.append(node)
    return gen


def plot(road):
    x, y = zip(*road)
    plt.scatter(x, y)
    plt.plot(x, y, color='red')
    plt.show()


def distance(road):
    dist = 0
    for j in range(1, len(road)):
        dist = dist + math.sqrt((path[j][0] - path[j-1][0])**2 + (path[j][1] - path[j-1][1])**2)
    return dist


for i in range(10):
    path = generation(20)
    plot(path)
