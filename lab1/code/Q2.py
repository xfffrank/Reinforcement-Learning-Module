import sys
import numpy as np
import matplotlib.pyplot as plt


def allPairsShortestPath(g):
    dist = {}
    pred = {}
    for u in g:
        dist[u] = {}
        pred[u] = {}
        for v in g:
            dist[u][v] = sys.maxsize
            pred[u][v] = None
        dist[u][u] = 0
        pred[u][u] = None
        for v in g[u]:
            dist[u][v] = g[u][v]
            pred[u][v] = u
    for mid in g:
        for u in g:
            for v in g:
                newlen = dist[u][mid] + dist[mid][v]
                if newlen < dist[u][v]:
                    dist[u][v] = newlen
                    pred[u][v] = pred[mid][v]
    return dist, pred

def constructShortestPath(s, t, pred):
    path = [t]
    while t != s:
        t = pred[s][t]
        if t is None:
            return None
        path.insert(0,t)
    return path

def create_directed_graph(num_of_nodes, density):
    g = dict()
    for i in range(num_of_nodes):
        g[i] = dict()
    for node, dic in g.items():
        for neighbor in range(num_of_nodes):
            if node != neighbor and np.random.uniform(low=0.0, high=1.0) < density:
                dic[neighbor] = 1
    return g

def plot_pairwise_distances(distances, density):
    d_set = []
    for i in distances.values():
        for k, v in i.items():
            d_set.append(v)
    plt.hist(d_set)
    plt.title("Number of nodes: 100, Connection density: " + str(density))
    plt.show()


if __name__ == '__main__':
    for den in [0.1, 0.3, 0.5]:
        g = create_directed_graph(100, den)
        # print(g)
        dist, pred = allPairsShortestPath(g)
        plot_pairwise_distances(dist, den)  