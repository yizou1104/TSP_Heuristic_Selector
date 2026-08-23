"""The four TSP construction heuristics used throughout the pipeline.

Each function takes an NxN distance matrix (list-of-lists or numpy array)
and returns ``(tour, cost)`` where ``tour`` is a closed cycle (starts and
ends at the same node) and ``cost`` is its total length.

Extracted, final (only) implementations from the notebook:
  - nearest_neighbour_tsp   <- nearest_neighbor,          original lines ~405-420
  - greedy_tsp_from_matrix  <- greedy_tsp_from_matrix,     original lines ~783-860
  - insertion_tsp_from_matrix <- insertion_tsp_from_matrix, original lines ~897-941
  - christofides_tsp        <- christofides_tsp_from_matrix, original lines ~647-670

The notebook also contains exact/near-exact solvers (brute_force_tsp_matrix,
held_karp_tsp_matrix, original lines ~481-646) and 2-opt/3-opt local search
(original lines ~1164-1327). Those are not part of the paper's 4-heuristic
selection problem and are intentionally not carried forward here.
"""

from __future__ import annotations

import networkx as nx
import numpy as np


def nearest_neighbour_tsp(dist_matrix, start: int = 0):
    """Nearest Neighbour construction heuristic.

    Complexity: O(n^2) — at each of the n-1 remaining steps, scans all n
    cities for the nearest unvisited one.
    """
    dist_matrix = np.asarray(dist_matrix)
    n = len(dist_matrix)
    visited = [False] * n
    tour = [start]
    cost = 0.0
    visited[start] = True
    current = start
    for _ in range(n - 1):
        next_city = int(
            np.argmin([dist_matrix[current][j] if not visited[j] else np.inf for j in range(n)])
        )
        visited[next_city] = True
        cost += dist_matrix[current][next_city]
        tour.append(next_city)
        current = next_city
    cost += dist_matrix[current][start]
    tour.append(start)
    return tour, cost


def greedy_tsp_from_matrix(dist):
    """Greedy (cheapest-link) construction heuristic.

    Sorts all O(n^2) candidate edges by weight, then repeatedly adds the
    shortest remaining edge that does not create a vertex of degree 3 or a
    sub-cycle shorter than a full Hamiltonian tour (using union-find).

    Complexity: O(n^2 log n), dominated by the edge sort.
    """
    n = len(dist)

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((dist[i][j], i, j))
    edges.sort()

    degree = [0] * n
    chosen_edges = []

    for w, u, v in edges:
        if degree[u] == 2 or degree[v] == 2:
            continue

        if find(u) == find(v):
            if len(chosen_edges) == n - 1:
                chosen_edges.append((u, v))
                degree[u] += 1
                degree[v] += 1
            continue

        chosen_edges.append((u, v))
        union(u, v)
        degree[u] += 1
        degree[v] += 1

        if len(chosen_edges) == n:
            break

    adj = {i: [] for i in range(n)}
    for u, v in chosen_edges:
        adj[u].append(v)
        adj[v].append(u)

    tour = [0]
    prev = -1
    curr = 0
    while True:
        nxt = adj[curr][0] if adj[curr][0] != prev else adj[curr][1]
        if nxt == 0:
            break
        tour.append(nxt)
        prev, curr = curr, nxt
    tour.append(0)

    cost = sum(dist[tour[i]][tour[i + 1]] for i in range(len(tour) - 1))
    return tour, cost


def insertion_tsp_from_matrix(dist):
    """Cheapest Insertion construction heuristic.

    Starts with a 2-node subtour (city 0 and its nearest neighbour), then
    repeatedly inserts the unvisited city at the tour position that
    minimises the cost increase ``dist[a][city] + dist[city][b] - dist[a][b]``.

    Complexity: naive O(n^3) — O(n) insertion steps, each scanning O(n)
    unvisited cities against O(n) tour edges.
    """
    n = len(dist)

    start = 0
    nearest = int(np.argmin([dist[start][j] if j != start else np.inf for j in range(n)]))

    tour = [start, nearest, start]
    unvisited = set(range(n))
    unvisited.remove(start)
    unvisited.remove(nearest)

    while unvisited:
        best_increase = np.inf
        best_city = None
        best_position = None

        for city in unvisited:
            for i in range(len(tour) - 1):
                a = tour[i]
                b = tour[i + 1]
                increase = dist[a][city] + dist[city][b] - dist[a][b]
                if increase < best_increase:
                    best_increase = increase
                    best_city = city
                    best_position = i + 1

        tour.insert(best_position, best_city)
        unvisited.remove(best_city)

    cost = sum(dist[tour[i]][tour[i + 1]] for i in range(len(tour) - 1))
    return tour, cost


def christofides_tsp(dist):
    """Christofides' algorithm.

    Builds a minimum spanning tree, finds a minimum-weight perfect matching
    on the odd-degree MST vertices (via networkx), combines them into an
    Eulerian multigraph, extracts an Eulerian circuit, then shortcuts
    repeated vertices to get a Hamiltonian tour.

    Complexity: O(n^3), dominated by the minimum-weight perfect matching
    step. Guarantees a 3/2-optimality bound on metric (triangle-inequality
    respecting) instances.
    """
    dist = np.asarray(dist)
    n = dist.shape[0]

    g = nx.Graph()
    for i in range(n):
        for j in range(i + 1, n):
            g.add_edge(i, j, weight=dist[i, j])

    tour = nx.algorithms.approximation.traveling_salesman.christofides(g)

    if tour[0] != tour[-1]:
        tour.append(tour[0])

    cost = sum(dist[tour[i], tour[i + 1]] for i in range(len(tour) - 1))
    return tour, cost


# Registry mapping the dataset's `heuristic` column values to their
# implementations, for use by build_dataset.py.
HEURISTICS = {
    "NN": nearest_neighbour_tsp,
    "Greedy": greedy_tsp_from_matrix,
    "Insertion": insertion_tsp_from_matrix,
    "CH": christofides_tsp,
}
