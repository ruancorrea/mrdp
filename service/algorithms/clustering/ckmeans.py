from dataclasses import dataclass, field
from sklearn.cluster import KMeans
from service.utils.distances import euclidean_matrix
import numpy as np
import pulp

@dataclass
class CKMeans:
    n_clusters: int
    total_capacity: int
    max_iters: int = 20
    tol: float = 1e-4
    beta: float = 0.7
    random_state: int = 0

    labels_: np.ndarray = field(init=False, repr=False)
    cluster_centers_: np.ndarray = field(init=False, repr=False)

    def _adjust_capacity(self):
        '''Adjusts capacity based on a beta factor.'''
        # This logic seems a bit strange, but I'm keeping it as it was.
        # It seems it would be better to use total_capacity * beta directly
        # Also, the returned beta is not used.
        if self.total_capacity * self.beta < self.total_capacity:
            return self.total_capacity * self.beta
        return self.total_capacity
        # Returns the capacity scaled by beta, capped at the original total_capacity.
        # Effectively: min(total_capacity, total_capacity * beta)
        return min(self.total_capacity, self.total_capacity * self.beta)

    def _capacitated_assignment_mip(self, dist_mat, weights, capacity):
        '''Solves the assignment problem using mixed-integer programming.'''
        m, k = dist_mat.shape
        prob = pulp.LpProblem("cap_assign", pulp.LpMinimize)

        x = { (i,j): pulp.LpVariable(f"x_{i}_{j}", cat="Binary") for i in range(m) for j in range(k) }

        # Objective
        prob += pulp.lpSum(dist_mat[i, j] * x[(i, j)] for i in range(m) for j in range(k))

        # Each point assigned to exactly one cluster
        for i in range(m):
            prob += pulp.lpSum(x[(i, j)] for j in range(k)) == 1

        # Capacity constraints
        for j in range(k):
            prob += pulp.lpSum(weights[i] * x[(i, j)] for i in range(m)) <= capacity

        # Solve the problem
        prob.solve(pulp.PULP_CBC_CMD(msg=False))

        assign = np.zeros(m, dtype=int)
        for i in range(m):
            for j in range(k):
                val = pulp.value(x[(i, j)])
                if val is not None and val > 0.5:
                    assign[i] = j
                    break
        return assign

    def fit(self, X, weights):
        '''
        Performs capacitated k-means clustering.
        '''
        capacity = self._adjust_capacity()

        # Initial centers from standard KMeans
        km = KMeans(n_clusters=self.n_clusters, init="k-means++", n_init=10, random_state=self.random_state).fit(X)
        centers = km.cluster_centers_

        for _ in range(self.max_iters):
            dist = euclidean_matrix(X, centers)
            assign = self._capacitated_assignment_mip(dist, weights, capacity)

            # Update centers
            new_centers = np.zeros_like(centers)
            for j in range(self.n_clusters):
                idx = np.where(assign == j)[0]
                if len(idx) == 0:
                    farthest_point_idx = np.argmax(np.sum(dist, axis=1))
                    new_centers[j] = X[farthest_point_idx]
                else:
                    w = weights[idx]
                    new_centers[j] = np.average(X[idx], axis=0, weights=w)

            shift = np.linalg.norm(new_centers - centers)
            centers = new_centers
            if shift < self.tol:
                break

        # Final assignment
        final_dist = euclidean_matrix(X, centers)
        self.labels_ = self._capacitated_assignment_mip(final_dist, weights, capacity)
        self.cluster_centers_ = centers

        return self
