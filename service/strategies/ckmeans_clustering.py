from collections import defaultdict
from typing import List, Dict
import numpy as np
from service.strategies.contracts import ClusteringStrategy, RoutingStrategy, UniqueStrategy
from service.utils.structures import Delivery, Vehicle, Point
from service.algorithms.clustering.ckmeans import CKMeans

class CKMeansClustering(ClusteringStrategy):
    def cluster(
        self,
        deliveries: List[Delivery],
        vehicles: List[Vehicle],
        depot_origin: np.array
    ) -> Dict[int, List[Delivery]]:
        print("  -> Usando Estratégia de Clusterização: CK-Means")

        delivery_map = {i: d for i, d in enumerate(deliveries)}
        print(deliveries)
        points = np.array([[d.point.lat, d.point.lng] for d in deliveries])
        weights = np.array([d.size for d in deliveries])
        vehicle_capacity = int(np.mean([v.capacity for v in vehicles]))
        n_clusters = min(len(vehicles), len(deliveries))

        if n_clusters == 0: return {}
        ckmeans = CKMeans(
            n_clusters=n_clusters,
            total_capacity=vehicle_capacity,
            max_iters=20,
            tol=1e-4,
            beta=0.7,
            random_state=0
        )
        ckmeans.fit(points, weights)
        assignments, _ = ckmeans.labels_, ckmeans.cluster_centers_

        # Mapeia os clusters para os veículos disponíveis
        clusters = defaultdict(list)
        for delivery_idx, cluster_id in enumerate(assignments):
            clusters[cluster_id].append(deliveries[delivery_idx])

        vehicle_iterator = iter(vehicles)
        deliveries_by_vehicle = {}
        for cluster_id, deliveries_in_cluster in clusters.items():
            try:
                vehicle = next(vehicle_iterator)
                deliveries_by_vehicle[vehicle.id] = deliveries_in_cluster
            except StopIteration:
                break

        return deliveries_by_vehicle
