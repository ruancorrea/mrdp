from typing import List, Dict, Any
import numpy as np
from service.strategies.contracts import ClusteringStrategy, RoutingStrategy
from service.utils.structures import Delivery, Vehicle
from service.algorithms.heuristics.greedy_routing import GreedyRouting as GreedyRoutingAlgorithm

class GreedyRouting(RoutingStrategy):
    def generate_routes(
        self,
        deliveries_by_vehicle:
        Dict[int, List[Delivery]],
        depot_origin: np.array,
        avg_speed_kmh: int
    ) -> Dict[int, Dict[str, Any]]:
        print("  -> Usando Estratégia de Roteirização: Greedy (Cheapest Insertion)")
        routes_details = {}
        for vehicle_id, deliveries in deliveries_by_vehicle.items():
            if not deliveries: continue
            greedy_routing = GreedyRoutingAlgorithm(deliveries, depot_origin, avg_speed_kmh)
            route_details = greedy_routing.solve()
            if route_details:
                # O `cheapest_insertion_heuristic` não retorna o node_map, então criamos aqui.
                # O `node_map` da heurística é {0: entrega_A, 1: entrega_B, ...}
                route_details["node_map"] = {i: d for i, d in enumerate(deliveries)}
            routes_details[vehicle_id] = route_details

        return routes_details
