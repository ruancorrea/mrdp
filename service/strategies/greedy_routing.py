from typing import List, Dict, Any
import numpy as np
from service.strategies.contracts import ClusteringStrategy, RoutingStrategy
from service.utils.structures import Delivery, Vehicle
from service.algorithms.heuristics.greedy_routing import GreedyRouting as GreedyRoutingAlgorithm
from service.algorithms.config import Config

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
        
        config = Config.load_config("config.json")
        metric = config.distance_metric.value if hasattr(config.distance_metric, 'value') else config.distance_metric
        service_time = getattr(config, 'service_time_minutes', 0.0)

        for vehicle_id, deliveries in deliveries_by_vehicle.items():
            if not deliveries: continue
            greedy_routing = GreedyRoutingAlgorithm(
                deliveries, 
                depot_origin, 
                avg_speed_kmh,
                distance_metric=metric,
                service_time_minutes=service_time
            )
            route_details = greedy_routing.solve()
            if route_details:
                # O `cheapest_insertion_heuristic` não retorna o node_map, então criamos aqui.
                # O `node_map` da heurística é {0: entrega_A, 1: entrega_B, ...}
                route_details["node_map"] = {i: d for i, d in enumerate(deliveries)}
            routes_details[vehicle_id] = route_details

        return routes_details
