from typing import List, Dict, Any
import numpy as np
from service.algorithms.metaheuristics.brkga import BRKGA
from service.strategies.contracts import RoutingStrategy
from service.utils.structures import Delivery, Point
from service.utils.distances import get_time_matrix, get_distance_matrix

class BRKGARouting(RoutingStrategy):
    def generate_routes(
        self,
        deliveries_by_vehicle: Dict[int, List[Delivery]],
        depot_origin: np.array,
        avg_speed_kmh: int
    ) -> Dict[int, Dict[str, Any]]:
        print("  -> Usando Estratégia de Roteirização: BRKGA")
        routes_details = {}
        brkga_solver = BRKGA()
        for vehicle_id, deliveries in deliveries_by_vehicle.items():
            if not deliveries: continue
            print(f"  -> Roteirizando para Veículo {vehicle_id} com {len(deliveries)} pedidos.")
            
            node_map = {i: d for i, d in enumerate(deliveries)}
            node_ids = list(node_map.keys())
            if isinstance(depot_origin, Point):
                depot_origin = np.array([depot_origin.lng, depot_origin.lat])
            cluster_points = np.array(
                [depot_origin.tolist()] + [[d.point.lng, d.point.lat]
                for d in deliveries]
            )
            distance_matrix = get_distance_matrix(cluster_points)
            time_matrix = get_time_matrix(distance_matrix, avg_speed_kmh)

            P_dt_map = {i: d.preparation_dt for i, d in node_map.items()}
            T_dt_map = {i: d.time_dt for i, d in node_map.items()}
            depot_index = len(deliveries)

            seq, _, asap_eval_dt = brkga_solver.solve(
                node_ids=node_ids,
                travel_time=time_matrix,
                P_dt_map=P_dt_map,
                T_dt_map=T_dt_map,
                depot_index=depot_index
            )
            asap_eval_dt["sequence"] = seq
            asap_eval_dt["node_map"] = node_map
            routes_details[vehicle_id] = asap_eval_dt

        return routes_details
