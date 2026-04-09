from typing import List, Dict, Any
import numpy as np
from service.algorithms.metaheuristics.brkga import BRKGA
from service.strategies.contracts import RoutingStrategy
from service.utils.structures import Delivery, Point
from service.utils.distances import build_time_matrix
from service.algorithms.config import Config

class BRKGARouting(RoutingStrategy):
    def generate_routes(
        self,
        deliveries_by_vehicle: Dict[int, List[Delivery]],
        depot_origin: np.array,
        avg_speed_kmh: int
    ) -> Dict[int, Dict[str, Any]]:
        print("  -> Usando Estratégia de Roteirização: BRKGA")
        routes_details = {}
        
        # Carrega os parâmetros de serviço dinamicamente para os grupos
        config = Config.load_config("config.json")
        metric = config.distance_metric.value if hasattr(config.distance_metric, 'value') else config.distance_metric
        service_time = getattr(config, 'service_time_minutes', 0.0)

        brkga_solver = BRKGA()
        for vehicle_id, deliveries in deliveries_by_vehicle.items():
            if not deliveries: continue
            print(f"  -> Roteirizando para Veículo {vehicle_id} com {len(deliveries)} pedidos.")
            
            node_map = {i: d for i, d in enumerate(deliveries)}
            node_ids = list(node_map.keys())
            if isinstance(depot_origin, Point):
                depot_origin = np.array([depot_origin.lng, depot_origin.lat])
            
            depot_point = Point(lng=depot_origin[0], lat=depot_origin[1])
            all_points = np.array([depot_point] + [d.point for d in deliveries])
            time_matrix = build_time_matrix(all_points, metric=metric, avg_speed_kmh=avg_speed_kmh)

            P_dt_map = {i: d.preparation_dt for i, d in node_map.items()}
            T_dt_map = {i: d.time_dt for i, d in node_map.items()}
            depot_index = 0 # O depósito está posicionado no índice 0
            service_times_dict = {i: service_time for i in node_ids}

            seq, _, asap_eval_dt = brkga_solver.solve(
                node_ids=node_ids,
                travel_time=time_matrix,
                P_dt_map=P_dt_map,
                T_dt_map=T_dt_map,
                service_times=service_times_dict,
                depot_index=depot_index
            )
            asap_eval_dt["sequence"] = seq
            asap_eval_dt["node_map"] = node_map
            routes_details[vehicle_id] = asap_eval_dt

        return routes_details
