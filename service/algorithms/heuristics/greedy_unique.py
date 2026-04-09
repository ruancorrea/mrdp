from datetime import timedelta
from copy import deepcopy
from typing import List, Dict, Any
import numpy as np

from service.strategies.contracts import UniqueStrategy
from service.utils.evaluate import Evaluate
from service.utils.structures import Delivery, Point, Vehicle
from service.utils.distances import build_time_matrix
from service.utils import Time

class GreedyUniqueStrategy(UniqueStrategy):
    '''
    Estratégia de etapa unica que constrói a solução de forma gulosa,
    inserindo um pedido por vez na melhor posição possível.
    '''
    def __init__(self):
        self.evaluate = Evaluate()
        self.time = Time()

    def generate_solution(
        self,
        deliveries: List[Delivery],
        vehicles: List[Vehicle],
        depot_origin: np.array,
        avg_speed_kmh: int,
        distance_metric: str = 'osrm',
        service_time_minutes: float = 0.0
    ) -> Dict[int, Dict[str, Any]]:
        """
        Gera uma solução completa (atribuição e roteirização) usando uma
        heurística de inserção gulosa.
        """
        if not deliveries or not vehicles:
            return {}

        # 1. Prepara os dados
        delivery_map = {d.id: d for d in deliveries}
        delivery_ids = list(delivery_map.keys())

        depot_origin = Point(lng=depot_origin[0], lat=depot_origin[1])

        all_points = [depot_origin] + [d.point for d in deliveries]

        # Mapeia IDs de entrega para índices na matriz de distância/tempo
        id_to_idx = {d_id: i + 1 for i, d_id in enumerate(delivery_ids)}
        idx_to_id = {i + 1: d_id for i, d_id in enumerate(delivery_ids)}
        depot_idx = 0

        time_matrix = build_time_matrix(all_points, metric=distance_metric, avg_speed_kmh=avg_speed_kmh)

        p_dt_map = {id_to_idx[d.id]: d.preparation_dt for d in deliveries}
        t_dt_map = {id_to_idx[d.id]: d.time_dt for d in deliveries}
        p_min, t_min, ref_ts = self.time.datetimes_map_to_minutes(p_dt_map, t_dt_map)
        service_times_dict = {i: service_time_minutes for i in range(1, len(delivery_ids) + 1)}

        routes = {v.id: [] for v in vehicles}
        remaining_capacities = {v.id: v.capacity for v in vehicles}
        unassigned_deliveries = set(delivery_ids)

        while unassigned_deliveries:
            best_insertion = None
            min_cost_increase = float('inf')

            for delivery_id in unassigned_deliveries:
                delivery = delivery_map[delivery_id]
                delivery_idx = id_to_idx[delivery_id]

                for vehicle in vehicles:
                    if remaining_capacities[vehicle.id] < delivery.size:
                        continue

                    current_route_indices = [id_to_idx[d_id] for d_id in routes[vehicle.id]]

                    original_cost = 0
                    if current_route_indices:
                        eval_res = self.evaluate.evaluate_sequence(current_route_indices, time_matrix, p_min, t_min, service_times=service_times_dict, depot_index=depot_idx)
                        original_cost = eval_res.total_penalty

                    for i in range(len(current_route_indices) + 1):
                        temp_route = current_route_indices[:i] + [delivery_idx] + current_route_indices[i:]

                        eval_res = self.evaluate.evaluate_sequence(temp_route, time_matrix, p_min, t_min, service_times=service_times_dict, depot_index=depot_idx)
                        new_cost = eval_res.total_penalty

                        cost_increase = new_cost - original_cost

                        if cost_increase < min_cost_increase:
                            min_cost_increase = cost_increase
                            best_insertion = (vehicle.id, i, delivery_id)

            if best_insertion:
                vehicle_id, position, delivery_id = best_insertion

                routes[vehicle_id].insert(position, delivery_id)
                remaining_capacities[vehicle_id] -= delivery_map[delivery_id].size
                unassigned_deliveries.remove(delivery_id)
            else:
                print(f"Não foi possível alocar {len(unassigned_deliveries)} pedidos.")
                break

        solution = {}
        for vehicle_id, route_ids in routes.items():
            if not route_ids:
                continue

            route_indices = [id_to_idx[d_id] for d_id in route_ids]
            final_eval = self.evaluate.evaluate_sequence(route_indices, time_matrix, p_min, t_min, service_times=service_times_dict, depot_index=depot_idx)

            arrival_datetimes = [self.time.minutes_to_datetime(t, ref_ts) for t in final_eval.arrival_times]
            start_datetime = self.time.minutes_to_datetime(final_eval.start_time, ref_ts)

            arrivals_map = {node_idx: arrival_datetimes[i] for i, node_idx in enumerate(route_indices)}
            penalties_map = {node_idx: final_eval.penalties[i] for i, node_idx in enumerate(route_indices)}
            return_time = start_datetime + timedelta(minutes=final_eval.total_route_time)

            node_map = {idx: delivery_map[idx_to_id[idx]] for idx in route_indices}

            vehicle_solution = deepcopy(final_eval.__dict__)
            vehicle_solution.update({
                "sequence": route_indices,
                "node_map": node_map,
                "deliveries": [delivery_map[d_id] for d_id in route_ids],
                "arrival_datetimes": arrival_datetimes,
                "start_datetime": start_datetime,
                "arrivals_map": arrivals_map,
                "penalties_map": penalties_map,
                "ref_timestamp_seconds": ref_ts,
                "return_depot": return_time,
            })
            solution[vehicle_id] = vehicle_solution

        return solution