from typing import List, Dict, Any
import numpy as np
from service.utils.time import Time
from datetime import timedelta
from copy import deepcopy
from service.algorithms.heuristics.manual_chinainbox import ManualChinaInbox
from service.utils.evaluate import Evaluate
from service.strategies.contracts import UniqueStrategy
from service.utils.structures import Delivery, Vehicle, Point
from service.utils.distances import calculate_duration_matrix_m

class ManualChinaInboxUnique(UniqueStrategy):
    def generate_solution(
        self,
        deliveries: List[Delivery],
        vehicles: List[Vehicle],
        depot_origin: np.array,
        avg_speed_kmh: int
    ) -> Dict[int, Dict[str, Any]]:
        print("  -> Usando Estratégia Híbrida: Manual ChinaInbox (com Penalidades)")

        if not deliveries or not vehicles:
            return {}

        depot_point = Point(lng=depot_origin[0], lat=depot_origin[1])

        # --- 1. Preparar Matrizes Globais e Mapeamentos ---
        all_points = [depot_point] + [d.point for d in deliveries]
        delivery_to_matrix_idx = {d.id: i + 1 for i, d in enumerate(deliveries)}

        time_matrix_min = calculate_duration_matrix_m(all_points)
        time = Time()

        # --- 2. Agrupar entregas por veículo ---
        assignments = ManualChinaInbox().assign(
            deliveries,
            vehicles,
            depot_point
        )

        # --- 3. Avaliar cada rota e calcular resultados detalhados ---
        solution = {}
        for vehicle_id, deliveries_for_vehicle in assignments.items():
            if not deliveries_for_vehicle:
                continue

            # --- a. Preparar dados para evaluate_sequence ---
            # O `seq` para evaluate_sequence deve ser o índice da entrega na lista original de `deliveries`,
            # não o índice da matriz global.
            delivery_map = {i: d for i, d in enumerate(deliveries_for_vehicle)}
            node_ids = list(delivery_map.keys())

            # Criar matriz de tempo apenas para os nós desta rota
            route_matrix_indices = [0] + [delivery_to_matrix_idx[d.id] for d in deliveries_for_vehicle]
            route_time_matrix = time_matrix_min[np.ix_(route_matrix_indices, route_matrix_indices)]

            # Converter datetimes para minutos relativos
            P_dt_map = {i: d.preparation_dt for i, d in delivery_map.items()}
            T_dt_map = {i: d.time_dt for i, d in delivery_map.items()}
            P_min, T_min, ref_ts = time.datetimes_map_to_minutes(P_dt_map, T_dt_map)

            # --- b. Chamar evaluate_sequence ---
            # A sequência aqui é simples, por ordem de agrupamento.
            # O depot_index para esta sub-matriz é 0.
            ev_min = Evaluate().evaluate_sequence(
                seq=node_ids,
                travel_time=route_time_matrix,
                P_min=P_min,
                T_min=T_min,
                depot_index=0
            )

            # --- c. Converter resultados para datetimes e formato final ---
            start_datetime = time.minutes_to_datetime(ev_min.start_time, ref_ts)
            return_time = start_datetime + timedelta(minutes=ev_min.total_route_time)

            arrivals_map = {
                node: time.minutes_to_datetime(ev_min.arrival_times[i], ref_ts)
                for i, node in enumerate(node_ids)
            }

            ev_with_dt = deepcopy(ev_min.__dict__)
            ev_with_dt.update({
                "sequence": node_ids,
                "node_map": delivery_map,
                "arrival_datetimes": list(arrivals_map.values()),
                "start_datetime": start_datetime,
                "arrivals_map": arrivals_map,
                "ref_timestamp_seconds": ref_ts,
                "return_depot": return_time,
            })
            solution[vehicle_id] = ev_with_dt

        return solution
