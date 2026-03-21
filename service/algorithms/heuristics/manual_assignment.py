from typing import List, Dict
import numpy as np

from service.utils.structures import Delivery, Vehicle

class ManualAssignment:
    def __init__(self, max_travel_time: float = 8.0, stop_penalty_min: float = 2.0):
        self.max_travel_time = max_travel_time
        # stop_penalty_min is not used in this implementation
        self.stop_penalty_min = stop_penalty_min

    def assign(
        self,
        deliveries: List[Delivery],
        vehicles: List[Vehicle],
        time_matrix: np.ndarray,
        delivery_indices: Dict[str, int],
        depot_idx: int = 0
    ) -> Dict[int, List[Delivery]]:
        '''
        Agrupa entregas para veículos com base em uma heurística manual simples.
        Buscando simular a definição das rotas sem uso de algoritmos
        Opera com base em tempo de viagem, não mais em distância.
        Retorna um dicionário mapeando ID do veículo para uma lista de entregas.
        '''

        # 1. Enriquecer entregas com tempo de viagem do depósito e folga (slack)
        enriched_deliveries = []
        for d in deliveries:
            delivery_idx = delivery_indices[d.id]
            travel_time = time_matrix[depot_idx, delivery_idx]
            slack = d.time - travel_time

            enriched_deliveries.append({
                "delivery": d,
                "travel_time": travel_time,
                "slack": slack
            })

        # 2. Ordenar por urgência (menor slack primeiro)
        enriched_deliveries.sort(key=lambda x: x["slack"])

        # 3. Agrupar rotas e alocar aos veículos
        assignments = {v.id: [] for v in vehicles}
        vehicles_sorted = sorted(vehicles, key=lambda v: v.capacity, reverse=True)
        assigned_deliveries = set()

        for vehicle in vehicles_sorted:
            for current_enriched in enriched_deliveries:
                if current_enriched["delivery"].id in assigned_deliveries:
                    continue

                route = [current_enriched["delivery"]]
                assigned_deliveries.add(current_enriched["delivery"].id)

                # Tenta agrupar mais pedidos na mesma rota
                for candidate_enriched in enriched_deliveries:
                    if len(route) >= vehicle.capacity:
                        break
                    if candidate_enriched["delivery"].id in assigned_deliveries:
                        continue

                    # Regra de agrupamento por tempo de viagem do depósito
                    if candidate_enriched["travel_time"] <= self.max_travel_time:
                        route.append(candidate_enriched["delivery"])
                        assigned_deliveries.add(candidate_enriched["delivery"].id)

                assignments[vehicle.id].extend(route)
                # Se um veículo é totalmente preenchido, podemos parar de alocar para ele
                if len(assignments[vehicle.id]) >= vehicle.capacity:
                    break

        return assignments