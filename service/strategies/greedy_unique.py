from service.algorithms.heuristics.greedy_unique import GreedyUniqueStrategy as GreedyUniqueHeuristic
from service.strategies.contracts import UniqueStrategy
from typing import List, Dict, Any
import numpy as np
from service.utils.structures import Delivery, Vehicle

class GreedyUnique(UniqueStrategy):
    def generate_solution(
        self,
        deliveries: List[Delivery],
        vehicles: List[Vehicle],
        depot_origin: np.array,
        avg_speed_kmh: int
    ) -> Dict[int, Dict[str, Any]]:
        print("  -> Usando Estratégia Híbrida: Greedy Insertion")
        solver = GreedyUniqueHeuristic()
        return solver.generate_solution(deliveries, vehicles, depot_origin, avg_speed_kmh)
