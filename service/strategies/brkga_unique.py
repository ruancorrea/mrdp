from typing import List, Dict, Any
import numpy as np
from service.strategies.contracts import UniqueStrategy
from service.utils.structures import Delivery, Vehicle
from service.algorithms.metaheuristics.brkga_unique import BRKGAUnique as BRKGAUniqueAlgorithm

class BRKGAUnique(UniqueStrategy):
    def generate_solution(
        self,
        deliveries: List[Delivery],
        vehicles: List[Vehicle],
        depot_origin: np.array,
        avg_speed_kmh: int
    ) -> Dict[int, Dict[str, Any]]:
        print("  -> Usando Estratégia Híbrida: BRKGA com Inserção Gulosa")
        brkga_unique = BRKGAUniqueAlgorithm(avg_speed_kmh=avg_speed_kmh)
        return brkga_unique.solve(deliveries, vehicles, depot_origin)
