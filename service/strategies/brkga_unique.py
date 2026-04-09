from typing import List, Dict, Any
import numpy as np
from service.strategies.contracts import UniqueStrategy
from service.utils.structures import Delivery, Vehicle
from service.algorithms.metaheuristics.brkga_unique import BRKGAUnique as BRKGAUniqueAlgorithm
from service.algorithms.config import Config

class BRKGAUnique(UniqueStrategy):
    def generate_solution(
        self,
        deliveries: List[Delivery],
        vehicles: List[Vehicle],
        depot_origin: np.array,
        avg_speed_kmh: int
    ) -> Dict[int, Dict[str, Any]]:
        print("  -> Usando Estratégia Híbrida: BRKGA com Inserção Gulosa")
        
        config = Config.load_config("config.json")
        metric = config.distance_metric.value if hasattr(config.distance_metric, 'value') else config.distance_metric
        service_time = getattr(config, 'service_time_minutes', 0.0)

        brkga_unique = BRKGAUniqueAlgorithm(
            avg_speed_kmh=avg_speed_kmh,
            distance_metric=metric,
            service_time_minutes=service_time
        )
        return brkga_unique.solve(deliveries, vehicles, depot_origin)
