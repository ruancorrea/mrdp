from service.algorithms.heuristics.greedy_unique import GreedyUniqueStrategy as GreedyUniqueHeuristic
from service.strategies.contracts import UniqueStrategy
from typing import List, Dict, Any
import numpy as np
from service.utils.structures import Delivery, Vehicle
from service.algorithms.config import Config

class GreedyUnique(UniqueStrategy):
    def generate_solution(
        self,
        deliveries: List[Delivery],
        vehicles: List[Vehicle],
        depot_origin: np.array,
        avg_speed_kmh: int
    ) -> Dict[int, Dict[str, Any]]:
        print("  -> Usando Estratégia Híbrida: Greedy Insertion")
        
        config = Config.load_config("config.json")
        metric = config.distance_metric.value if hasattr(config.distance_metric, 'value') else config.distance_metric
        service_time = getattr(config, 'service_time_minutes', 0.0)

        solver = GreedyUniqueHeuristic()
        return solver.generate_solution(
            deliveries, 
            vehicles, 
            depot_origin, 
            avg_speed_kmh,
            distance_metric=metric,
            service_time_minutes=service_time
        )
