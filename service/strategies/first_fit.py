from typing import List, Dict, Any
import numpy as np
from service.strategies.contracts import ClusteringStrategy
from service.utils.structures import Delivery, Vehicle
from service.algorithms.heuristics.first_fit import FirstFit as FirstFitHeuristic

class FirstFit(ClusteringStrategy):
    def cluster(
        self,
        deliveries: List[Delivery],
        vehicles: List[Vehicle],
        depot_origin: np.array
    ) -> Dict[int, List[Delivery]]:
        print("  -> Usando Estratégia de Clusterização: Greedy (Sequential Assignment)")
        return FirstFitHeuristic().cluster(deliveries, vehicles, depot_origin)
