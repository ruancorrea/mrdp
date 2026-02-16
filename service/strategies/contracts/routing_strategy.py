from abc import ABC, abstractmethod
from typing import List, Dict, Any
from service.utils.structures import Delivery
import numpy as np

class RoutingStrategy(ABC):
    @abstractmethod
    def generate_routes(
        self,
        deliveries_by_vehicle: Dict[int, List[Delivery]],
        depot_origin: np.array,
        avg_speed_kmh: int
    ) -> Dict[int, Dict[str, Any]]:
        '''
        Recebe as entregas agrupadas por veículo e retorna os detalhes da rota para cada um.
        O dicionário de retorno deve conter a rota otimizada (sequência, tempos, etc.).
        '''
        pass