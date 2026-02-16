from abc import ABC, abstractmethod
from typing import List, Dict, Any
from service.utils.structures import Delivery, Vehicle
import numpy as np

class UniqueStrategy(ABC):
    @abstractmethod
    def generate_solution(
        self,
        deliveries: List[Delivery],
        vehicles: List[Vehicle],
        depot_origin: np.array,
        avg_speed_kmh: int
    ) -> Dict[int, Dict[str, Any]]:
        '''
        Recebe todas as entregas e veículos elegíveis e retorna um dicionário
        mapeando o ID do veículo para os detalhes da rota otimizada.
        Esta estratégia é responsável por atribuir e roteirizar em uma única etapa.
        '''
        pass