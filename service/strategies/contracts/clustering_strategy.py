from abc import ABC, abstractmethod
from typing import List, Dict
from service.utils.structures import Delivery, Vehicle
import numpy as np

class ClusteringStrategy(ABC):
    @abstractmethod
    def cluster(
        self,
        deliveries: List[Delivery],
        vehicles: List[Vehicle],
        depot_origin: np.array
    ) -> Dict[int, List[Delivery]]:
        '''
        Recebe uma lista de entregas e veículos, e retorna um dicionário
        mapeando o ID do veículo para uma lista de entregas atribuídas a ele.
        '''
        pass