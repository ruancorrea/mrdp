from dataclasses import dataclass
from enum import Enum
import json
import os
from typing import Optional

class ClusteringAlgorithm(Enum):
    CKMEANS = "ckmeans"
    GREEDY = "greedy_clustering"

class RoutingAlgorithm(Enum):
    BRKGA = "brkga"
    GREEDY = "greedy_routing"

class UniqueAlgorithm(Enum):
    GREEDY_INSERTION = "greedy_insertion"
    BRKGA_UNIQUE = "brkga_unique"
    MANUAL = "manual"

@dataclass
class Config:
    clustering_algo: Optional[ClusteringAlgorithm] = None
    routing_algo: Optional[RoutingAlgorithm] = None
    unique_algo: Optional[UniqueAlgorithm] = None
    dispatch_delay_buffer_minutes: int = 15
    urgent_order_time: int = 30
    avg_speed_kmh: int = 30

    def __str__(self):
        name = ''
        if self.clustering_algo:
            name += 'CLUSTERIZAÇÃO com ' + self.clustering_algo.value + ' | '
        if self.routing_algo:
            name += 'ROTEIRIZAÇÃO com ' + self.routing_algo.value
        if self.unique_algo:
            name += 'SEM ETAPAS com ' + self.unique_algo.value
        return name

    @staticmethod
    def load_config(config_path: str = "config.json") -> 'Config':
        '''Carrega a configuração a partir de um arquivo JSON.'''
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config_data = json.load(f)

            def get_enum(enum_cls, val):
                return enum_cls[val] if val and val in enum_cls.__members__ else None

            return Config(
                clustering_algo=get_enum(ClusteringAlgorithm, config_data.get("clustering_algo")),
                routing_algo=get_enum(RoutingAlgorithm, config_data.get("routing_algo")),
                unique_algo=get_enum(UniqueAlgorithm, config_data.get("unique_algo")),
                dispatch_delay_buffer_minutes=config_data.get("dispatch_delay_buffer_minutes", 15),
                urgent_order_time=config_data.get("urgent_order_time", 30),
                avg_speed_kmh=config_data.get("avg_speed_kmh", 30)
            )
        else:
            print(f"Aviso: {config_path} não encontrado. Usando configuração padrão.")
            return Config(unique_algo=UniqueAlgorithm.GREEDY_INSERTION)