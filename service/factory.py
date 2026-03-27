from service.algorithms.config import (
    Config,
    ClusteringAlgorithm,
    RoutingAlgorithm,
    UniqueAlgorithm
)

from service.strategies import (
    CKMeansClustering,
    FirstFit,
    BRKGARouting,
    GreedyRouting,
    GreedyUnique,
    BRKGAUnique,
    ManualChinaInboxUnique,
    ManualAssignmentUnique
)

def get_strategies(config: Config):
    '''Fábrica que retorna as instâncias de estratégia com base na config.'''

    # Se uma estratégia híbrida for definida, ela tem precedência
    if config.unique_algo:
        unique_strategy_map = {
            UniqueAlgorithm.GREEDY_INSERTION: GreedyUnique,
            UniqueAlgorithm.BRKGA_UNIQUE: BRKGAUnique,
            UniqueAlgorithm.MANUAL: ManualAssignmentUnique,
            UniqueAlgorithm.MANUAL_CHINAINBOX: ManualChinaInboxUnique,
        }
        return None, None, unique_strategy_map[config.unique_algo]()

    # Caso contrário, retorna as estratégias de clusterização e roteamento
    clustering_strategy_map = {
        ClusteringAlgorithm.CKMEANS: CKMeansClustering,
        ClusteringAlgorithm.GREEDY: FirstFit,
    }

    routing_strategy_map = {
        RoutingAlgorithm.BRKGA: BRKGARouting,
        RoutingAlgorithm.GREEDY: GreedyRouting,
    }

    clustering_strategy = clustering_strategy_map.get(config.clustering_algo)
    routing_strategy = routing_strategy_map.get(config.routing_algo)

    return clustering_strategy(), routing_strategy(), None