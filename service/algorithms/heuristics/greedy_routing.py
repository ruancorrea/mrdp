from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
import numpy as np
from datetime import timedelta

from service.utils.structures import Delivery
from service.utils.distances import get_distance_matrix, get_time_matrix
from service.utils.evaluate import Evaluate
from service.utils.time import Time


@dataclass
class GreedyRouting:
    '''
    Implementa a heurística de Inserção Mais Barata para criar uma rota para um veículo.

    A lógica é a seguinte:
    1.  Cria uma matriz de tempo de viagem entre o depósito e todas as entregas.
    2.  Inicializa a rota com a entrega mais próxima do depósito.
    3.  Iterativamente, para cada entrega ainda não roteirizada:
        a. Encontra a posição na rota atual onde a inserção dessa entrega causa o menor aumento de tempo.
        b. O custo de inserção de um ponto 'k' entre 'i' e 'j' é: tempo(i, k) + tempo(k, j) - tempo(i, j).
    4.  Adiciona a entrega que tem o menor custo de inserção em sua melhor posição.
    5.  Repete até que todas as entregas estejam na rota.
    6.  Calcula os tempos de chegada, penalidades e outros detalhes.
    '''
    deliveries: List[Delivery]
    depot_origin: np.ndarray
    avg_speed_kmh: int

    # Atributos que serão inicializados
    time_matrix: np.ndarray = field(init=False)
    num_deliveries: int = field(init=False)
    depot_idx: int = 0  # O depósito é sempre o índice 0 na matriz

    def __post_init__(self):
        '''
        Prepara os dados necessários para a execução do algoritmo.
        '''
        if not self.deliveries:
            self.num_deliveries = 0
            return

        self.num_deliveries = len(self.deliveries)

        # Monta a matriz de pontos (depósito no índice 0, entregas de 1 a n)
        all_points = np.array(
            [self.depot_origin.tolist()] + [[d.point.lat, d.point.lng] for d in self.deliveries]
        )
        dist_matrix = get_distance_matrix(all_points)
        self.time_matrix = get_time_matrix(dist_matrix, self.avg_speed_kmh)

    def solve(self) -> Dict[str, Any]:
        '''
        Executa a heurística da inserção mais barata.
        '''
        if not self.deliveries:
            return {}

        try:
            initial_route, unvisited_nodes = self._initialize_route()
            final_route = self._build_route(initial_route, unvisited_nodes)
            return self._format_output(final_route)
        except Exception as e:
            print(f"  -> ERRO na heurística de inserção: Não foi possível gerar rota. Detalhes: {e}")
            return {}

    def _initialize_route(self) -> Tuple[List[int], List[int]]:
        '''
        Encontra a entrega mais próxima do depósito para iniciar a rota.
        Retorna a rota inicial (com índices da matriz) e a lista de nós não visitados.
        '''
        # Índices na matriz de tempo para as entregas são de 1 a n
        times_from_depot = self.time_matrix[self.depot_idx][1:]
        first_node_matrix_idx = np.argmin(times_from_depot) + 1

        initial_route = [first_node_matrix_idx]

        # Nós não visitados (índices da matriz: 1 a n)
        unvisited_nodes = list(range(1, self.num_deliveries + 1))
        unvisited_nodes.remove(first_node_matrix_idx)

        return initial_route, unvisited_nodes

    def _find_best_insertion(self, route: List[int], unvisited_nodes: List[int]) -> Tuple[int, int]:
        '''
        Encontra o melhor nó para inserir na rota e a posição de inserção.
        Retorna o nó a ser inserido e a posição na rota.
        '''
        best_cost = float('inf')
        best_node_to_insert = -1
        best_position_in_route = -1

        for node_to_insert in unvisited_nodes:
            # Testa a inserção em todas as posições possíveis da rota.
            # Se a rota é [A, B], as posições de inserção são 0 (antes de A), 1 (entre A,B), 2 (depois de B).
            for i in range(len(route) + 1):
                prev_node = self.depot_idx if i == 0 else route[i-1]
                next_node = self.depot_idx if i == len(route) else route[i]

                cost = self.time_matrix[prev_node][node_to_insert] + self.time_matrix[node_to_insert][next_node] - self.time_matrix[prev_node][next_node]

                if cost < best_cost:
                    best_cost = cost
                    best_node_to_insert = node_to_insert
                    best_position_in_route = i

        return best_node_to_insert, best_position_in_route

    def _build_route(self, initial_route: List[int], unvisited_nodes: List[int]) -> List[int]:
        '''
        Constrói a rota iterativamente inserindo o nó mais barato.
        '''
        route = initial_route.copy()
        current_unvisited = unvisited_nodes.copy()

        while current_unvisited:
            node_to_insert, position = self._find_best_insertion(route, current_unvisited)

            if node_to_insert == -1:
                break  # Não encontrou inserção válida

            route.insert(position, node_to_insert)
            current_unvisited.remove(node_to_insert)

        return route

    def _format_output(self, route_matrix_indices: List[int]) -> Dict[str, Any]:
        '''
        Formata a rota final no padrão esperado pelo sistema.
        '''
        # Converte os índices da rota (1..n) para os índices de entrega originais (0..n-1)
        route_delivery_indices = [idx - 1 for idx in route_matrix_indices]

        # Prepara dados para a avaliação
        p_dt_map = {i: d.preparation_dt for i, d in enumerate(self.deliveries)}
        t_dt_map = {i: d.time_dt for i, d in enumerate(self.deliveries)}

        time_converter = Time()
        p_min, t_min, ref_ts = time_converter.datetimes_map_to_minutes(p_dt_map, t_dt_map)

        evaluator = Evaluate()
        eval_result = evaluator.evaluate_sequence(
            sequence=route_delivery_indices,
            time_matrix=self.time_matrix,
            p_min=p_min,
            t_min=t_min,
            depot_index=self.depot_idx
        )

        # Converte tempos de volta para datetime
        start_datetime = time_converter.minutes_to_datetime(eval_result["start_time"], ref_ts)
        return_datetime = start_datetime + timedelta(minutes=eval_result["total_route_time"])

        arrival_datetimes = [time_converter.minutes_to_datetime(t, ref_ts) for t in eval_result["arrival_times"]]
        arrivals_map = {node_idx: arrival_datetimes[i] for i, node_idx in enumerate(route_delivery_indices)}

        return {
            "sequence": route_delivery_indices,
            "total_penalty": eval_result["total_penalty"],
            "total_route_time": eval_result["total_route_time"],
            "start_datetime": start_datetime,
            "return_depot": return_datetime,
            "arrivals_map": arrivals_map,
        }
