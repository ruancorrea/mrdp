from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
import numpy as np
from datetime import timedelta

from service.utils.structures import Delivery, Point
from service.utils.distances import build_time_matrix
from service.utils.evaluate import Evaluate
from service.utils.time import Time


@dataclass
class GreedyRouting:
    '''
    Implementa a heurística de Inserção Mais Barata (Cheapest Insertion) adaptada para criar uma rota para um veículo.
    Nesta variante para o MDRP-SU, o "custo" minimizado na inserção é a penalidade por atraso (SLA), e não apenas o tempo de viagem.

    A lógica é a seguinte:
    1.  Cria uma matriz de tempo de viagem entre o depósito e todas as entregas.
    2.  Inicializa uma rota vazia.
    3.  Iterativamente, para cada entrega ainda não roteirizada:
        a. Avalia a inserção do nó candidato em todas as posições possíveis da rota atual.
        b. O custo da inserção é obtido simulando a sequência completa com o evaluator, retornando a penalidade total baseada no tempo de preparo e nos prazos (SLA).
    4.  Adiciona a entrega que causa o menor aumento na penalidade na sua melhor posição (usando o tempo de rota como desempate).
    5.  Repete até que todas as entregas estejam na rota.
    6.  Calcula os tempos de chegada, penalidades e outros detalhes.
    '''
    deliveries: List[Delivery]
    depot_origin: np.ndarray
    avg_speed_kmh: int
    distance_metric: str = 'osrm'
    service_time_minutes: float = 0.0

    # Atributos que serão inicializados
    time_matrix: np.ndarray = field(init=False)
    num_deliveries: int = field(init=False)
    depot_idx: int = 0  # O depósito é sempre o índice 0 na matriz
    p_min: Dict[int, float] = field(init=False)
    t_min: Dict[int, float] = field(init=False)
    ref_ts: float = field(init=False)
    evaluator: Any = field(init=False)
    time_converter: Any = field(init=False)

    def __post_init__(self):
        '''
        Prepara os dados necessários para a execução do algoritmo.
        '''
        if not self.deliveries:
            self.num_deliveries = 0
            return

        self.num_deliveries = len(self.deliveries)
        self.depot_origin = np.array(self.depot_origin)

        depot_origin = Point(lng=self.depot_origin[0], lat=self.depot_origin[1])

        # Monta a matriz de pontos (depósito no índice 0, entregas de 1 a n)
        all_points = [depot_origin] + [d.point for d in self.deliveries]
        self.time_matrix = build_time_matrix(all_points, metric=self.distance_metric, avg_speed_kmh=self.avg_speed_kmh)

        # Prepara dados para avaliação do SLA real nas tentativas de inserção
        p_dt_map = {i + 1: d.preparation_dt for i, d in enumerate(self.deliveries)}
        t_dt_map = {i + 1: d.time_dt for i, d in enumerate(self.deliveries)}
        self.time_converter = Time()
        self.p_min, self.t_min, self.ref_ts = self.time_converter.datetimes_map_to_minutes(p_dt_map, t_dt_map)
        self.evaluator = Evaluate()

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
        Inicia a rota vazia. A heurística de inserção descobrirá o melhor primeiro
        nó priorizando as restrições de tempo/SLA ao invés de usar apenas distância física.
        '''
        initial_route = []

        # Nós não visitados (índices da matriz: 1 a n)
        unvisited_nodes = list(range(1, self.num_deliveries + 1))

        return initial_route, unvisited_nodes

    def _find_best_insertion(self, route: List[int], unvisited_nodes: List[int]) -> Tuple[int, int]:
        '''
        Encontra o melhor nó para inserir na rota e a posição de inserção.
        Retorna o nó a ser inserido e a posição na rota.
        '''
        best_cost = float('inf')
        best_route_time = float('inf')
        best_node_to_insert = -1
        best_position_in_route = -1

        for node_to_insert in unvisited_nodes:
            # Testa a inserção em todas as posições possíveis da rota.
            for i in range(len(route) + 1):
                candidate_route = route.copy()
                candidate_route.insert(i, node_to_insert)
                
                candidate_seq = candidate_route.copy()
                service_times_dict = {idx: self.service_time_minutes for idx in candidate_seq}
                
                eval_result = self.evaluator.evaluate_sequence(
                    seq=candidate_seq,
                    travel_time=self.time_matrix,
                    P_min=self.p_min,
                    T_min=self.t_min,
                    service_times=service_times_dict,
                    depot_index=self.depot_idx
                )

                cost = eval_result.total_penalty
                route_time = eval_result.total_route_time

                # Seleciona o que causa menor penalidade. Desempata por tempo mínimo de rota.
                if cost < best_cost or (cost == best_cost and route_time < best_route_time):
                    best_cost = cost
                    best_route_time = route_time
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
        service_times_dict = {idx: self.service_time_minutes for idx in route_matrix_indices}
        eval_result = self.evaluator.evaluate_sequence(
            seq=route_matrix_indices,
            travel_time=self.time_matrix,
            P_min=self.p_min,
            T_min=self.t_min,
            service_times=service_times_dict,
            depot_index=self.depot_idx
        )

        # Converte tempos de volta para datetime
        start_datetime = self.time_converter.minutes_to_datetime(eval_result.start_time, self.ref_ts)
        return_datetime = start_datetime + timedelta(minutes=eval_result.total_route_time)

        arrival_datetimes = [self.time_converter.minutes_to_datetime(t, self.ref_ts) for t in eval_result.arrival_times]
        
        # Reverte os índices de 1..n para 0..n-1 para manter compatibilidade com o chamador
        final_seq = [idx - 1 for idx in route_matrix_indices]
        arrivals_map = {idx - 1: arrival_datetimes[i] for i, idx in enumerate(route_matrix_indices)}

        return {
            "sequence": final_seq,
            "total_penalty": eval_result.total_penalty,
            "total_route_time": eval_result.total_route_time,
            "start_datetime": start_datetime,
            "return_depot": return_datetime,
            "arrivals_map": arrivals_map,
        }
