from typing import List, Dict, Any
import numpy as np
import random
from copy import deepcopy
from datetime import timedelta

from service.utils.structures import Delivery, Vehicle, Point
from service.utils.distances import get_distance_matrix, get_time_matrix, calculate_duration_matrix_m
from service.utils.time import Time
from service.utils.evaluate import Evaluate

class BRKGAUnique:
    '''
    Implementa o Biased Random-Key Genetic Algorithm (BRKGA) para resolver o
    problema de roteamento de veículos com janelas de tempo.
    '''

    def __init__(
        self,
        pop_size: int=50,
        elite_frac: float=0.3,
        mutant_frac: float=0.15,
        bias: float=0.7,
        max_gens: int=70,
        no_improve_limit: int=15,
        avg_speed_kmh: int=30,
        unassigned_penalty: int=100000,
        penalty_multiplier: int=1000,
        min_block_penalty: float=5.0,
        penalty_per_block: int=100
    )-> None:
        '''
        Explicação:
        Este é o construtor da classe. Ele inicializa o resolvedor BRKGA com todos os
        hiperparâmetros necessários para o algoritmo genético (como tamanho da população,
        fração de elite) e para o problema em si (como velocidade média, penalidades).
        Ele também configura as classes auxiliares para cálculos de tempo e avaliação.
        '''
        # Hiperparâmetros do BRKGA
        self.pop_size = pop_size
        self.elite_frac = elite_frac
        self.mutant_frac = mutant_frac
        self.bias = bias
        self.max_gens = max_gens
        self.no_improve_limit = no_improve_limit

        # Parâmetros específicos do problema
        self.avg_speed_kmh = avg_speed_kmh
        self.unassigned_penalty = unassigned_penalty
        self.penalty_multiplier = penalty_multiplier

        # Inicializa os ajudantes
        self.time_helper = Time() # Usa o fuso horário padrão
        self.evaluate_helper = Evaluate(min_block=min_block_penalty, penalty_per_block=penalty_per_block)

    def _decode_chromosome(
        self,
        chromosome: List[float],
        deliveries: List[Delivery],
        vehicles: List[Vehicle],
        time_matrix: np.ndarray,
        p_min: np.ndarray,
        t_min: np.ndarray,
        id_to_idx: Dict[str, int]
    ) -> Dict[str, Any]:
        '''
        Decodifica um cromossomo (lista de chaves aleatórias) em uma solução de roteamento válida.

        EXPLICAÇÃO DETALHADA DO FUNCIONAMENTO:
        ---------------------------------------
        1. O CONCEITO DE CHAVES ALEATÓRIAS (Random Keys):
           O cromossomo não contém a rota diretamente. Ele contém uma lista de números reais (ex: [0.2, 0.8, 0.5]),
           onde cada número corresponde a um pedido. Esses números funcionam como "prioridades".
           - Passo 1: Associamos cada gene ao seu pedido correspondente.
           - Passo 2: Ordenamos os pedidos com base no valor do gene (do menor para o maior).
           Isso cria uma fila de prioridade determinística. Se o gene do Pedido A é 0.1 e do Pedido B é 0.9,
           o Pedido A será processado e roteirizado antes do Pedido B. O algoritmo genético evolui esses valores
           para encontrar a melhor ordem de inserção.

        2. CONSTRUÇÃO GULOSA (Greedy Insertion):
           Com a lista de pedidos ordenada, o algoritmo tenta inserir um pedido por vez na solução parcial já construída.
           Para o pedido atual da lista, ele faz a seguinte pergunta:
           "Qual é o veículo e qual é a posição na rota desse veículo onde este pedido se encaixa com o MENOR custo adicional?"

        3. BUSCA EXAUSTIVA DE POSIÇÕES:
           Para responder à pergunta acima, o algoritmo itera sobre:
           - Todos os veículos disponíveis.
           - Se o veículo tem capacidade (peso/volume) para o pedido.
           - Todas as posições possíveis na rota atual desse veículo (início, meio, fim).

        4. CÁLCULO DO CUSTO MARGINAL:
           Para cada posição testada, calcula-se o aumento no custo total (Tempo + Penalidades).
           A posição que resultar no menor aumento (delta) é escolhida.

        5. ATRIBUIÇÃO:
           O pedido é inserido na melhor posição encontrada e a capacidade do veículo é atualizada.
           Se não couber em nenhum veículo, aplica-se uma penalidade de "não atribuído".
        '''
        # Ordena as entregas com base nos valores de prioridade do cromossomo
        indexed_deliveries = list(enumerate(deliveries))
        sorted_indexed_deliveries = sorted(indexed_deliveries, key=lambda x: chromosome[x[0]])

        routes = {v.id: [] for v in vehicles}
        route_costs = {v.id: 0 for v in vehicles}
        remaining_capacities = {v.id: v.capacity for v in vehicles}
        unassigned_penalty_total = 0

        for _, delivery in sorted_indexed_deliveries:
            best_insertion = {"vehicle_id": None, "position": -1, "cost_increase": float('inf')}
            delivery_idx = id_to_idx[delivery.id]

            for vehicle in vehicles:
                if remaining_capacities[vehicle.id] < delivery.size:
                    continue

                current_route_ids = routes[vehicle.id]
                current_route_indices = [id_to_idx[d_id] for d_id in current_route_ids]
                original_cost = route_costs[vehicle.id]

                for i in range(len(current_route_indices) + 1):
                    temp_route = current_route_indices[:i] + [delivery_idx] + current_route_indices[i:]
                    eval_res = self.evaluate_helper.evaluate_sequence(temp_route, time_matrix, p_min, t_min, depot_index=0)
                    new_cost = eval_res.total_penalty * self.penalty_multiplier + eval_res.total_route_time
                    cost_increase = new_cost - original_cost

                    if cost_increase < best_insertion["cost_increase"]:
                        best_insertion = {"vehicle_id": vehicle.id, "position": i, "cost_increase": cost_increase}

            if best_insertion["vehicle_id"]:
                v_id = best_insertion["vehicle_id"]
                routes[v_id].insert(best_insertion["position"], delivery.id)
                remaining_capacities[v_id] -= delivery.size
                updated_route_indices = [id_to_idx[d_id] for d_id in routes[v_id]]
                final_eval = self.evaluate_helper.evaluate_sequence(updated_route_indices, time_matrix, p_min, t_min, depot_index=0)
                route_costs[v_id] = final_eval.total_penalty * self.penalty_multiplier + final_eval.total_route_time
            else:
                unassigned_penalty_total += self.unassigned_penalty

        total_penalty = unassigned_penalty_total
        total_route_time = 0
        for vehicle_id, route_ids in routes.items():
            if route_ids:
                route_indices = [id_to_idx[d_id] for d_id in route_ids]
                final_eval = self.evaluate_helper.evaluate_sequence(route_indices, time_matrix, p_min, t_min, depot_index=0)
                total_penalty += final_eval.total_penalty
                total_route_time += final_eval.total_route_time

        return {"total_penalty": total_penalty, "total_route_time": total_route_time, "routes": routes}

    def _format_solution(
        self,
        chromosome: List[float],
        deliveries: List[Delivery],
        vehicles: List[Vehicle],
        depot_origin: np.array,
        time_matrix: np.ndarray,
        p_min: np.ndarray,
        t_min: np.ndarray,
        ref_ts: int,
        id_to_idx: Dict[str, int]
    ) -> Dict[int, Dict[str, Any]]:
        '''
        Explicação:
        Quando o algoritmo encontra o melhor cromossomo, este método o formata
        em um dicionário detalhado e fácil de entender. Ele reconstrói as rotas
        e as enriquece com informações extras, como horários de chegada,
        penalidades e a sequência completa de paradas.
        '''
        decoded = self._decode_chromosome(chromosome, deliveries, vehicles, time_matrix, p_min, t_min, id_to_idx)
        routes = decoded.get("routes", {})

        solution = {}
        delivery_map = {d.id: d for d in deliveries}

        for v in vehicles:
            route_ids = routes.get(v.id)
            if not route_ids: continue

            route_indices = [id_to_idx[d_id] for d_id in route_ids]
            final_eval = self.evaluate_helper.evaluate_sequence(route_indices, time_matrix, p_min, t_min, depot_index=0)

            arrival_datetimes = [self.time_helper.minutes_to_datetime(t, ref_ts) for t in final_eval.arrival_times]
            start_datetime = self.time_helper.minutes_to_datetime(final_eval.start_time, ref_ts)

            return_depot = start_datetime + timedelta(minutes=final_eval.total_route_time)
            node_map = {idx: delivery_map[d_id] for idx, d_id in zip(route_indices, route_ids)}
            arrivals_map = {idx: dt for idx, dt in zip(route_indices, arrival_datetimes)}
            penalties_map = {idx: p for idx, p in zip(route_indices, final_eval.penalties)}

            solution[v.id] = {
                **final_eval.__dict__,
                "sequence": route_indices,
                "deliveries": [delivery_map[d_id] for d_id in route_ids],
                "arrival_datetimes": arrival_datetimes,
                "start_datetime": start_datetime,
                "node_map": node_map,
                "arrivals_map": arrivals_map,
                "penalties_map": penalties_map,
                "return_depot": return_depot,
                "ref_timestamp_seconds": ref_ts
            }
        return solution

    def solve(self, deliveries: List[Delivery], vehicles: List[Vehicle], depot_origin: np.array) -> Dict[int, Dict[str, Any]]:
        '''
        Explicação:
        Este é o motor principal do algoritmo. Ele orquestra todo o processo do BRKGA.
        Começa criando uma população inicial de cromossomos aleatórios. Então, ao longo de
        muitas gerações, ele decodifica cada cromossomo em uma solução, avalia sua qualidade (fitness)
        e usa os princípios da evolução — sobrevivência do mais apto (elitismo), reprodução (crossover)
        e mutação aleatória — para criar uma nova população, esperançosamente melhor, para a próxima geração.
        Ele para quando atinge o número máximo de gerações ou deixa de ver melhorias.
        '''
        if not deliveries or not vehicles: return {}

        # Preparação de dados
        num_deliveries = len(deliveries)
        depot_origin = Point(lng=depot_origin[0], lat=depot_origin[1])
        all_points = np.array([depot_origin()] + [[d.point] for d in deliveries])
        #time_matrix = get_time_matrix(get_distance_matrix(all_points), self.avg_speed_kmh)
        time_matrix = calculate_duration_matrix_m(all_points)
        id_to_idx = {d.id: i + 1 for i, d in enumerate(deliveries)}
        p_dt_map = {id_to_idx[d.id]: d.preparation_dt for d in deliveries}
        t_dt_map = {id_to_idx[d.id]: d.time_dt for d in deliveries}
        p_min, t_min, ref_ts = self.time_helper.datetimes_map_to_minutes(p_dt_map, t_dt_map)

        population = [[random.random() for _ in range(num_deliveries)] for _ in range(self.pop_size)]
        best_fitness_ever = (float('inf'), float('inf'))
        best_chrom_ever = None
        no_improve_count = 0

        for gen in range(self.max_gens):
            fitness_results = []
            for chrom in population:
                decoded = self._decode_chromosome(chrom, deliveries, vehicles, time_matrix, p_min, t_min, id_to_idx)
                fitness = (decoded["total_penalty"], decoded["total_route_time"])
                fitness_results.append((fitness, chrom))

            population_sorted = sorted(fitness_results, key=lambda x: x[0])

            if population_sorted[0][0] < best_fitness_ever:
                best_fitness_ever, best_chrom_ever = population_sorted[0]
                no_improve_count = 0
            else:
                no_improve_count += 1

            if no_improve_count >= self.no_improve_limit:
                break

            elite_size = max(1, int(self.pop_size * self.elite_frac))
            mutant_size = max(1, int(self.pop_size * self.mutant_frac))

            elites = [item[1] for item in population_sorted[:elite_size]]
            non_elites = [item[1] for item in population_sorted[elite_size:]]

            next_population = list(elites)

            while len(next_population) < self.pop_size - mutant_size:
                parent_e = random.choice(elites)
                parent_o = random.choice(non_elites)
                child = [p_e if random.random() < self.bias else p_o for p_e, p_o in zip(parent_e, parent_o)]
                next_population.append(child)

            while len(next_population) < self.pop_size:
                next_population.append([random.random() for _ in range(num_deliveries)])

            population = next_population

        if best_chrom_ever is None: return {}

        return self._format_solution(
            best_chrom_ever,
            deliveries,
            vehicles,
            depot_origin,
            time_matrix,
            p_min,
            t_min,
            ref_ts,
            id_to_idx
        )

'''
p_min (Minutos de Preparação): Esta variável armazena o horário de início da janela de preparação de cada entrega,
convertido para minutos. É um tempo "normalizado", que representa quantos minutos se passaram desde o evento mais cedo do sistema.
Por exemplo, se a preparação de entrega mais antiga começa às 9:00, uma entrega preparada às 9:15 teria um p_min de 15.

t_min (Minutos do Prazo): Similar ao p_min, esta variável armazena o horário de entrega ideal para cada pedido,
também convertido em minutos relativos àquele mesmo evento mais cedo. Isso permite que o algoritmo calcule facilmente o atraso
comparando o tempo de chegada real (em minutos) com este tempo ideal (em minutos).

ref_ts (Timestamp de Referência): Este é o ponto de referência crucial para p_min e t_min. É um timestamp (em segundos)
que representa o início da janela de tempo mais antiga entre todas as entregas. Todos os tempos relativos
em minutos (p_min, t_min, tempos de chegada) são calculados a partir deste ponto de partida único, garantindo que todos os
cálculos de tempo sejam consistentes.

best_chrom_ever (Melhor Cromossomo de Todos): No algoritmo genético, cada solução potencial é codificada como um "cromossomo".
O algoritmo cria e testa milhares desses cromossomos ao longo de muitas gerações. best_chrom_ever é a variável que armazena
o melhor cromossomo encontrado até agora — aquele que resultou na solução mais otimizada (menor penalidade e tempo de viagem) — em
todas as gerações executadas. É a solução "campeã" que é finalmente retornada.
'''