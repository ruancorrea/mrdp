import random
import math
import numpy as np
from copy import deepcopy
from datetime import datetime, timezone, timedelta

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None

from service.utils.distances import get_distance_matrix, get_time_matrix
from service.utils.time import Time
from service.utils.evaluate import Evaluate

class BRKGA:
    """
    Implementa o Biased Random-Key Genetic Algorithm (BRKGA) para resolver o
    problema de roteamento de veículos com janelas de tempo, incluindo otimizações
    de busca local.
    """
    def __init__(self, pop_size=50, elite_frac=0.3, mutant_frac=0.15, bias=0.7,
                 max_gens=70, no_improve_limit=15):
        self.pop_size = pop_size
        self.elite_frac = elite_frac
        self.mutant_frac = mutant_frac
        self.bias = bias
        self.max_gens = max_gens
        self.no_improve_limit = no_improve_limit

        self.time_helper = Time()
        self.evaluate_helper = Evaluate()

    def _decode_keys_to_sequence(self, keys, node_ids):
        pairs = list(zip(keys, node_ids))
        pairs.sort(key=lambda x: x[0])
        return [p[1] for p in pairs]

    def _two_opt(self, seq, evaluate_func):
        best_seq, best_eval = seq, evaluate_func(seq)
        improved = True
        while improved:
            improved = False
            n = len(best_seq)
            for i in range(n - 1):
                for j in range(i + 1, n):
                    new_seq = best_seq[:i] + best_seq[i:j+1][::-1] + best_seq[j+1:]
                    new_eval = evaluate_func(new_seq)
                    if (new_eval.total_penalty < best_eval.total_penalty) or \
                       (new_eval.total_penalty == best_eval.total_penalty and new_eval.total_route_time < best_eval.total_route_time):
                        best_seq, best_eval, improved = new_seq, new_eval, True
                        break
                if improved:
                    break
        return best_seq, best_eval

    def _relocate(self, seq, evaluate_func):
        best_seq, best_eval = seq, evaluate_func(seq)
        n = len(seq)
        improved = True
        while improved:
            improved = False
            for i in range(n):
                node_to_move = best_seq[i]
                temp_seq = best_seq[:i] + best_seq[i+1:]
                for j in range(n):
                    new_seq = temp_seq[:j] + [node_to_move] + temp_seq[j:]
                    new_eval = evaluate_func(new_seq)
                    if (new_eval.total_penalty < best_eval.total_penalty) or \
                       (new_eval.total_penalty == best_eval.total_penalty and new_eval.total_route_time < best_eval.total_route_time):
                        best_seq, best_eval, improved = new_seq, new_eval, True
                        break
                if improved:
                    break
        return best_seq, best_eval

    def _or_opt(self, seq, k, evaluate_func):
        best_seq, best_eval = seq, evaluate_func(seq)
        n = len(seq)
        improved = True
        while improved:
            improved = False
            for block_size in range(1, k + 1):
                for i in range(n - block_size + 1):
                    block = best_seq[i:i + block_size]
                    remainder = best_seq[:i] + best_seq[i+block_size:]
                    for j in range(len(remainder) + 1):
                        new_seq = remainder[:j] + block + remainder[j:]
                        new_eval = evaluate_func(new_seq)
                        if (new_eval.total_penalty < best_eval.total_penalty) or \
                           (new_eval.total_penalty == best_eval.total_penalty and new_eval.total_route_time < best_eval.total_route_time):
                            best_seq, best_eval, improved = new_seq, new_eval, True
                            break
                    if improved: break
                if improved: break
            if improved: break
        return best_seq, best_eval

    def solve(self, node_ids, travel_time, P_dt_map, T_dt_map,
              service_times=None, depot_index=None):
        P_min, T_min, ref_ts = self.time_helper.datetimes_map_to_minutes(P_dt_map, T_dt_map)
        n = len(node_ids)

        def eval_keys(keys):
            seq = self._decode_keys_to_sequence(keys, node_ids)
            return self.evaluate_helper.evaluate_sequence(seq, travel_time, P_min, T_min, service_times, depot_index)

        pop = [[random.random() for _ in range(n)] for _ in range(self.pop_size)]
        best_keys = pop[0]
        best_eval = eval_keys(best_keys)
        no_improve = 0

        for _ in range(self.max_gens):
            pop_fitness = [(k, eval_keys(k)) for k in pop]
            pop_sorted = sorted(pop_fitness, key=lambda x: (x[1].total_penalty, x[1].total_route_time))

            elites = [item[0] for item in pop_sorted[:max(1, int(self.pop_size * self.elite_frac))]]
            non_elites = [item[0] for item in pop_sorted[max(1, int(self.pop_size * self.elite_frac)):]]

            current_best_keys, current_best_eval = pop_sorted[0]
            if (current_best_eval.total_penalty < best_eval.total_penalty) or \
               (current_best_eval.total_penalty == best_eval.total_penalty and current_best_eval.total_route_time < best_eval.total_route_time):
                best_keys, best_eval = current_best_keys, current_best_eval
                no_improve = 0
            else:
                no_improve += 1
            
            if no_improve >= self.no_improve_limit:
                break
            
            next_pop = elites
            while len(next_pop) < self.pop_size - max(1, int(self.pop_size * self.mutant_frac)):
                parent_e = random.choice(elites)
                parent_o = random.choice(non_elites)
                child = [p_e if random.random() < self.bias else p_o for p_e, p_o in zip(parent_e, parent_o)]
                next_pop.append(child)
            
            while len(next_pop) < self.pop_size:
                next_pop.append([random.random() for _ in range(n)])
            pop = next_pop

        best_seq = self._decode_keys_to_sequence(best_keys, node_ids)
        
        def eval_seq_wrapper(s):
            return self.evaluate_helper.evaluate_sequence(s, travel_time, P_min, T_min, service_times, depot_index)

        seq, ev = self._two_opt(best_seq, eval_seq_wrapper)
        seq, ev = self._or_opt(seq, 3, eval_seq_wrapper)
        seq, ev = self._relocate(seq, eval_seq_wrapper)
        
        arrival_datetimes = [self.time_helper.minutes_to_datetime(t, ref_ts) for t in ev.arrival_times]
        start_datetime = self.time_helper.minutes_to_datetime(ev.start_time, ref_ts)
        
        ev_with_dt = deepcopy(ev.__dict__)
        ev_with_dt.update({
            "arrival_datetimes": arrival_datetimes,
            "start_datetime": start_datetime,
            "arrivals_map": {node: arrival_datetimes[i] for i, node in enumerate(seq)},
            "penalties_map": {node: ev.penalties[i] for i, node in enumerate(seq)},
            "ref_timestamp_seconds": ref_ts,
            "return_depot": start_datetime + timedelta(minutes=ev.total_route_time),
        })
        return seq, ev, ev_with_dt

def apply(data, origin, average_speed_kmh=30):
    points = np.array([origin.tolist()] + [[b.point.lat, b.point.lng] for b in data])
    preparations = {i: b.preparation_dt for i, b in enumerate(data)}
    times = {i: b.time_dt for i, b in enumerate(data)}

    distance_matrix = get_distance_matrix(points)
    travel_time = get_time_matrix(distance_matrix, average_speed_kmh)

    node_ids = list(range(len(data)))
    service_times = {i: 2 for i in node_ids}

    brkga_solver = BRKGA(pop_size=60, max_gens=200)
    seq, ev_min, ev_dt = brkga_solver.solve(node_ids, travel_time, preparations, times,
                                           service_times=service_times,
                                           depot_index=0) # Depot is the first element in points

    print("Sequence order (visit order):", seq)
    print("Start datetime (route):", ev_dt.start_datetime)
    print("Total penalty:", ev_min.total_penalty)
    print("Total route time (min):", ev_min.total_route_time)
    print("Expected delivery times per node (datetime):")
    for node in seq:
        print(f"  Node {node}: arrival={ev_dt['arrivals_map'][node]}, penalty={ev_dt['penalties_map'][node]}")

if __name__ == "__main__":
    DEFAULT_TZ_NAME = "America/Sao_Paulo"
    if ZoneInfo:
        tz = ZoneInfo(DEFAULT_TZ_NAME)
    else:
        tz = timezone.utc

    base_dt = datetime(2025, 10, 13, 8, 0, tzinfo=tz)
    P_dt = {i: base_dt + timedelta(minutes=random.randint(5, 15)) for i in range(5)}
    T_dt = {i: P_dt[i] + timedelta(minutes=random.randint(20, 50)) for i in range(5)}
    
    n_orders = 5
    depot_index = 5
    total_nodes = n_orders + 1
    rng = np.random.RandomState(2)
    mat = rng.randint(2, 15, size=(total_nodes, total_nodes)).astype(float)
    np.fill_diagonal(mat, 0)
    travel_time = mat.tolist()
    
    node_ids = list(range(n_orders))
    service_times = {i: 2 for i in node_ids}
    
    brkga_solver = BRKGA(pop_size=60, max_gens=200)
    seq, ev_min, ev_dt = brkga_solver.solve(node_ids, travel_time, P_dt, T_dt,
                                           service_times=service_times,
                                           depot_index=depot_index)

    print("Sequence order (visit order):", seq)
    print("Start datetime (route):", ev_dt.start_datetime)
    print("Total penalty:", ev_min.total_penalty)
    print("Total route time (min):", ev_min.total_route_time)
    print("Return to depot (arrival):", ev_dt.return_depot)
    print("Expected delivery times per node (datetime):")
    for node in seq:
        print(f"  Node {node}: arrival={ev_dt['arrivals_map'][node]}, penalty={ev_dt['penalties_map'][node]}")