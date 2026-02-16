from dataclasses import dataclass
import math

@dataclass
class Solution:
    arrival_times: list
    penalties: list
    total_penalty: int
    total_route_time: float
    start_time: float

class Evaluate:
    def __init__(self, min_block: float=5.0, penalty_per_block: int =100) -> None:
        self.min_block = min_block
        self.penalty_per_block = penalty_per_block

    def compute_penalty_from_arrival(self, arrival: float, T: float) -> int:
        '''
        Calcula a penalidade da chegada ao depot.
        '''
        lateness = max(0.0, arrival - T)
        if lateness <= 0:
            return 0
        blocks = math.ceil(lateness / self.min_block)
        return int(blocks * self.penalty_per_block)

    def evaluate_sequence(
        self,
        seq: list,
        travel_time: list,
        P_min: list,
        T_min: list,
        service_times: dict=None,
        depot_index=None
    ) -> Solution:
        '''
        Avalia uma sequência de entregas (seq) dada uma matriz de tempos de viagem (travel_time),
        tempos de preparação mínimos (P_min) e tempos de entrega desejados (T_min).
        '''
        if service_times is None:
            service_times = {i: 0.0 for i in seq}
        arrival_times = [] ; penalties = []
        # start = max P of all orders in seq
        start_time = max(P_min[i] for i in seq)
        time = start_time
        # leave depot -> first
        if depot_index is not None:
            time += travel_time[depot_index][seq[0]]
        for idx, node in enumerate(seq):
            if idx > 0:
                prev = seq[idx-1]
                time += travel_time[prev][node]
            arrival = time
            arrival_times.append(arrival)
            pen = self.compute_penalty_from_arrival(arrival, T_min[node])
            penalties.append(pen)
            time += service_times.get(node, 0.0)
        # after last, return to depot
        if depot_index is not None:
            time += travel_time[seq[-1]][depot_index]
        return Solution(
            arrival_times=arrival_times,
            penalties=penalties,
            total_penalty=sum(penalties),
            total_route_time=time - start_time,
            start_time=start_time)
