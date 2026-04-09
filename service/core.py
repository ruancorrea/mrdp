from datetime import datetime, timedelta, timezone
from typing import Dict
from service.algorithms.config import Config
from service.factory import get_strategies
from service.utils import Monitor, Evaluate
from service.utils.enums import EventType, VehicleStatus
from service.utils.structures import Event, Vehicle, Delivery
from service.utils.enums import OrderStatus

import numpy as np
import heapq

class Core:
    def __init__(
        self,
        config: Config,
        vehicles: list[Vehicle],
        origin: np.array,
    ) -> None:
        self.config = config
        self.event_queue: list[Event] = []
        self.active_deliveries: Dict[str, Delivery] = {}
        self.simulation_time = None

        self.origin = origin
        self.dispatch_delay_buffer_minutes = timedelta(minutes=config.dispatch_delay_buffer_minutes)
        self.urgent_order_time = timedelta(minutes=config.urgent_order_time)
        self.avg_speed_kmh = config.avg_speed_kmh
        # Lê a política escolhida. Por padrão usa 'DYNAMIC' (ASAP+JIT)
        self.dispatch_policy = getattr(config, 'dispatch_policy', 'DYNAMIC').upper()
        self.slack_usage_ratio=config.slack_usage_ratio
        self.shift_route_limit_ratio = getattr(config, 'shift_route_limit_ratio', 0.5)
        self.vehicles: Dict[int, Vehicle] = {v.id: v for v in vehicles}

        self.clustering_strategy, self.routing_strategy, self.unique_strategy = get_strategies(config)
        self.monitor = Monitor()
        self.evaluator = Evaluate(
            min_block=getattr(config, 'min_block', 5.0),
            penalty_per_block=getattr(config, 'penalty_per_block', 100)
        ) # Helper de cálculo de penalidade

    def add_delivery(self, delivery: Delivery) -> None:
        '''
        Adds a new delivery to the system and schedules its corresponding events.
        The routing logic is NOT triggered here.
        '''
        self.active_deliveries[delivery.id] = delivery
        print(f'[{delivery.timestamp_dt.strftime("%H:%M")}] New Delivery Received: id={delivery.id}')

        self._schedule_event(EventType.ORDER_CREATED, delivery.timestamp_dt, delivery.id)
        self._schedule_event(EventType.ORDER_READY, delivery.preparation_dt, delivery.id)
        self._schedule_event(EventType.PICKUP_DEADLINE, delivery.time_dt, delivery.id) # time_dt is the deadline
        self.monitor.total_deliveries_created += 1

    def _schedule_event(self, event_type: EventType, timestamp: datetime, delivery_id: str) -> Event:
        '''
        Schedules an event in the event queue.
        '''
        event = Event(event_type, timestamp, delivery_id)
        heapq.heappush(self.event_queue, event)
        return event

    def process_events(self) -> list:
        processed_events = []
        while self.event_queue and self.event_queue[0].timestamp <= self.simulation_time:
            event = heapq.heappop(self.event_queue)
            if event.event_type == EventType.VEHICLE_RETURN:
                vehicle = self.vehicles.get(event.delivery_id)
                if vehicle:
                    return_event = self._handle_vehicle_return(event, vehicle)
                    processed_events.append(return_event)
                continue # Pula para o próximo evento

            delivery = self.active_deliveries.get(event.delivery_id)

            if not delivery or delivery.status in [OrderStatus.DELIVERED, OrderStatus.CANCELLED]:
                continue

            handler = getattr(self, f'_handle_{event.event_type.name.lower()}', None)
            if handler:
                processed_event = handler(event, delivery)
                if processed_event:
                    processed_events.append(processed_event)
        return processed_events

    def _handle_order_created(self, event: Event, delivery: Delivery) -> None:
        print(f"[{self.simulation_time.strftime('%H:%M')}] Evento: Pedido {delivery.id} foi criado.")

    def _handle_order_ready(self, event: Event, delivery: Delivery) -> None:
        print(f"[{self.simulation_time.strftime('%H:%M')}] Evento: Pedido {delivery.id} está pronto (às {delivery.preparation_dt.strftime('%H:%M')})!")
        delivery.status = OrderStatus.READY

    def _handle_pickup_deadline(self, event: Event, delivery: Delivery) -> None:
        '''
        Este handler é acionado quando o tempo máximo de espera de um pedido é atingido.
        Ele não altera o estado do pedido, apenas registra o alerta e atualiza o monitor.
        '''
        # A verificação garante que não estamos alertando sobre um pedido que já está a caminho ou foi entregue.
        if delivery.status not in [OrderStatus.DISPATCHED, OrderStatus.DELIVERED]:
            print(f"[{self.simulation_time.strftime('%H:%M')}] ALERTA DE ATRASO: Prazo do Pedido {delivery.id} ({delivery.time_dt.strftime('%H:%M')}) foi ultrapassado!")
            if not hasattr(delivery, 'is_marked_late'):
                self.monitor.total_deliveries_late += 1
                delivery.is_marked_late = True

    def _handle_expected_delivery(self, event: Event, delivery: Delivery) -> dict:
        # Removemos a lógica de liberar o veículo daqui, pois agora é um evento separado.
        print(f"[{self.simulation_time.strftime('%H:%M')}] ENTREGA: Pedido {delivery.id} entregue.")
        delivery.status = OrderStatus.DELIVERED
        delivery.completion_time = self.simulation_time  # Grava o horário da entrega
        self.monitor.total_deliveries_completed += 1

        # Calcula a penalidade real da entrega baseada no momento da conclusão
        lateness_minutes = (self.simulation_time - delivery.time_dt).total_seconds() / 60.0
        delivery.penalty = self.evaluator.compute_penalty_from_arrival(lateness_minutes, 0.0)

        vehicle = self.vehicles.get(delivery.assigned_vehicle_id)
        if vehicle:
            vehicle.completed_deliveries.append({
                "delivery_id": delivery.id,
                "expected_deadline": delivery.time_dt.strftime('%H:%M:%S'),
                "completion_time": delivery.completion_time.strftime('%H:%M:%S'),
                "penalty": delivery.penalty
            })

        return {
            "type": "delivery_completed",
            "data": {
                "delivery_id": delivery.id,
                "completion_time": self.simulation_time.isoformat(),
                "delivery": delivery.to_dict()
            }
        }

    def _handle_vehicle_return(self, event: Event, vehicle: Vehicle) -> dict:
        print(f"[{self.simulation_time.strftime('%H:%M')}] Evento: Veículo {vehicle.id} retornou ao depósito.")
        vehicle.status = VehicleStatus.IDLE
        vehicle.current_route = []
        vehicle.completed_routes += 1
        vehicle.route_end_time = None
        return {
            "type": "driver_returned",
            "data": {
                "vehicle_id": vehicle.id,
                "return_time": self.simulation_time.isoformat()
            }
        }

    def strategies_apply(self, eligible_deliveries: list, available_vehicles: list) -> dict:
        asap_routes_details = None
        #deliveries_by_vehicle = {}

        if self.unique_strategy:
            asap_routes_details = self.unique_strategy.generate_solution(
                eligible_deliveries, available_vehicles, self.origin, self.avg_speed_kmh
            )
            # Para a lógica JIT, precisamos reconstruir o 'deliveries_by_vehicle'
            # a partir dos resultados da rota.
            # (Esta parte pode precisar de ajuste dependendo do retorno do seu algoritmo híbrido)

        elif self.clustering_strategy and self.routing_strategy:
            deliveries_by_vehicle = self.clustering_strategy.cluster(
                eligible_deliveries, available_vehicles, self.origin
            )
            asap_routes_details = self.routing_strategy.generate_routes(
                deliveries_by_vehicle, self.origin, self.avg_speed_kmh
            )
        return asap_routes_details

    def _calculate_delayed_dispatch(self, asap_eval_dt: dict, node_map: dict) -> dict:
        '''
        Calcula um novo horário de despacho atrasado (JIT) com base na folga da rota.

        Args:
            asap_eval_dt (dict): O dicionário de resultados do BRKGA com tempos ASAP.
            node_map (dict): Mapeamento de índice de nó para objeto Delivery.

        Returns:
            dict: Um novo dicionário de resultados com todos os tempos ajustados para o futuro.
        '''
        asap_start_time = asap_eval_dt["start_datetime"]
        route_sequence = asap_eval_dt["sequence"]

        min_slack = timedelta(days=999) # Começa com um valor muito grande

        for i, node_idx in enumerate(route_sequence):
            delivery = node_map[node_idx]
            deadline = delivery.time_dt

            asap_arrival_time = asap_eval_dt["arrivals_map"][node_idx]
            current_slack = deadline - asap_arrival_time

            if current_slack < min_slack:
                min_slack = current_slack

        # A folga máxima que podemos usar é a menor folga da rota, menos nosso buffer de segurança
        usable_delay = (min_slack - self.dispatch_delay_buffer_minutes) * self.slack_usage_ratio
        usable_delay = max(timedelta(seconds=0), usable_delay) # Não podemos ter um atraso negativo

        # Se há um atraso útil, criamos um novo dicionário de resultados com tempos atualizados
        if usable_delay > timedelta(seconds=0):
            print(f"  -> Política JIT: Atrasando a rota em {usable_delay} para aumentar a chance de consolidação.")

            new_eval_dt = asap_eval_dt.copy()
            new_eval_dt["start_datetime"] = asap_start_time + usable_delay
            new_eval_dt["return_depot"] = asap_eval_dt["return_depot"] + usable_delay

            new_arrivals_map = {}
            for node_idx, arrival_dt in asap_eval_dt["arrivals_map"].items():
                new_arrivals_map[node_idx] = arrival_dt + usable_delay
            new_eval_dt["arrivals_map"] = new_arrivals_map

            return new_eval_dt
        print("-> Política JIT: Nenhuma folga útil encontrada. Despachando ASAP.")
        return asap_eval_dt

    def dispatch_policy_use(self, eligible_deliveries: list, current_time: datetime) -> bool:
        if self.dispatch_policy == 'ONLY_ASAP':
            print(f"[{current_time.strftime('%H:%M')}] Política de Despacho: ONLY_ASAP. Despachando imediatamente.")
            return False
            
        if self.dispatch_policy == 'ONLY_JIT':
            print(f"[{current_time.strftime('%H:%M')}] Política de Despacho: ONLY_JIT. Aplicando retenção estratégica máxima.")
            return True

        # Lógica padrão: DYNAMIC (Ambas: JIT com gatilhos ASAP)
        urgent_orders = [
            d for d in eligible_deliveries
            if d.time_dt - current_time < self.urgent_order_time
        ]
        use_jit_policy = True
        if len(eligible_deliveries) > 5 or len(urgent_orders) > 0:
            print(f"[{current_time.strftime('%H:%M')}] Política DYNAMIC: MODO DE URGÊNCIA ATIVADO. Despachando ASAP.")
            use_jit_policy = False

        return use_jit_policy


    def update_state(self, asap_routes_details: dict, use_jit_policy: bool) -> None:
        dispatched_events = []
        if not asap_routes_details:
            return

        for vehicle_id, asap_eval_dt in asap_routes_details.items():
            if not asap_eval_dt: continue

            print(asap_eval_dt)
            vehicle = self.vehicles[vehicle_id]
            node_map = asap_eval_dt["node_map"]
            seq = asap_eval_dt["sequence"]

            jit_eval_dt = asap_eval_dt
            if use_jit_policy:
                jit_eval_dt = self._calculate_delayed_dispatch(asap_eval_dt, node_map)

            # Regra da duração da rota em relação ao limite do turno
            route_time_minutes = jit_eval_dt["total_route_time"]
            dispatch_time = jit_eval_dt["start_datetime"]
            limit_route_time = timedelta(minutes=route_time_minutes * self.shift_route_limit_ratio)
            
            if hasattr(vehicle, 'shift_end') and vehicle.shift_end and (dispatch_time + limit_route_time > vehicle.shift_end):
                print(f"  -> Rota rejeitada: {self.shift_route_limit_ratio * 100:.0f}% da duração da rota ({(route_time_minutes * self.shift_route_limit_ratio):.1f}m) ultrapassa o fim do turno do Veículo {vehicle.id} ({vehicle.shift_end.strftime('%H:%M')}).")
                # Finaliza o turno do motorista precocemente para evitar loop infinito de tentativas e rejeições
                vehicle.shift_end = self.simulation_time
                continue

            # UPDATE STATE
            print(f"  -> Rota JIT definida para Veículo {vehicle.id}: Saída às {jit_eval_dt['start_datetime'].strftime('%H:%M')}, Retorno às {jit_eval_dt['return_depot'].strftime('%H:%M')}")

            self.monitor.total_penalty_incurred += jit_eval_dt["total_penalty"]
            self.monitor.total_route_time_minutes += jit_eval_dt["total_route_time"]

            print(f"  -> Rota definida. Penalidade da rota: {jit_eval_dt['total_penalty']}. Tempo da rota: {jit_eval_dt['total_route_time']:.2f} min.")

            # Atualizar o veículo
            vehicle.status = VehicleStatus.ON_ROUTE
            vehicle.route_end_time = jit_eval_dt['return_depot']
            vehicle.current_route = [node_map[node_idx].id for node_idx in seq]
            
            if getattr(vehicle, 'is_dynamic', False):
                # Encerra o turno do motorista dinâmico assim que ele concluir essa entrega isolada
                vehicle.shift_end = vehicle.route_end_time
                print(f"  -> O motorista dinâmico {vehicle.id} atuará de forma isolada e será dispensado às {vehicle.shift_end.strftime('%H:%M')}.")
                
            dispatched_events.append({
                "vehicle_id": vehicle.id,
                "route": [node_map[node_idx].to_dict() for node_idx in seq],
                "dispatch_time": jit_eval_dt['start_datetime'].isoformat(),
                "return_time": jit_eval_dt['return_depot'].isoformat(),
            })

            return_event = Event(EventType.VEHICLE_RETURN, vehicle.route_end_time, vehicle.id)
            heapq.heappush(self.event_queue, return_event)

            print(f"  -> Veículo {vehicle.id} retornará às {vehicle.route_end_time.strftime('%H:%M')}. Evento agendado.")

            # Atualizar cada pedido na rota
            for node_idx in seq:
                delivery = node_map[node_idx]
                expected_delivery_time = jit_eval_dt['arrivals_map'][node_idx]
                delivery.status = OrderStatus.DISPATCHED
                delivery.assigned_vehicle_id = vehicle.id
                delivery.dispatch_time = jit_eval_dt['start_datetime'] # Grava o horário de despacho
                self._schedule_event(EventType.EXPECTED_DELIVERY, expected_delivery_time, delivery.id)
                print(f"    - Pedido {delivery.id} despachado. Entrega esperada: {expected_delivery_time.strftime('%H:%M')}")

    def routing_decision_logic(self) -> None:
        '''
        Delega a lógica para as estratégias configuradas.
        '''
        current_time = self.simulation_time or datetime.now(timezone.utc)
        eligible_deliveries = [
            d for d in self.active_deliveries.values()
            if d.status in [OrderStatus.READY, OrderStatus.PENDING]
        ]
            
        # Filtra os veículos que estão IDLE e dentro do horário de turno
        available_vehicles = [
            v for v in self.vehicles.values() 
            if v.status == VehicleStatus.IDLE 
            and getattr(v, 'shift_start', current_time) <= current_time < getattr(v, 'shift_end', current_time + timedelta(days=1))
        ]

        if not eligible_deliveries:
            return
            
        if not available_vehicles:
            # Lógica de chamada de Veículo Dinâmico (Crowdsourced)
            dynamic_arrival_minutes = getattr(self.config, 'dynamic_arrival_minutes', 10)
            dynamic_arrival_time = current_time + timedelta(minutes=dynamic_arrival_minutes)

            # Verifica quando o próximo veículo (fixo ou dinâmico já a caminho) ficará disponível
            incoming_vehicles = [
                v.route_end_time for v in self.vehicles.values() 
                if v.status == VehicleStatus.ON_ROUTE and v.route_end_time is not None
                and getattr(v, 'shift_end', current_time + timedelta(days=1)) >= v.route_end_time
            ]
            
            next_arrival_time = min(incoming_vehicles) if incoming_vehicles else None

            # Se não há veículos em rota ou o próximo vai demorar mais que o dinâmico
            if next_arrival_time is None or next_arrival_time > dynamic_arrival_time:
                new_v_id = max(self.vehicles.keys()) + 1 if self.vehicles else 1
                capacity = getattr(self.config, 'vehicle_capacity', 10)
                
                new_v = Vehicle(
                    id=new_v_id,
                    capacity=capacity,
                    status=VehicleStatus.ON_ROUTE, # Fica ON_ROUTE para simular o tempo de deslocamento até o restaurante
                    route_end_time=dynamic_arrival_time,
                    shift_start=current_time,
                    shift_end=current_time + timedelta(days=1), # Turno fictício longo para passar na validação dos 50%
                    is_dynamic=True
                )
                self.vehicles[new_v_id] = new_v
                self._schedule_event(EventType.VEHICLE_RETURN, dynamic_arrival_time, new_v_id)
                print(f"[{current_time.strftime('%H:%M')}] ALERTA: Nenhum veículo disponível. VEÍCULO DINÂMICO ({new_v_id}) chamado, chegará às {dynamic_arrival_time.strftime('%H:%M')}.")
            return

        use_jit_policy = self.dispatch_policy_use(eligible_deliveries, current_time)

        print(f"[{current_time.strftime('%H:%M')}] Lógica de Roteamento: {len(eligible_deliveries)} pedidos prontos e {len(available_vehicles)} veículos disponíveis.")
        asap_routes_details = self.strategies_apply(eligible_deliveries, available_vehicles)
        self.update_state(asap_routes_details, use_jit_policy)

    def run_simulation(self, start_time: datetime, end_time: datetime, incoming_deliveries_schedule: dict) -> tuple[Monitor, dict, dict]:
        self.simulation_time = start_time
        print(f'--- Iniciando Simulação em {start_time} ---')

        while self.simulation_time <= end_time:
            print(f"\n--- Relógio: {self.simulation_time.strftime('%Y-%m-%d %H:%M')} ---")
            if self.simulation_time in incoming_deliveries_schedule:
                for delivery in incoming_deliveries_schedule[self.simulation_time]:
                    self.add_delivery(delivery)
            self.process_events()
            self.routing_decision_logic()

            # --- LOGS DE DEBURAÇÃO ---
            ready_count = sum(1 for d in self.active_deliveries.values() if d.status == OrderStatus.READY)
            dispatched_count = sum(1 for d in self.active_deliveries.values() if d.status == OrderStatus.DISPATCHED)
            idle_vehicles = sum(1 for v in self.vehicles.values() if v.status == VehicleStatus.IDLE)

            print(f"[{self.simulation_time.strftime('%H:%M')}] Status: "
                f"Ready={ready_count}, Dispatched={dispatched_count}, "
                f"Idle Vehicles={idle_vehicles}")
            self.simulation_time += timedelta(minutes=1)

        return self.monitor, self.active_deliveries, self.vehicles