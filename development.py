from datetime import timedelta
from collections import defaultdict
import json
import os

from service import Core
from service.utils.monitor import Monitor
from service.utils.structures import Vehicle

import numpy as np
from service.utils.load_instances import get_instances, process_instances, get_delivery_for_time, get_initial_time
from service.algorithms.config import Config, ClusteringAlgorithm, RoutingAlgorithm, UniqueAlgorithm

from service.utils.output import SimulationOutput

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None

# Defina o fuso horário que será usado em toda a simulação
# Deve ser o mesmo usado no seu módulo de rotas
SIMULATION_TZ_NAME = "America/Sao_Paulo"
if ZoneInfo:
    SIMULATION_TZ = ZoneInfo(SIMULATION_TZ_NAME)
else:
    # Fallback para UTC se zoneinfo não estiver disponível
    from datetime import timezone
    SIMULATION_TZ = timezone.utc
    print("Aviso: zoneinfo não encontrado. Usando UTC como fuso horário.")

if __name__ == "__main__":
    path_eval = './data/dev'
    path_train = './data/train'
    data_base: str = '01/01/2025'
    hours: int = 18
    minutes: int = 0
    number_instance = 2
    instances = get_instances(path_eval, number_instance=number_instance)
    all_deliveries_by_time = process_instances(instances[:1], data_base, hours, minutes, tzinfo=SIMULATION_TZ)
    origin = np.array([-35.739118, -9.618276])

    simulation_start_time = get_initial_time(data_base, hours, minutes, tzinfo=SIMULATION_TZ)
    simulation_end_time = simulation_start_time + timedelta(hours=9)
    simulation_time = simulation_start_time

    print(f"Iniciando simulação de {simulation_start_time} até {simulation_end_time}")
    delivery_for_time = get_delivery_for_time(all_deliveries_by_time[0])

    vehicles_list = [
        Vehicle(id=1, capacity=90, shift_start=simulation_start_time, shift_end=simulation_end_time),
        Vehicle(id=2, capacity=90, shift_start=simulation_start_time, shift_end=simulation_end_time),
    ]

    incoming_deliveries_schedule = defaultdict(list)

    for delivery in all_deliveries_by_time[0]:
       if delivery.timestamp_dt > simulation_end_time:
            break
       incoming_deliveries_schedule[delivery.timestamp_dt].append(delivery)

    # --- Carregar Configuração ---
    config = Config.load_config("config.json")

    # --- Execução ---
    system = Core(
        config=config,
        vehicles=vehicles_list,
        origin=origin,
    )

    final_monitor, final_deliveries, final_vehicles = system.run_simulation(
        simulation_start_time,
        simulation_end_time + timedelta(hours=12),
        incoming_deliveries_schedule
    )

    # --- Processamento e Exibição dos Resultados ---
    print("\nCONFIGURAÇÃO DA SIMULAÇÃO:", config)

    output_processor = SimulationOutput(
        monitor=final_monitor,
        deliveries=final_deliveries,
        vehicles=final_vehicles
    )

    output_processor.display_final_summary()
    output_processor.display_delivery_lifecycle()
    output_processor.display_vehicle_summaries()
    output_processor.export_vehicle_summary_json("vehicle_summary.json")
