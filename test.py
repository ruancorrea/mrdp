import csv
import os
import time
from datetime import timedelta
from collections import defaultdict
import random
from copy import deepcopy

from service.core import Core
import numpy as np
from service.utils import load_instances as Instances
from service.utils.structures import Vehicle
try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None

from service.algorithms.config import Config, ClusteringAlgorithm, RoutingAlgorithm, UniqueAlgorithm
from dataclasses import replace
from service.utils.output import SimulationOutput

avg_speed_kmh=80

SIMULATION_TZ_NAME = "America/Sao_Paulo"
if ZoneInfo:
    SIMULATION_TZ = ZoneInfo(SIMULATION_TZ_NAME)
else:
    from datetime import timezone
    SIMULATION_TZ = timezone.utc
    print("Aviso: zoneinfo não encontrado. Usando UTC como fuso horário.")

def run_test(instance_number, strategy_name, vehicles, config, n_drivers):
    # FIXAR SEMENTE PARA REPRODUTIBILIDADE
    random.seed(42)
    np.random.seed(42)
    config.avg_speed_kmh = avg_speed_kmh
    #path_eval = './data/dev'
    path_eval = './willy_instances'
    data_base: str = '01/01/2025'
    hours: int = 9
    minutes: int = 0
    
    instances_data = Instances.get_instances(path_eval, number_instance=instance_number)
    if not instances_data:
        return None

    all_deliveries_by_time = Instances.process_instances(instances_data[:1], data_base, hours, minutes, tzinfo=SIMULATION_TZ)
    origin = np.array([-35.739118, -9.618276])

    simulation_start_time = Instances.get_initial_time(data_base, hours, minutes, tzinfo=SIMULATION_TZ)
    simulation_end_time = simulation_start_time + timedelta(hours=16)

    # Turnos já vêm configurados da instanciação global.
    # Removida a sobreescrita aqui para preservar os horários específicos de cada motorista.

    incoming_deliveries_schedule = defaultdict(list)
    for delivery in all_deliveries_by_time[0]:
       if delivery.timestamp_dt > simulation_end_time:
            break
       incoming_deliveries_schedule[delivery.timestamp_dt].append(delivery)
    
    system = Core(
        config=config,
        vehicles=vehicles,
        origin=origin,
    )
    
    start_time = time.time()
    final_monitor_results, final_deliveries, final_vehicles = system.run_simulation(simulation_start_time, simulation_end_time + timedelta(hours=5), incoming_deliveries_schedule)
    end_time = time.time()
    name_reports = f"{avg_speed_kmh}_1"
    # --- Geração de Report Detalhado ---
    report_dir = os.path.join("tests", "reports", name_reports, strategy_name)
    os.makedirs(report_dir, exist_ok=True)
    
    # Nome do arquivo inclui o número de veículos para evitar sobrescrita no loop de veículos
    report_file = os.path.join(report_dir, f"instance_{instance_number}_v{len(vehicles)}.csv")
    
    # Exporta o resumo JSON com os veículos fixos e dinâmicos
    output_processor = SimulationOutput(
        monitor=final_monitor_results,
        deliveries=final_deliveries,
        vehicles=final_vehicles
    )
    json_report_file = os.path.join(report_dir, f"vehicle_summary_instance_{instance_number}_v{len(vehicles)}.json")
    output_processor.export_vehicle_summary_json(json_report_file)

    with open(report_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["delivery_id", "lat", "lng", "dispatch_time", "completion_time", "deadline_time", "vehicle_id"])
        
        for d in final_deliveries.values():
            writer.writerow([
                d.id,
                d.point.lat,
                d.point.lng,
                getattr(d, 'dispatch_time', ''),
                getattr(d, 'completion_time', ''),
                d.time_dt,
                d.assigned_vehicle_id
            ])
    
    execution_time = end_time - start_time
    avg_penalty = final_monitor_results.get_average_penalty_per_delivery()
    
    print(f"Instance: {instance_number}, Strategy: {strategy_name}, Avg Penalty: {avg_penalty:.2f}, Execution Time: {execution_time:.2f}s")
    
    deliveries_per_vehicle = defaultdict(int)
    for d in final_deliveries.values():
        if d.assigned_vehicle_id is not None:
            v_assigned = final_vehicles[d.assigned_vehicle_id]
            real_id = v_assigned.label if v_assigned.label else str(v_assigned.id)
            deliveries_per_vehicle[str(real_id)] += 1

    # Extrai o uso da capacidade flexível (motoristas dinâmicos)
    dynamic_vehicles_count = sum(1 for v in final_vehicles.values() if getattr(v, 'is_dynamic', False))
    dynamic_deliveries_count = sum(len(v.completed_deliveries) for v in final_vehicles.values() if getattr(v, 'is_dynamic', False))

    return {
        'instance': instance_number,
        'strategy': strategy_name,
        'avg_penalty': f"{avg_penalty:.2f}",
        'execution_time': f"{execution_time:.4f}",
        'n_vehicles': f'{n_drivers}',
        'n_deliveries': final_monitor_results.total_deliveries_completed,
        'dynamic_vehicles_called': dynamic_vehicles_count,
        'dynamic_deliveries': dynamic_deliveries_count,
        'deliveries_per_vehicle': dict(deliveries_per_vehicle)
    }

if __name__ == "__main__":
    os.makedirs("tests", exist_ok=True)

    # Carrega as configurações globais do arquivo JSON
    base_config = Config.load_config("config.json")

    strategies_to_test = {
        "greedy_clustering+brkga": replace(base_config,
            clustering_algo=ClusteringAlgorithm.GREEDY,
            routing_algo=RoutingAlgorithm.BRKGA,
            unique_algo=None # Anula o unique base para forçar o uso da abordagem sequencial
        ),
        "ckmeans+brkga": replace(base_config,
            clustering_algo=ClusteringAlgorithm.CKMEANS,
            routing_algo=RoutingAlgorithm.BRKGA,
            unique_algo=None # Anula o unique base para forçar o uso da abordagem sequencial
        ),
        #"brkga_hybrid": replace(base_config, unique_algo=UniqueAlgorithm.BRKGA_UNIQUE, clustering_algo=None, routing_algo=None),
        #"greedy_hybrid": replace(base_config, unique_algo=UniqueAlgorithm.GREEDY_INSERTION, clustering_algo=None, routing_algo=None),
        #"manual_hybrid": replace(base_config, unique_algo=UniqueAlgorithm.MANUAL, clustering_algo=None, routing_algo=None),
    }

    # Definimos a data base para construir os turnos exatos
    base_dt = Instances.get_initial_time('01/01/2025', 9, 0, tzinfo=SIMULATION_TZ)
    
    # Lê a capacidade dos veículos a partir da configuração global (fallback para 10 caso não exista)
    v_capacity = getattr(base_config, 'vehicle_capacity', 10)
    
    # Criamos os veículos em uma lista de 1 dimensão, usando a label para agrupar os turnos
    vehicles = [
        Vehicle(id=1, label="1", capacity=v_capacity, shift_start=base_dt.replace(hour=11, minute=0), shift_end=base_dt.replace(hour=16, minute=0)),
        Vehicle(id=2, label="1", capacity=v_capacity, shift_start=base_dt.replace(hour=18, minute=30), shift_end=base_dt.replace(hour=22, minute=30)),
        Vehicle(id=3, label="2", capacity=v_capacity, shift_start=base_dt.replace(hour=11, minute=0), shift_end=base_dt.replace(hour=19, minute=30)),
        Vehicle(id=4, label="3", capacity=v_capacity, shift_start=base_dt.replace(hour=12, minute=0), shift_end=base_dt.replace(hour=17, minute=0)),
        Vehicle(id=5, label="3", capacity=v_capacity, shift_start=base_dt.replace(hour=20, minute=0), shift_end=base_dt.replace(hour=22, minute=30))
    ]

    all_results = []
    for j in range(len(vehicles)):
        for i in range(90):  # For each instance from 0 to 6
            for strategy_name, config in strategies_to_test.items():
                v_copy = [deepcopy(v) for v in vehicles[:j+1]]
                n_drivers = j + 1
                result = run_test(i, strategy_name, v_copy, config, n_drivers)
                
                if not result:
                    continue

                # Process deliveries_per_vehicle
                deliveries_count = result.pop('deliveries_per_vehicle', {})
                
                current_driver_ids = set()
                for v in v_copy:
                    real_id = str(v.label) if v.label else str(v.id)
                    current_driver_ids.add(real_id)

                for k in range(1, 6):
                    vehicle_key = f'vehicle_{k}'
                    if str(k) in current_driver_ids:
                        result[vehicle_key] = deliveries_count.get(str(k), 0)
                    else:
                        result[vehicle_key] = -1

                all_results.append(result)

    output_csv = os.path.join("tests", f'results_test_datadeval_{avg_speed_kmh}khm_1.csv')
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['instance', 'strategy', 'avg_penalty', 'execution_time', 'n_vehicles', 'n_deliveries', 
                      'dynamic_vehicles_called', 'dynamic_deliveries', 'vehicle_1', 'vehicle_2', 'vehicle_3', 'vehicle_4', 'vehicle_5']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nTestes concluídos. Resultados salvos em {output_csv}")
