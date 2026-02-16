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

avg_speed_kmh=80

SIMULATION_TZ_NAME = "America/Sao_Paulo"
if ZoneInfo:
    SIMULATION_TZ = ZoneInfo(SIMULATION_TZ_NAME)
else:
    from datetime import timezone
    SIMULATION_TZ = timezone.utc
    print("Aviso: zoneinfo não encontrado. Usando UTC como fuso horário.")

def run_test(instance_number, strategy_name, vehicles, config):
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
            deliveries_per_vehicle[d.assigned_vehicle_id] += 1

    return {
        'instance': instance_number,
        'strategy': strategy_name,
        'avg_penalty': f"{avg_penalty:.2f}",
        'execution_time': f"{execution_time:.4f}",
        'n_vehicles': f'{len(vehicles)}',
        'n_deliveries': final_monitor_results.total_deliveries_completed,
        'deliveries_per_vehicle': dict(deliveries_per_vehicle)
    }

if __name__ == "__main__":
    os.makedirs("tests", exist_ok=True)

    strategies_to_test = {
        "greedy_clustering+brkga": Config(
            clustering_algo=ClusteringAlgorithm.GREEDY,
            routing_algo=RoutingAlgorithm.BRKGA
        ),
        "ckmeans+brkga": Config(
            clustering_algo=ClusteringAlgorithm.CKMEANS,
            routing_algo=RoutingAlgorithm.BRKGA
        ),
        #"brkga_hybrid": Config(unique_algo=UniqueAlgorithm.BRKGA_UNIQUE),
        #"greedy_hybrid": Config(unique_algo=UniqueAlgorithm.GREEDY_INSERTION),
        #"manual_hybrid": Config(unique_algo=UniqueAlgorithm.MANUAL),
    }

    vehicles = [
        Vehicle(id=1, capacity=10),
        Vehicle(id=2, capacity=10),
        Vehicle(id=3, capacity=10),
        Vehicle(id=4, capacity=10),
        Vehicle(id=5, capacity=10),
    ]

    all_results = []
    for j in range(len(vehicles)):
        for i in range(90):  # For each instance from 0 to 6
            for strategy_name, config in strategies_to_test.items():
                v_copy = [deepcopy(v) for v in vehicles[:j+1]]
                result = run_test(i, strategy_name, v_copy, config)
                
                if not result:
                    continue

                # Process deliveries_per_vehicle
                deliveries_count = result.pop('deliveries_per_vehicle', {})
                current_vehicle_ids = [v.id for v in v_copy]

                for k in range(1, 6):
                    vehicle_key = f'vehicle_{k}'
                    if k in current_vehicle_ids:
                        result[vehicle_key] = deliveries_count.get(k, 0)
                    else:
                        result[vehicle_key] = -1

                all_results.append(result)

    output_csv = os.path.join("tests", f'results_test_datadeval_{avg_speed_kmh}khm_1.csv')
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['instance', 'strategy', 'avg_penalty', 'execution_time', 'n_vehicles', 'n_deliveries', 
                      'vehicle_1', 'vehicle_2', 'vehicle_3', 'vehicle_4', 'vehicle_5']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nTestes concluídos. Resultados salvos em {output_csv}")
