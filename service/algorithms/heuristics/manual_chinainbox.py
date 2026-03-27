from typing import List, Dict
import numpy as np

from service.utils.structures import Delivery, Vehicle, Point

class ManualChinaInbox:
    def __init__(self, neighborhood_radius_km: float = 1.0):
        self.neighborhood_radius_km = neighborhood_radius_km

    def assign(
        self,
        deliveries: List[Delivery],
        vehicles: List[Vehicle],
        depot_origin: Point
    ) -> Dict[int, List[Delivery]]:
        '''
        Agrupa entregas para veículos com base na ordem de chegada e proximidade de bairro.
        
        Retorna um dicionário mapeando ID do veículo para uma lista de entregas.
        '''

        assignments = {v.id: [] for v in vehicles}
        assigned_deliveries = set()

        # Sort deliveries by arrival time (timestamp_dt)
        sorted_deliveries = sorted(deliveries, key=lambda d: d.timestamp_dt)

        for vehicle in vehicles:
            current_route: List[Delivery] = []
            
            for delivery in sorted_deliveries:
                if delivery.id in assigned_deliveries:
                    continue

                # If the vehicle has no deliveries yet, add the delivery
                if not current_route:
                    current_route.append(delivery)
                    assigned_deliveries.add(delivery.id)
                    continue

                # Check neighborhood proximity
                last_delivery_point = current_route[-1].point
                distance = self._calculate_distance(last_delivery_point, delivery.point)
                
                if distance <= self.neighborhood_radius_km:
                    current_route.append(delivery)
                    assigned_deliveries.add(delivery.id)
                elif len(current_route) == 1:
                    # If only one delivery, dispatch the vehicle
                    assignments[vehicle.id] = current_route
                    break  # Move to the next vehicle
                else:
                    # Different neighborhood, assign current route to vehicle and break
                    assignments[vehicle.id] = current_route
                    break
            else:
                assignments[vehicle.id] = current_route  # Assign any remaining deliveries to the vehicle

        return assignments

    def _calculate_distance(self, point1: Point, point2: Point) -> float:
        '''
        Calculates Euclidean distance between two points.
        '''
        return np.sqrt(
            (point1.lat - point2.lat)**2 + (point1.lng - point2.lng)**2
        )
