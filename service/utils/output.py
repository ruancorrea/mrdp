from typing import Dict
from service.utils.monitor import Monitor
from service.utils.structures import Delivery, Vehicle
from service.utils.enums import OrderStatus

class SimulationOutput:
    '''
    Classe dedicada a processar e exibir todos os resultados de uma simulação.
    '''
    def __init__(self, monitor: Monitor, deliveries: Dict[str, Delivery], vehicles: Dict[int, Vehicle]):
        self.monitor = monitor
        self.deliveries = deliveries
        self.vehicles = vehicles

    def display_final_summary(self):
        '''
        Exibe o relatório final consolidado da simulação, similar ao antigo Monitor.
        '''
        print("\n================== RELATÓRIO FINAL DA SIMULAÇÃO ==================")
        self.monitor.display()

        avg_penalty = self.monitor.get_average_penalty_per_delivery()
        print(f"\nAnálise: A penalidade média de {avg_penalty:.2f} por entrega indica o nível de serviço.")
        if avg_penalty > 50:
            print("Sugestão: A penalidade média é alta. Considere adicionar mais veículos ou otimizar os parâmetros do algoritmo.")
        else:
            print("Resultado: O nível de serviço parece aceitável.")
        print("==================================================================")


    def display_delivery_lifecycle(self):
        '''
        Exibe o ciclo de vida de cada entrega, mostrando os tempos e o motorista.
        '''
        print("\n================== CICLO DE VIDA DAS ENTREGAS ==================")
        
        sorted_deliveries = sorted(self.deliveries.values(), key=lambda d: d.timestamp_dt)

        for d in sorted_deliveries:
            if d.status == OrderStatus.DELIVERED:
                motorista = f"Veículo {d.assigned_vehicle_id}" if d.assigned_vehicle_id else "N/A"
                tempo_saida = d.dispatch_time.strftime('%H:%M:%S') if d.dispatch_time else "N/A"
                tempo_entrega = d.completion_time.strftime('%H:%M:%S') if d.completion_time else "N/A"

                print(f"- Pedido {d.id}:")
                print(f"  - Status: {d.status.value}")
                print(f"  - Despachado: {tempo_saida}")
                print(f"  - Entregue: {tempo_entrega}")
                print(f"  - Por: {motorista}")

            elif d.status in [OrderStatus.PENDING, OrderStatus.READY]:
                 print(f"- Pedido {d.id}:")
                 print(f"  - Status: {d.status.value} (Não foi entregue ao final da simulação)")

        print("==================================================================")
