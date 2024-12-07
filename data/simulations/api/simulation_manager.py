from concurrent import futures
import grpc
import simulation_pb2
import simulation_pb2_grpc

class SimulationManager(simulation_pb2_grpc.SimulationManagerServicer):
    def StartSimulation(self, request, context):
        # Logic for starting a simulation task
        return simulation_pb2.SimulationResponse(message="Simulation started successfully!")

    def StopSimulation(self, request, context):
        # Logic for stopping a simulation task
        return simulation_pb2.SimulationResponse(message="Simulation stopped successfully!")

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    simulation_pb2_grpc.add_SimulationManagerServicer_to_server(SimulationManager(), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    print("Simulation Manager is running on port 50051")
    server.wait_for_termination()

if __name__ == "__main__":
    serve()
