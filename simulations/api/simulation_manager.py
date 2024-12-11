from threading import Lock

class SimulationManager(simulation_pb2_grpc.SimulationManagerServicer):
    def __init__(self):
        self.simulations = {}
        self.lock = Lock()

    def StartSimulation(self, request, context):
        try:
            with self.lock:
                simulation_id = request.simulation_id
                if simulation_id in self.simulations:
                    return simulation_pb2.SimulationResponse(
                        message=f"Simulation {simulation_id} already running."
                    )
                self.simulations[simulation_id] = {"status": "running"}
            return simulation_pb2.SimulationResponse(
                message=f"Simulation {simulation_id} started successfully!"
            )
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return simulation_pb2.SimulationResponse(message="Error starting simulation.")

    def StopSimulation(self, request, context):
        try:
            with self.lock:
                simulation_id = request.simulation_id
                if simulation_id not in self.simulations:
                    return simulation_pb2.SimulationResponse(
                        message=f"Simulation {simulation_id} not found."
                    )
                del self.simulations[simulation_id]
            return simulation_pb2.SimulationResponse(
                message=f"Simulation {simulation_id} stopped successfully!"
            )
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return simulation_pb2.SimulationResponse(message="Error stopping simulation.")