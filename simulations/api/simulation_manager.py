"""
Simulation Manager for ACM Project

This module handles the initiation, termination, and monitoring of simulation tasks.
Designed for seamless integration with Unreal Engine and the broader ACM architecture.
"""

import logging
from concurrent import futures
import grpc
import simulation_pb2
import simulation_pb2_grpc


class SimulationManager(simulation_pb2_grpc.SimulationManagerServicer):
    def __init__(self):
        """
        Initialize the Simulation Manager.
        Manages active simulations and provides status updates.
        """
        self.active_simulations = {}
        logging.basicConfig(level=logging.INFO)

    def StartSimulation(self, request, context):
        """
        Start a new simulation.
        Args:
            request: gRPC request containing simulation parameters.
            context: gRPC context for error handling.
        Returns:
            SimulationResponse: Status message.
        """
        try:
            simulation_id = request.simulation_id
            if simulation_id in self.active_simulations:
                context.set_code(grpc.StatusCode.ALREADY_EXISTS)
                context.set_details("Simulation already running.")
                return simulation_pb2.SimulationResponse(
                    message=f"Simulation {simulation_id} already running."
                )

            # Placeholder for actual simulation logic
            self.active_simulations[simulation_id] = "running"
            logging.info(f"Started simulation {simulation_id}")
            return simulation_pb2.SimulationResponse(
                message=f"Simulation {simulation_id} started successfully."
            )
        except Exception as e:
            logging.error(f"Error starting simulation: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return simulation_pb2.SimulationResponse(message="Failed to start simulation.")

    def StopSimulation(self, request, context):
        """
        Stop a running simulation.
        Args:
            request: gRPC request with simulation ID.
            context: gRPC context for error handling.
        Returns:
            SimulationResponse: Status message.
        """
        try:
            simulation_id = request.simulation_id
            if simulation_id not in self.active_simulations:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details("Simulation not found.")
                return simulation_pb2.SimulationResponse(
                    message=f"Simulation {simulation_id} not found."
                )

            # Placeholder for actual stop logic
            del self.active_simulations[simulation_id]
            logging.info(f"Stopped simulation {simulation_id}")
            return simulation_pb2.SimulationResponse(
                message=f"Simulation {simulation_id} stopped successfully."
            )
        except Exception as e:
            logging.error(f"Error stopping simulation: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return simulation_pb2.SimulationResponse(message="Failed to stop simulation.")


def serve():
    """
    Start the Simulation Manager gRPC server.
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    simulation_pb2_grpc.add_SimulationManagerServicer_to_server(SimulationManager(), server)
    server.add_insecure_port("[::]:50051")
    logging.info("Simulation Manager server started on port 50051.")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
