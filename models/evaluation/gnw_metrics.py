import torch
import time

class GNWMetrics:
    def __init__(self, num_modules: int, ignition_threshold_delta: float = 0.5, min_modules_for_ignition: int = 2, logger=None):
        """
        Initializes the GNWMetrics calculator.

        Args:
            num_modules (int): The number of distinct modules or areas in the GNW being monitored.
            ignition_threshold_delta (float): The minimum change (delta) in activation for a module
                                             to be considered part of an ignition event.
            min_modules_for_ignition (int): The minimum number of modules that must surpass the
                                           threshold simultaneously for an ignition event to be declared.
            logger (MetricsLogger, optional): An instance of MetricsLogger.
        """
        self.num_modules = num_modules
        self.ignition_threshold_delta = ignition_threshold_delta
        self.min_modules_for_ignition = max(1, min_modules_for_ignition) # Ensure at least 1
        self.logger = logger
        
        self.previous_activations = None # torch.Tensor of shape (num_modules,)
        self.event_timestamps = {} # For Global Availability Latency: {event_id: timestamp}
        
        print(f"GNWMetrics module initialized. Num modules: {num_modules}, Ignition Threshold Delta: {self.ignition_threshold_delta}, Min Modules for Ignition: {self.min_modules_for_ignition}")

    def update_activations(self, current_activations: torch.Tensor, step: int) -> int:
        """
        Updates module activations and checks for GNW ignition events.

        Args:
            current_activations (torch.Tensor): A tensor representing the current activation
                                                levels of the GNW modules.
                                                Expected shape: (num_modules,).
            step (int): The current simulation or training step.

        Returns:
            int: 1 if an ignition event occurred, 0 otherwise.
        """
        if not isinstance(current_activations, torch.Tensor) or current_activations.shape != (self.num_modules,):
            # Log this with the system logger if available, or print
            print(f"Error: GNWMetrics current_activations has incorrect shape or type. Expected ({self.num_modules},), got {current_activations.shape if isinstance(current_activations, torch.Tensor) else type(current_activations)}")
            if self.logger:
                 self.logger.log_scalar_data("gnw_error", step, 1, {"error_type": "invalid_activation_input"})
            return 0

        ignition_event_occurred = 0
        if self.previous_activations is not None:
            delta_activations = current_activations - self.previous_activations
            
            # Check for ignition
            # Modules that surpassed the positive delta threshold
            ignited_modules_mask = delta_activations > self.ignition_threshold_delta
            num_ignited_modules = torch.sum(ignited_modules_mask).item()

            if num_ignited_modules >= self.min_modules_for_ignition:
                ignition_event_occurred = 1
                print(f"Step {step}: GNW Ignition Event Detected! {num_ignited_modules} modules ignited.")
                if self.logger:
                    self.logger.log_scalar_data(
                        metric_name="gnw_ignition_event",
                        step=step,
                        scalar_value=1,
                        metadata={
                            "num_ignited_modules": num_ignited_modules,
                            "threshold_delta": self.ignition_threshold_delta,
                            "min_modules_required": self.min_modules_for_ignition
                        }
                    )
                    # Log which modules ignited if needed
                    # ignited_indices = torch.where(ignited_modules_mask)[0].tolist()
                    # self.logger.log_scalar_data("gnw_ignited_module_indices", step, ignited_indices) 
            else:
                 if self.logger: # Log even if no ignition, to see the count
                    self.logger.log_scalar_data("gnw_ignition_event", step, 0, {"num_ignited_modules": num_ignited_modules})


        self.previous_activations = current_activations.clone()
        return ignition_event_occurred

    def log_sensory_event_start(self, event_id: str):
        """
        Logs the timestamp when a sensory event (or any event of interest) starts.
        Used for calculating Global Availability Latency.

        Args:
            event_id (str): A unique identifier for the event.
        """
        self.event_timestamps[event_id] = time.perf_counter()
        # print(f"GNW: Sensory event '{event_id}' started at {self.event_timestamps[event_id]}")

    def log_event_reuse(self, event_id: str, module_name: str, step: int):
        """
        Logs when a module reuses or processes information related to a previously logged event.
        Calculates and logs the Global Availability Latency.

        Args:
            event_id (str): The unique identifier of the event being reused.
            module_name (str): Name of the module reusing the event information.
            step (int): The current simulation or training step.
        """
        if event_id in self.event_timestamps:
            current_time = time.perf_counter()
            latency = current_time - self.event_timestamps[event_id]
            
            print(f"Step {step}: Event '{event_id}' reused by module '{module_name}'. Latency: {latency:.4f}s")
            if self.logger:
                self.logger.log_scalar_data(
                    metric_name="gnw_global_availability_latency",
                    step=step,
                    scalar_value=latency,
                    metadata={
                        "event_id": event_id,
                        "reusing_module": module_name
                    }
                )
            # Optionally remove the event to prevent re-logging for the same first reuse,
            # or allow multiple reuse logs if that's desired.
            # del self.event_timestamps[event_id] 
        else:
            # print(f"Warning: Event '{event_id}' not found for latency calculation by module '{module_name}'.")
            if self.logger:
                self.logger.log_scalar_data("gnw_error", step, 1, {"error_type": "event_id_not_found_for_latency", "event_id": event_id})


if __name__ == '__main__':
    try:
        # This relative import will work if this script is run from the project root directory
        # and the 'scripts' directory is in the Python path.
        # For direct execution of this file, you might need to adjust sys.path or run as a module.
        from scripts.logging.metrics_logger import MetricsLogger
        logger_instance = MetricsLogger(experiment_name="gnw_metrics_test")
    except ImportError:
        print("MetricsLogger not found, running GNWMetrics without logging for this example.")
        print("Ensure 'scripts' directory is in PYTHONPATH or run this script as part of the project.")
        logger_instance = None

    # Example Usage
    num_test_modules = 5
    gnw_evaluator = GNWMetrics(
        num_modules=num_test_modules, 
        ignition_threshold_delta=0.5, 
        min_modules_for_ignition=2,
        logger=logger_instance
    )

    # Simulate activation updates
    activations_step0 = torch.tensor([0.1, 0.2, 0.1, 0.3, 0.2])
    gnw_evaluator.update_activations(activations_step0, step=0) # Initialize previous_activations

    activations_step1 = torch.tensor([0.2, 0.3, 0.8, 0.4, 0.9]) # Modules 2 and 4 should ignite (0-indexed)
    ignition1 = gnw_evaluator.update_activations(activations_step1, step=1)
    assert ignition1 == 1, f"Expected ignition at step 1, got {ignition1}"

    activations_step2 = torch.tensor([0.25, 0.35, 0.85, 0.45, 0.95]) # Small increase, no new ignition
    ignition2 = gnw_evaluator.update_activations(activations_step2, step=2)
    assert ignition2 == 0, f"Expected no ignition at step 2, got {ignition2}"
    
    activations_step3 = torch.tensor([1.0, 1.0, 0.2, 1.0, 0.3]) # Modules 0, 1, 3 should ignite
    ignition3 = gnw_evaluator.update_activations(activations_step3, step=3)
    assert ignition3 == 1, f"Expected ignition at step 3, got {ignition3}"

    # Simulate latency tracking
    event_id_1 = "visual_stimulus_001"
    gnw_evaluator.log_sensory_event_start(event_id_1)
    
    # Simulate some processing time
    time.sleep(0.05) 
    gnw_evaluator.log_event_reuse(event_id_1, module_name="ConsciousnessCore", step=4)
    
    time.sleep(0.02)
    gnw_evaluator.log_event_reuse(event_id_1, module_name="MemoryModule", step=4) # Example of logging multiple reuses

    if logger_instance:
        logger_instance.close()