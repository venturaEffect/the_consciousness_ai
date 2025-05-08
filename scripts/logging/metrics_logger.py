import json
import os
import datetime
import torch # Assuming PyTorch for tensor handling

class MetricsLogger:
    def __init__(self, log_dir="logs/metrics_data", experiment_name=None):
        """
        Initializes the MetricsLogger.

        Args:
            log_dir (str): Directory to save log files.
            experiment_name (str, optional): A name for the current experiment or run.
                                            If None, a timestamp will be used.
        """
        if experiment_name is None:
            experiment_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.log_path = os.path.join(log_dir, experiment_name)
        os.makedirs(self.log_path, exist_ok=True)
        
        self.log_files = {} # To store open file handlers for different metrics
        print(f"MetricsLogger initialized. Logging to: {self.log_path}")

    def _get_log_file(self, metric_name):
        """
        Gets or creates a log file for a specific metric.
        Each metric will be logged in a separate JSONL file.
        """
        if metric_name not in self.log_files:
            file_path = os.path.join(self.log_path, f"{metric_name}.jsonl")
            self.log_files[metric_name] = open(file_path, 'a')
        return self.log_files[metric_name]

    def log_tensor_data(self, metric_name: str, step: int, tensor_data: torch.Tensor, metadata: dict = None):
        """
        Logs tensor data along with a step and optional metadata.
        The tensor data will be converted to a list for JSON serialization.

        Args:
            metric_name (str): Name of the metric being logged (e.g., "gnw_activation", "phi_star_input").
            step (int): Current simulation step or training iteration.
            tensor_data (torch.Tensor): The tensor data to log.
            metadata (dict, optional): Additional dictionary of metadata to log.
        """
        if not isinstance(tensor_data, torch.Tensor):
            print(f"Warning: tensor_data for {metric_name} is not a PyTorch tensor. Skipping.")
            return

        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "step": step,
            "metric_name": metric_name,
            "tensor_shape": list(tensor_data.shape),
            "tensor_data": tensor_data.detach().cpu().tolist(), # Convert to list for JSON
            "metadata": metadata if metadata else {}
        }
        
        log_file = self._get_log_file(metric_name)
        log_file.write(json.dumps(log_entry) + '\n')
        log_file.flush() # Ensure data is written to disk

    def log_scalar_data(self, metric_name: str, step: int, scalar_value, metadata: dict = None):
        """
        Logs scalar data along with a step and optional metadata.

        Args:
            metric_name (str): Name of the metric (e.g., "loss", "reward", "phi_value").
            step (int): Current simulation step or training iteration.
            scalar_value: The scalar value to log.
            metadata (dict, optional): Additional dictionary of metadata to log.
        """
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "step": step,
            "metric_name": metric_name,
            "value": scalar_value,
            "metadata": metadata if metadata else {}
        }
        
        log_file = self._get_log_file(metric_name)
        log_file.write(json.dumps(log_entry) + '\n')
        log_file.flush()

    def close(self):
        """
        Closes all open log files.
        """
        for f_handler in self.log_files.values():
            f_handler.close()
        self.log_files = {}
        print("MetricsLogger closed.")

if __name__ == '__main__':
    # Example Usage
    logger = MetricsLogger(experiment_name="test_run_001")

    # Simulate logging some tensor data
    example_tensor = torch.rand((2, 3, 4))
    logger.log_tensor_data(
        metric_name="sample_hidden_state", 
        step=1, 
        tensor_data=example_tensor,
        metadata={"layer": "encoder_block_3", "module": "ConsciousnessCore"}
    )
    
    example_tensor_2 = torch.ones((5,5))
    logger.log_tensor_data(
        metric_name="attention_weights",
        step=1,
        tensor_data=example_tensor_2,
        metadata={"head": 2}
    )

    # Simulate logging some scalar data
    logger.log_scalar_data(
        metric_name="phi_star_value", 
        step=1, 
        scalar_value=0.75,
        metadata={"partition_type": "min_cut"}
    )
    logger.log_scalar_data(
        metric_name="gnw_ignition_events", 
        step=1, 
        scalar_value=1
    )
    
    # ... more logging at different steps ...
    logger.log_scalar_data(
        metric_name="phi_star_value", 
        step=2, 
        scalar_value=0.78,
        metadata={"partition_type": "min_cut"}
    )

    logger.close()

    print(f"Example logs created in {logger.log_path}")
    print("Check the .jsonl files in that directory.")
