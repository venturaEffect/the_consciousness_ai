import pandas as pd
from threading import Lock
import subprocess
import unreal

class SimulationManager:
    def __init__(self):
        self.lock = Lock()

    def execute_code(self, script: str):
        """
        Executes the provided Python code within the simulation environment.
        """
        try:
            with self.lock:
                # Save the script to a temporary file
                with open("temp_script.py", "w") as temp_file:
                    temp_file.write(script)
                
                # Execute the script
                result = subprocess.run(["python", "temp_script.py"], capture_output=True, text=True)

                # Log the result
                if result.returncode == 0:
                    print(f"Script executed successfully: {result.stdout}")
                else:
                    print(f"Script execution failed: {result.stderr}")

                return result
        except Exception as e:
            print(f"Error during script execution: {str(e)}")

    def load_interaction_data(self):
        """Load INTERACTION and UE-HRI datasets for simulations."""
        try:
            # Load INTERACTION dataset
            interaction_data = pd.read_csv('/data/simulations/interaction_data.csv')
            print("INTERACTION data loaded successfully.")

            # Load UE-HRI dataset
            ue_hri_data = pd.read_csv('/data/simulations/ue_hri_data.csv')
            print("UE-HRI data loaded successfully.")

        except Exception as e:
            print(f"Error loading datasets: {e}")

# Example usage
if __name__ == "__main__":
    manager = SimulationManager()
    manager.execute_code("print('Hello, Unreal Engine!')")
    manager.load_interaction_data()
