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

# Example usage
if __name__ == "__main__":
    manager = SimulationManager()
    manager.execute_code("print('Hello, Unreal Engine!')")