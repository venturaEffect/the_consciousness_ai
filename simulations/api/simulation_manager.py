import pandas as pd
from threading import Lock
import subprocess
import unreal
from models.self_model.reinforcement_core import ReinforcementCore

class SimulationManager:
    def __init__(self, config):
        self.lock = Lock()
        self.rl_core = ReinforcementCore(config)
        self.current_scenario = None

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

    def run_interaction(self, agent, environment, max_steps=1000):
        """
        Run interaction loop with reinforcement learning
        """
        state = environment.reset()
        total_reward = 0
        
        for step in range(max_steps):
            # Get action from agent
            action = agent.get_action(state)
            
            # Take step in environment
            next_state, reward, done, info = environment.step(action)
            
            # Get emotional response
            emotion_values = info.get('emotion_values', None)
            
            # Compute emotional reward
            emotional_reward = self.rl_core.compute_reward(
                state, emotion_values
            )
            
            # Update RL core
            update_info = self.rl_core.update(
                state, action, emotional_reward, next_state, done
            )
            
            total_reward += emotional_reward
            state = next_state
            
            if done:
                break
                
        return {
            'total_reward': total_reward,
            'steps': step + 1,
            'update_info': update_info
        }

# Example usage
if __name__ == "__main__":
    manager = SimulationManager(config={})
    manager.execute_code("print('Hello, Unreal Engine!')")
    manager.load_interaction_data()
