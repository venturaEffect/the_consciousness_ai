from transformers import AutoModelForCausalLM, AutoTokenizer

class NarrativeEngine:
    def __init__(self, model_name="meta-llama/Llama-3.3-13b-chat-hf"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            use_auth_token=True
        )

    def generate_interaction_code(self, task_description, environment_state):
        """
        Generates Python code to interact with the simulation based on the given task and environment state.
        """
        prompt = f"Write Python code to {task_description} given the environment state: {environment_state}"
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True
            )
        code = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return code

# Example usage
if __name__ == "__main__":
    engine = NarrativeEngine()
    generated_code = engine.generate_interaction_code(
        "move an object to a new location",
        "an object at position (0, 0, 0) must be moved to (100, 200, 50)"
    )
    print(generated_code)
