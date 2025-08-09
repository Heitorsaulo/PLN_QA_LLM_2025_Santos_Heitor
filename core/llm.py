from torch import device
from transformers import pipeline

class HuggingFaceLLM:
    def __init__(self, model_name: str = 'google/flan-t5-base', device : device = 'cpu'):
        self.generator = pipeline('text2text-generation', model=model_name, device=device)

    def generate(self, prompt: str) -> str:
        # Limitar o prompt para não exceder o limite do modelo
        max_input_length = 1100  # Deixar margem de segurança
        if len(prompt) > max_input_length:
            prompt = prompt[:max_input_length] + "..."

        result = self.generator(prompt, max_new_tokens=2048, max_length=4096)
        return result[0]['generated_text'] if 'generated_text' in result[0] else result[0]['text']
