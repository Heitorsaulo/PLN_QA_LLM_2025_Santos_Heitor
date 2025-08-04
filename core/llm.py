from transformers import pipeline

class HuggingFaceLLM:
    def __init__(self, model_name: str = 'google/flan-t5-base'):
        self.generator = pipeline('text2text-generation', model=model_name)

    def generate(self, prompt: str) -> str:
        # Limitar o prompt para não exceder o limite do modelo
        max_input_length = 700  # Deixar margem de segurança
        if len(prompt) > max_input_length:
            prompt = prompt[:max_input_length] + "..."

        result = self.generator(prompt, max_new_tokens=512, max_length=2048)
        return result[0]['generated_text'] if 'generated_text' in result[0] else result[0]['text']
