import torch
from sympy.physics.units import temperature
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM


class HuggingFaceLLM:
    def __init__(self, model_name: str = 'google/flan-t5-xl', device: torch.device = None):
        self.device = device or torch.device("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=True) #AutoModelForSeq2SeqLM para text2text-generation
        self.model.to(self.device)
        self.max_model_input_tokens = getattr(self.tokenizer, "model_max_length", 6000)

    def generate(self, prompt: str, max_new_tokens: int = 128) -> str:
        """
        Gera texto garantindo que a entrada não ultrapasse o limite de tokens do modelo.
        Reservamos parte do espaço de tokens para a saída (max_new_tokens).
        """
        reserved_output = max_new_tokens
        allowed_input = max(1, self.max_model_input_tokens - reserved_output)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=allowed_input
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Geração de texto
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature = 0.4
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
