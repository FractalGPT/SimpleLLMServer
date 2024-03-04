from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from llm_tools.baseLLM import BaseLLM


class T5HuggingFace(BaseLLM):
    def __init__(self, device: str):
        self.model = None
        self.tokenizer = None
        self.model_type = None
        self.device = device

        # Default generation parameters
        self.max_length = 50
        self.temperature = 0.6
        self.top_k = 20
        self.top_p = 1.0
        self.no_repeat_ngram_size = 3

    def load_model(self, model_name_or_path: str, model_type: str = 't5') -> None:
        """
        Loads the specified model and tokenizer based on the model type.
        """
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    def get_answer(self, prompt: str) -> str:
        """
        Generates an answer for the given prompt using the loaded model.
        """
        if not self.model or not self.tokenizer:
            raise Exception("Model not loaded. Please load a model before generating text.")

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        generation_parameters = {
            "max_length": self.max_length,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "num_return_sequences": 1,
            "do_sample": True,
            "no_repeat_ngram_size": self.no_repeat_ngram_size
        }

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(input_ids, **generation_parameters)

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def set_generation_parameters(self, max_length: int = None, temperature: float = None, top_k: int = None,
                                  top_p: float = None, no_repeat_ngram_size: int = None):
        """
        Allows setting the generation parameters: max_length, temperature, top_k, and top_p.
        """
        if max_length is not None:
            self.max_length = max_length
        if temperature is not None:
            self.temperature = temperature
        if top_k is not None:
            self.top_k = top_k
        if top_p is not None:
            self.top_p = top_p
        if top_p is not None:
            self.no_repeat_ngram_size = no_repeat_ngram_size

    @staticmethod
    def get_set_types():
        return {'t5', 'fred'}