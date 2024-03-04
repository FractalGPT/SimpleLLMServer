from llm_tools.baseLLM import BaseLLM
from llm_tools.gptHuggingFace import GPTHuggingFace
from llm_tools.t5HuggingFace import T5HuggingFace


class LLMInference(BaseLLM):
    def __init__(self, device: str = 'cpu'):
        self.llm = None
        self.device = device
        self.llms_type = [GPTHuggingFace, T5HuggingFace]

    def load_model(self, model_name_or_path: str, model_type: str = 'gpt2') -> None:
        """
        Loads the specified model and tokenizer based on the model type.
        """
        model_type_ = model_type.lower()

        for llm_type in self.llms_type:
            if model_type_ in llm_type.get_set_types():
                self.llm = llm_type(self.device)
                break
        self.llm.load_model(model_name_or_path, model_type)

    def get_answer(self, prompt: str) -> str:
        """
        Generates an answer for the given prompt using the loaded model.
        """
        return self.llm.get_answer(prompt)

    def set_generation_parameters(self, max_length: int = None, temperature: float = None, top_k: int = None,
                                  top_p: float = None, no_repeat_ngram_size: int = None):
        """
        Allows setting the generation parameters: max_length, temperature, top_k, and top_p.
        """
        self.llm.set_generation_parameters(max_length, temperature, top_k, top_p, no_repeat_ngram_size)
