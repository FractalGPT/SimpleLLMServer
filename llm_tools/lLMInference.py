from llm_tools.baseLLM import BaseLLM
from llm_tools.huggingFaceLLM import HuggingFaceLLM
# from llm_tools.mistralGguf import MistralGGUF


class LLMInference(BaseLLM):
    def __init__(self, device: str = 'cpu'):
        self.llm = None
        self.device = device

    def load_model(self, model_name_or_path: str, model_type: str = 'gpt2') -> None:
        """
        Loads the specified model and tokenizer based on the model type.
        """
        if model_type in HuggingFaceLLM.get_set_types():
            self.llm = HuggingFaceLLM(self.device)
        # if model_type in MistralGGUF.get_set_types():
        #     self.llm = MistralGGUF(self.device)

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
