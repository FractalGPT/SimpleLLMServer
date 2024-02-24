from abc import ABC, abstractmethod


class BaseLLM(ABC):

    @abstractmethod
    def load_model(self, model_name_or_path: str, model_type: str):
        pass

    @abstractmethod
    def get_answer(self, prompt: str):
        pass

    @abstractmethod
    def set_generation_parameters(self, max_length: int = None, temperature: float = None, top_k: int = None,
                                  top_p: float = None, no_repeat_ngram_size: int = None):
        pass
