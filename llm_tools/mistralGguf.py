from abc import ABC
from llama_cpp import Llama
from llm_tools.baseLLM import BaseLLM
from llm_tools.utils.fileManager import create_folder, download_file
from llm_tools.utils.mistralGgufTools import MistralGgufAssistant
import os


class MistralGGUF(BaseLLM, ABC):

    def __init__(self, device: str):
        self.model = None

        # Default generation parameters
        self.temperature = 0.6
        self.top_k = 20
        self.top_p = 1.0

    def load_model(self, model_path: str, model_type: str = 'mistral_gguf') -> None:
        """
        Loads the specified model and tokenizer based on the model type.
        """

        folder = 'mistral/model'
        file_name = model_path.split('/')[-1]
        create_folder(folder)
        download_file(model_path, folder, file_name)

        model = Llama(
            model_path=os.path.join('/', folder, file_name),
            n_ctx=2000,
            n_parts=1,
        )
        self.model = MistralGgufAssistant(model)

    def get_answer(self, prompt: str) -> str:
        """
        Generates an answer for the given prompt using the loaded model.
        """
        self.model.get_answer(prompt, self.top_k, self.top_p, self.temperature)

    def set_generation_parameters(self, max_length: int = None, temperature: float = None, top_k: int = None,
                                  top_p: float = None, no_repeat_ngram_size: int = None):
        """
        Allows setting the generation parameters: max_length, temperature, top_k, and top_p.
        """
        if temperature is not None:
            self.temperature = temperature
        if top_k is not None:
            self.top_k = top_k
        if top_p is not None:
            self.top_p = top_p

    @staticmethod
    def get_set_types():
        return {'mistral_gguf'}
