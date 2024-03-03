from pydantic import BaseModel


class TextGenerationRequest(BaseModel):
    prompt: str = 'Вопрос: привет\nОтвет:'
    max_length: int = 50
    temperature: float = 0.6
    top_k: int = 15
    top_p: float = 0.8
    no_repeat_ngram_size: int = 3

class LoadModelRequest(BaseModel):
    model_name: str = 'IlyaGusev/rugpt_medium_turbo_instructed'
    model_type: str = 'gpt2'
