from fastapi import APIRouter

from llm_tools.lLMInference import LLMInference
from llm_tools.models.generationModels import LoadModelRequest, TextGenerationRequest
from settings import load_config

router = APIRouter()
llm = LLMInference()

load_config(llm=llm)


@router.post("/load_llm_model/")
async def load_llm_model(request: LoadModelRequest):
    llm.load_model(request.model_name, request.model_type)
    return {"message": "Model loaded successfully"}


@router.post("/text_generation/")
async def text_generation(request: TextGenerationRequest):
    # Set the generation parameters before generating the answer
    llm.set_generation_parameters(max_length=request.max_length, temperature=request.temperature, top_k=request.top_k,
                                  top_p=request.top_p,
                                  no_repeat_ngram_size=request.no_repeat_ngram_size)
    return {"answer": llm.get_answer(request.prompt)}
