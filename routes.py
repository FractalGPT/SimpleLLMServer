from fastapi import APIRouter, Query

from llm_tools.lLMInference import LLMInference

router = APIRouter()
llm = LLMInference()

llm.load_model('IlyaGusev/rugpt_medium_turbo_instructed')


@router.get("/load_llm_model/{model_name}")
async def load_llm_model(model_name: str, model_type: str = Query('gpt2', alias="model type")):
    llm.load_model(model_name, model_type)


@router.get("/text_generation/{prompt}")
async def text_generation(
        prompt: str,
        max_length: int = Query(50, alias="maxLen"),
        temperature: float = Query(0.6, alias="temp"),
        top_k: int = Query(15, alias="topK"),
        top_p: float = Query(0.8, alias="topP"),
        no_repeat_ngram_size: int = Query(3, alias="no_repeat_ngram_size")
):
    # Set the generation parameters before generating the answer
    llm.set_generation_parameters(max_length=max_length, temperature=temperature, top_k=top_k, top_p=top_p,
                                  no_repeat_ngram_size=no_repeat_ngram_size)
    return {"answer": llm.get_answer(prompt)}
