from fastapi import APIRouter, Query

from llm_tools.lLMInference import LLMInference

router = APIRouter()
llm = LLMInference()


@router.get("/load_gpt_model/{model_name}")
async def load_gpt_model(model_name: str):
    llm.load_model(model_name)


@router.get("/load_t5_model/{model_name}")
async def load_t5_model(model_name: str):
    llm.load_model(model_name, "t5")


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
    llm.set_generation_parameters(max_length=max_length, temperature=temperature, top_k=top_k, top_p=top_p, no_repeat_ngram_size=no_repeat_ngram_size)
    return {"answer": llm.get_answer(prompt)}
