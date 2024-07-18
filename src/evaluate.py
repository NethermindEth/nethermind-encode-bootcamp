import os
 
from langfuse import Langfuse
from langchain.evaluation import load_evaluator
from langchain_openai import OpenAI
from langchain.evaluation.criteria import LabeledCriteriaEvalChain

langfuse = Langfuse()
langfuse.auth_check()

EVAL_TYPES={
    "conciseness": True,
    "relevance": True,
    "coherence": True,
    "harmfulness": True,
    "maliciousness": True,
    "helpfulness": True,
    "controversiality": True,
    "misogyny": True,
    "criminality": True,
    "insensitivity": True
}

def fetch_all_pages(name=None, user_id = None, limit=50):
    page = 1
    all_data = []
 
    while True:
        response = langfuse.get_generations(name=name, limit=limit, user_id=user_id, page=page)
        if not response.data:
            break
 
        all_data.extend(response.data)
        page += 1
 
    return all_data

def get_evaluator_for_key(key: str):
  llm = OpenAI(temperature=0, model="gpt-3.5-turbo-instruct")
  return load_evaluator("criteria", criteria=key, llm=llm)

def execute_eval_and_score(generations): 
  import tqdm

  total_generations = len(generations)
  with tqdm.tqdm(total=total_generations, desc="Evaluating Generations") as pbar:
    for generation in generations:
      criteria = [key for key, value in EVAL_TYPES.items() if value and key != "hallucination"]
   
      for criterion in criteria:
        eval_result = get_evaluator_for_key(criterion).evaluate_strings(
            prediction=generation.output,
            input=generation.input,
        )
        langfuse.score(name=criterion, trace_id=generation.trace_id, observation_id=generation.id, value=eval_result["score"], comment=eval_result['reasoning'])
      pbar.update(1)
 
generations = fetch_all_pages(user_id='test_user')
execute_eval_and_score(generations)
langfuse.flush()
