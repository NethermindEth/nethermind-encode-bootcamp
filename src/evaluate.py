import os
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
from langfuse import Langfuse
from langchain.evaluation import load_evaluator
from langchain_openai import OpenAI
from langchain.evaluation.criteria import LabeledCriteriaEvalChain
from datetime import datetime, timedelta
from pydantic import ValidationError

langfuse = Langfuse()
langfuse.auth_check()

# https://python.langchain.com/v0.1/docs/guides/productionization/evaluation/string/criteria_eval_chain/
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
        response = langfuse.get_generations(name=name, 
                                            limit=limit, 
                                            user_id=user_id, 
                                            page=page,
                                            from_start_time=datetime.now() - timedelta(days=1))
        if not response.data:
            break
 
        all_data.extend(response.data)
        page += 1
 
    return all_data

def get_evaluator_for_key(key: str):
  llm = OpenAI(temperature=0, model="gpt-3.5-turbo-instruct")
  return load_evaluator("criteria", criteria=key, llm=llm)

def execute_eval_and_score(generations):
  total_generations = len(generations) * len(EVAL_TYPES)
  with Progress(
      SpinnerColumn(),
      TextColumn("[progress.description]{task.description}"),
      BarColumn(bar_width=None),
      TaskProgressColumn(),
      TimeElapsedColumn(),
      TimeRemainingColumn(),
      MofNCompleteColumn(),
      expand=True
  ) as progress:
    task = progress.add_task("Evaluating Generations", total=total_generations)
    for generation in generations:
      criteria = [key for key, value in EVAL_TYPES.items() if value and key != "hallucination"]
      for criterion in criteria:
        try:
          eval_result = get_evaluator_for_key(criterion).evaluate_strings(
            prediction=generation.output,
            input=generation.input,
          )
          progress.update(task, advance=1)
          if eval_result["score"] is None:
            print(f"Skipping score for criterion '{criterion}' in generation {generation.id}: Score is None")
            continue
          langfuse.score(name=criterion, trace_id=generation.trace_id, observation_id=generation.id, value=eval_result["score"], comment=eval_result['reasoning'])
        except ValidationError as e:
          print(f"Validation error for criterion '{criterion}' in generation {generation.id}: {str(e)}")
        except Exception as e:
          print(f"Error evaluating criterion '{criterion}' in generation {generation.id}: {str(e)}")
      
 
try:
  generations = fetch_all_pages(user_id='test_user')
  execute_eval_and_score(generations)
  langfuse.flush()
except Exception as e:
  print(f"Error fetching generations: {str(e)}")

