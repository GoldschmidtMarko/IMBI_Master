import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.metrics import f1_score
import math
from decimal import Decimal

# Sample data: Language logs
language_logs = {
    "L_Plus": [0.3, 0.2, 0.2, 0.3],
    "L_Minus": [0.0, 0.5, 0.4, 0.1]
}

def f1_score(a, b):
  # try: catch
  if math.isclose(a+b,0):
    return 0
  return 2 * (a * b) / (a + b)

def get_distance(a, b):
  return wasserstein_distance(np.arange(len(a)), np.arange(len(a)), a, b)

def get_distance_result(language_A, language_B, language_Star):
  # Convert the data to numpy arrays
  l_plus = np.array(language_A)
  l_minus = np.array(language_B)
  l_star = np.array(language_Star)
  # Calculate the EMD between LStar and LPlus
  emd_lstar_lplus = get_distance(l_star, l_plus)
  # Calculate the EMD between LStar and LMinus
  emd_lstar_lminus = get_distance(l_star, l_minus)
  f1 = f1_score(1 - emd_lstar_lplus, emd_lstar_lminus)
  return {"emd_lstar_lplus": emd_lstar_lplus, "emd_lstar_lminus": emd_lstar_lminus, "f1": f1}

def generate_all_Star_candidates(list_length, step_size_float = 0.1):
  def generate_lists(list_length, step_size, remaining_sum=Decimal('1'), current_list=None, results=None, epsilon=Decimal('1e-10')):
      if current_list is None:
          current_list = []
      if results is None:
          results = []

      if list_length == 0:
          if abs(remaining_sum) < epsilon:
              results.append(current_list.copy())
          return results

      num_steps = int((remaining_sum + epsilon) / step_size)
      for i in range(num_steps + 1):
          value = i * step_size
          new_list = current_list + [value]
          generate_lists(list_length - 1, step_size, remaining_sum - value, new_list, results, epsilon)

      return results

  step_size = Decimal(str(step_size_float))
  possible_lists = generate_lists(list_length, step_size)
  possible_lists_float = []
  for l in possible_lists:
      possible_lists_float.append([float(value) for value in l])
      
  return possible_lists_float

def get_best_distance(input_logs, step_size_float = 0.1):
  candidates = generate_all_Star_candidates(len(input_logs["L_Plus"]), step_size_float)
  print("Candidates: ", len(candidates))
  
  best_candidate = None
  best_result = None
  alternative_solutions = []
  for candidate in candidates:
    result = get_distance_result(input_logs["L_Plus"], input_logs["L_Minus"], candidate)
    if best_result == None or result["f1"] > best_result["f1"]:
      best_candidate = candidate
      best_result = result
      number_best_results = 1
      alternative_solutions = []
    elif result["f1"] == best_result["f1"]:
      alternative_solutions.append(candidate)
    
  return {"best_candidate": best_candidate, "best_result": best_result, "alternative_solutions": alternative_solutions}
  

if __name__ == '__main__':
  print("Running algorithm for funding the best LStar")
  print()
  # Get the language logs
  res = get_best_distance(language_logs, step_size_float=0.05)
  print("LogP:  ", language_logs["L_Plus"])
  print("LogM:  ", language_logs["L_Minus"])
  print("LStar: ", res["best_candidate"])
  print("Best result: ", res["best_result"])
  print("Number of best results: ", len(res["alternative_solutions"]))
  print()
  print("Alternatives:")
  print("LogP:  ", language_logs["L_Plus"])
  print("LogM:  ", language_logs["L_Minus"])
  print()
  for i, alternative in enumerate(res["alternative_solutions"]):
    if i > 5:
      break
    print("LStar: ", alternative)
  
  



