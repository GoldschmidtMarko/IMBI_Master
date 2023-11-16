import numpy as np
from scipy.optimize import linear_sum_assignment

# Your custom distance function
def custom_distance(s1, s2):
  len_s1, len_s2 = len(s1), len(s2)

  # Create a 2D array to store the edit distances
  dp = [[0] * (len_s2 + 1) for _ in range(len_s1 + 1)]

  # Initialize the first row and column
  for i in range(len_s1 + 1):
      dp[i][0] = i
  for j in range(len_s2 + 1):
      dp[0][j] = j

  # Fill in the dynamic programming table
  for i in range(1, len_s1 + 1):
      for j in range(1, len_s2 + 1):
          if s1[i - 1] != s2[j - 1]:
              # Exclude substitution, only consider insertion and deletion
              dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1)
          else:
              dp[i][j] = dp[i - 1][j - 1]

  # The Levenshtein distance is the value in the bottom-right cell of the matrix
  return dp[len_s1][len_s2]

# Function to compute the Earth Mover's Distance (EMD) and matching frequencies
def earth_mover_distance_with_custom_distances(set1, set2):
    num_set1, num_set2 = len(set1), len(set2)

    # Create a cost matrix with custom distances
    cost_matrix = np.zeros((num_set1, num_set2))
    for i in range(num_set1):
        for j in range(num_set2):
            cost_matrix[i][j] = custom_distance(set1[i][0], set2[j][0])

    # Initialize the relative frequencies matrix
    relative_frequencies = np.zeros((num_set1, num_set2))

    for _ in range(min(num_set1, num_set2)):
        # Find the minimum cost element in the cost matrix
        min_cost_index = np.unravel_index(cost_matrix.argmin(), cost_matrix.shape)
        i, j = min_cost_index

        # Calculate the matched frequency based on the item frequencies
        matched_frequency = min(set1[i][1], set2[j][1])

        # Update the relative frequencies matrix
        relative_frequencies[i][j] = matched_frequency

        # Update item frequencies
        set1[i] = (set1[i][0], set1[i][1] - matched_frequency)
        set2[j] = (set2[j][0], set2[j][1] - matched_frequency)

        # Set the cost to a large value to avoid re-matching the same item
        cost_matrix[i, :] = np.inf
        cost_matrix[:, j] = np.inf

    # Calculate the total cost (EMD) based on the matched frequencies
    emd = np.sum(cost_matrix * relative_frequencies)

    return emd, relative_frequencies

# Example usage:
set1 = [("wasd", 0.2), ("oooo", 0.4), ("wwxxyyxy", 0.4)]
set2 = [("xsxsx", 0.5), ("oooo", 0.3), ("kikik", 0.2)]

emd, relative_frequencies = earth_mover_distance_with_custom_distances(set1, set2)
print("Earth Mover's Distance:", emd)
print("Relative Frequency Matching:")
print(relative_frequencies)