def find_max_difference_tuple(tuples):
    max_values = None
    iterator = None

    for i, (a, b) in enumerate(tuples):
      if max_values == None:
        max_values = (a, b)
        iterator = i
        continue
      if a > max_values[0] and b > max_values[1]:
        max_values = (a, b)
        iterator = i
        continue
      
      if (a - max_values[0]) + (b - max_values[1]) > 0:
        max_values = (a, b)
        iterator = i
        continue

    return iterator, max_values

# Example usage
tuples = [(0.8, 0.5), (1, 0.2), (0.8, 0.6)]
result = find_max_difference_tuple(tuples)
print("Tuple with values bigger than every other tuple value or with the highest difference:", result)
