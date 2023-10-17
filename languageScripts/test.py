from decimal import Decimal

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

# Example usage:
list_length = 3
step_size = Decimal('0.1')

possible_lists = generate_lists(list_length, step_size)
possible_lists_float = []
for l in possible_lists:
    possible_lists_float.append([float(value) for value in l])
