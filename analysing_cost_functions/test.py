import matplotlib.pyplot as plt

categories = ['Category 1', 'Category 2', 'Category 3', 'Category 4']
values = [10, 20, 15, 25]

plt.bar(categories, values, color='blue', alpha=0.7)

# Set the position of the bar where you want the dotted line (e.g., above Category 3)
dotted_line_position = categories.index('Category 3')
line_height = values[dotted_line_position] + 5  # Adjust the height of the line
dotted_line_position = categories.index('Category 3') + 0.5  # 0.5 is the middle of the bar

# Add a horizontal dotted line above the specific bar
plt.hlines(line_height, xmin=dotted_line_position-1, xmax=dotted_line_position, colors='red', linestyles='dotted', linewidth=2)
# Add a comment or label above the line
comment = "Above Category 3"
plt.text(dotted_line_position-0.5, line_height + 0.5, comment, ha='center', color='red')


plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('Bar Plot with Dotted Line Above One Bar')
plt.show()
