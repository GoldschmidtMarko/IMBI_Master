import matplotlib.pyplot as plt

import matplotlib.font_manager as fm
# print(fm.findSystemFonts(fontpaths=None, fontext='ttf'))
custom_font_bold = fm.FontProperties(family='Arial', size=16, weight='bold')
custom_font = fm.FontProperties(family='Arial', size=16)

# Sample data for find_partition_time, calc_partition_time, and number_partitions
find_partition_time = [0.0,   0.0,  0.0,  0.0,  0.0,  0.0,  0.02,  0.05,  0.13,   0.40,   1.23,   4.57]
calc_partition_time = [0.01,  0.03, 0.13, 0.51, 1.34, 8.6,  14.3, 42.24,  262.91, 688.62, 963.35, 2700]
number_activities =   [2,     3,    4,    5,    6,    7,    8,    9,      10,     11,     12,     13]
number_partitions =   [2,     6,    14,   30,   62,   126,  254,  510,    1022,   2046,   4094,   8183]

# Create a line plot
fig, ax = plt.subplots(figsize=(10, 4))  # Adjust the figure size as needed
ax.plot(number_activities, find_partition_time, marker='o', color="blue")
ax.plot(number_activities, calc_partition_time, marker='x', color="orange")

# Add number_partitions labels above the data points
# for i, txt in enumerate(number_partitions):
    # plt.annotate(txt, (number_activities[i], calc_partition_time[i]), textcoords="offset points", xytext=(0, 5), ha='center', color="green")
    
# green_patch = plt.Line2D([0], [0], color='green', marker='', label='$|partitions|$')

# Add labels and a legend
ax.set_xlabel('Number of Activities', fontproperties=custom_font)
ax.set_ylabel('Time in (s)', fontproperties=custom_font)
ax.set_xticks(number_activities)
ax.set_title('Runtime of IMbi on different Number of Activities', fontproperties=custom_font_bold)
ax.legend(handles=[plt.Line2D([], [], color='blue', marker='o', label='$finding\_partitions$'), plt.Line2D([], [], color='orange', marker='x', label='$calc\_partitions$')])


ax2 = ax.twinx()
# Plot number_partitions on the second axis (right side)
ax2.plot(number_activities, number_partitions, label='$number\_of\_partitions$', color='green', marker='s')
ax2.set_ylabel('Number of Partitions', fontproperties=custom_font)
ax2.tick_params(axis='y')

ax2.legend(loc='upper right')

# ax.set_yscale('log')
# ax2.set_yscale('log')

ax.grid(True, linestyle='--', alpha=0.6)
ax.set_axisbelow(True)

# Show the plot
plt.savefig("plot_runtime.png")

ax.set_yscale('log')
ax2.set_yscale('log')

plt.savefig("plot_runtime_log.png")
