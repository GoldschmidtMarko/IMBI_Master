import matplotlib.pyplot as plt

import matplotlib.font_manager as fm

custom_font_bold = fm.FontProperties(family='Arial', size=16, weight='bold')
custom_font = fm.FontProperties(family='Arial', size=16)

data_runtime = {8: (304.9293097138405, 41.03373319864273), 7: (110.61716252803802, 25.475533925294876),
                6: (45.13468473315239, 18.32646267414093), 5: (14.434719350337982, 10.27313573360443),
                4: (6.632375111579895, 6.124065421819687), 3: (2.4538823199272155, 2.896251004934311),
                2: (0.7977814471721649, 1.045875459909439)}

number_activities = []
runtime_reachability = []
runtime_gnn = []
for key, (value1, value2) in data_runtime.items():
    number_activities.append(key)
    runtime_reachability.append(value1)
    runtime_gnn.append(value2)

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))  # 1 row, 2 columns

# Plot data on the first subplot (ax1)
ax1.plot(number_activities, runtime_reachability, marker='o', color="blue")
ax1.plot(number_activities, runtime_gnn, marker='x', color="green")
ax1.set_xlabel('Number of Activities', fontproperties=custom_font)
ax1.set_ylabel('Time in (s)', fontproperties=custom_font)
ax1.set_xticks(number_activities)
ax1.legend(handles=[plt.Line2D([], [], color='blue', marker='o', label='reachability'), plt.Line2D([], [], color='green', marker='x', label='graph neural network')])
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.set_axisbelow(True)

# Plot data on the second subplot (ax2)
ax2.plot(number_activities, runtime_reachability, marker='o', color="blue")
ax2.plot(number_activities, runtime_gnn, marker='x', color="green")
ax2.set_xlabel('Number of Activities', fontproperties=custom_font)
ax2.set_ylabel('Time in (s)', fontproperties=custom_font)
ax2.set_xticks(number_activities)
ax2.legend(handles=[plt.Line2D([], [], color='blue', marker='o', label='reachability'), plt.Line2D([], [], color='green', marker='x', label='graph neural network')])
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.set_axisbelow(True)

# Set the scale of the second subplot to logarithmic
ax2.set_yscale('log')

# Save the plots
plt.tight_layout()  # Adjusts subplots to avoid overlap
plt.savefig("plot_runtime_gnn_comparison_side_by_side.pdf", format="pdf")
plt.show()
