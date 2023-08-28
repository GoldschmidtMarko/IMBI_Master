import matplotlib.pyplot as plt

# Assuming you have already defined df_measurement, column_feature, and df_grouped

num_cols = len(column_feature)
num_cut_types = len(df_grouped)

fig, axes = plt.subplots(nrows=num_cut_types, ncols=num_cols, figsize=(16, 10))
fig.suptitle('Synthetic Data Delta Measurement\nDelta = (No GNN) - (GNN)')

for i, (cut_type, group) in enumerate(df_grouped):
    for j, col in enumerate(column_feature):
        ax = axes[i, j]  # Select the specific axis for this subplot
        ax.set_title("Cut: " + cut_type + "\n" + col)  # Set title with both cut_type and column_feature
        ax.set_xlabel("Measurement")
        ax.set_ylabel("Values")
        group.boxplot(column=col, ax=ax)  # Use group instead of df for boxplot
        
        # Modify x-axis tick labels
        x_ticks = ax.get_xticks()
        x_labels = ["Δ " + col] * len(x_ticks)
        ax.set_xticklabels(x_labels, rotation=0)
        
        ax.set_ylim(-1, 1)  # Set y-axis limits
        ax.set_yticks([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

plt.tight_layout()
plt.show()
