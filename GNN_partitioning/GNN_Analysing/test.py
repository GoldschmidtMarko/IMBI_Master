import pandas as pd

def calculate_precision_difference(df):
    # Group the DataFrame by 'logP_Name', 'logM_Name', and 'cut_type'
    grouped = df.groupby(['logP_Name', 'logM_Name', 'cut_type'])
    
    # Calculate the precision difference within each group
    def custom_agg(group):
        precision_diff = group.loc[group['use_gnn'] == False, 'precision'].mean() - \
                        group.loc[group['use_gnn'] == True, 'precision'].mean()
        return pd.Series({'precision_diff': precision_diff})
    
    # Apply the custom aggregation function and reset the index
    result_df = grouped.apply(custom_agg).reset_index()
    
    return result_df

# Example usage:
data = {
    'logP_Name': [
        "\\seq\\Data_5\\Sup_1.0\\treeP_5_Sup_1.0_Data_test1",
        "\\seq\\Data_5\\Sup_1.0\\treeP_5_Sup_1.0_Data_test1",
        "\\seq\\Data_5\\Sup_1.0\\treeP_5_Sup_1.0_Data_test10",
        "\\seq\\Data_5\\Sup_1.0\\treeP_5_Sup_1.0_Data_test10",
    ],
    'logM_Name': ['seq'] * 4,
    'cut_type': ['seq'] * 4,
    'precision': [1.00, 1.00, 0.82, 0.82],
    'fitP': [1.00, 1.00, 0.84, 0.84],
    'use_gnn': [False, True, False, True],
}

df = pd.DataFrame(data)

result_df = calculate_precision_difference(df)
print(result_df)
