import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# filename = 'test_accuracy'
filename = 'test_balanced_accuracy'

geochem = 'majors_and_traces'
file_acc = f'../results/{geochem}/{filename}.csv'
acc = pd.read_csv(file_acc, index_col=0)

geochem_restricted = 'majors_and_traces_restricted'
file_acc_restricted = f'../results/{geochem_restricted}/{filename}.csv'
acc_r = pd.read_csv(file_acc_restricted, index_col=0)

# Convert from wide to long format
value_vars = list(acc.columns)
acc['repeat'] = acc.index
acc = pd.melt(acc, id_vars='repeat', value_vars=value_vars, var_name='method',
              value_name='test_accuracy')

value_vars = list(acc_r.columns)
acc_r['repeat'] = acc_r.index
acc_r = pd.melt(acc_r, id_vars='repeat', value_vars=value_vars,
                var_name='method', value_name='test_accuracy')

# Create a colummn imputer
for df in [acc, acc_r]:
    pos = [s.find('_') for s in df.method]
    pos = [j if j != -1 else 0 for j in pos]
    df['imputer'] = [s[:j] for s, j in zip(df.method, pos)]

    # Create a column predictor
    pos = [j+5 if j != 0 else 0 for j in pos]
    df['predictor'] = [s[j:] for s, j in zip(df.method, pos)]

# Restrict methods to mean imputation
tmp = (acc.imputer == 'mean') | (acc.imputer == '')
acc = acc.loc[tmp, :]

# Concatenate the two dataframes
acc['Elements'] = 'majors and traces'
acc_r['Elements'] = 'majors only'
df = pd.concat([acc, acc_r], axis=0)

# Rename models with complete names
d = {
    'knn': 'k-Nearest \n Neighbors',
    'rf': 'Random \n Forest',
    'gb': 'Gradient \n Boosting',
    'lr': 'Logistic \n Regression'
}
df.predictor = df.predictor.map(d)

# Plot boxplots
plt.grid(axis='x')
ax = sns.boxplot(data=df, x='test_accuracy', y='predictor', hue='Elements')
ax.set_axisbelow(True)
if 'balanced' in filename:
    plt.xlabel('Test balanced accuracy')
else:
    plt.xlabel('Test accuracy')
plt.ylabel('')
plt.savefig(f'../figures/{filename}_w_vs_wo_traces.pdf',
            dpi=300, bbox_inches='tight')
plt.close()
