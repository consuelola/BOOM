import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

geochem = 'majors_and_traces'
file_accuracies = f'../results/{geochem}/test_accuracy.csv'
acc = pd.read_csv(file_accuracies, index_col=0)

# Convert from wide to lon format
value_vars = list(acc.columns)
acc['repeat'] = acc.index
acc = pd.melt(acc, id_vars='repeat', value_vars=value_vars, var_name='method',
              value_name='test_accuracy')

# Create a colummn imputer
pos = [s.find('_') for s in acc.method]
pos = [j if j != -1 else 0 for j in pos]
acc['imputer'] = [s[:j] for s, j in zip(acc.method, pos)]

# Create a column predictor
pos = [j+5 if j != 0 else 0 for j in pos]
acc['predictor'] = [s[j:] for s, j in zip(acc.method, pos)]

# Plot boxplots
sns.boxplot(data=acc, x='test_accuracy', y='predictor', hue='imputer')
plt.savefig(f'../figures/{geochem}/test_accuracy.pdf',
            dpi=300, bbox_inches='tight')
plt.close()
