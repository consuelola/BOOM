import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# geochem = 'majors_or_traces'
geochem = 'majors_and_traces'
# geochem = 'majors_and_traces_restricted'
filename = 'test_accuracy'
# filename = 'test_balanced_accuracy'
file_accuracies = f'../results/{geochem}/{filename}.csv'
acc = pd.read_csv(file_accuracies, index_col=0)

# Convert from wide to long format
value_vars = list(acc.columns)
acc['repeat'] = acc.index
acc = pd.melt(acc, id_vars='repeat', value_vars=value_vars, var_name='method',
              value_name='test_accuracy')

# Create a colummn imputer
pos = [s.find('_') for s in acc.method]
pos = [j if j != -1 else 0 for j in pos]
acc['Imputer'] = [s[:j] for s, j in zip(acc.method, pos)]
acc.Imputer = acc.Imputer.map({
    'knn': 'kNN', 'mean': 'Mean', 'br': 'Iterative BR', 'rf': 'Iterative RF',
    '': ''})

# Create a column predictor
pos = [j+5 if j != 0 else 0 for j in pos]
acc['predictor'] = [s[j:] for s, j in zip(acc.method, pos)]

# Rename models with complete names
d = {
    'knn': 'k-Nearest \n Neighbors',
    'rf': 'Random \n Forest',
    'gb': 'Gradient \n Boosting',
    'lr': 'Logistic \n Regression'
}
acc.predictor = acc.predictor.map(d)

# Plot boxplots
plt.grid(axis='x')
ax = sns.boxplot(data=acc, x='test_accuracy', y='predictor', hue='Imputer')
ax.set_axisbelow(True)
if 'balanced' in filename:
    plt.xlabel('Test balanced accuracy')
else:
    plt.xlabel('Test accuracy')
plt.ylabel('')
plt.savefig(f'../figures/{geochem}/{filename}.pdf',
            dpi=300, bbox_inches='tight')
plt.close()
