import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc

# activate latex text rendering
rc('text', usetex=True)

# Rename models
d = {
    'knn': 'KNN',
    'rf': 'RF',
    'gb': 'GB',
    'lr': 'LR'
}

fig, axes = plt.subplots(2, 2, sharey=True, figsize=(9, 9),
                         gridspec_kw={'width_ratios': [2.5, 1]})

# Plot majors_or_traces
for i, filename in enumerate(['test_accuracy', 'test_balanced_accuracy']):
    ax = axes[i, 0]
    is_legend = i == 0

    geochem = 'majors_or_traces'

    file_accuracies = f'../results/{geochem}/{filename}.csv'
    acc = pd.read_csv(file_accuracies, index_col=0)

    # Convert from wide to long format
    value_vars = list(acc.columns)
    acc['repeat'] = acc.index
    acc = pd.melt(acc, id_vars='repeat', value_vars=value_vars,
                  var_name='method', value_name='test_accuracy')

    # Create a colummn imputer
    pos = [s.find('_') for s in acc.method]
    pos = [j if j != -1 else 0 for j in pos]
    acc['Imputer'] = [s[:j] for s, j in zip(acc.method, pos)]
    acc.Imputer = acc.Imputer.map({
        'knn': 'kNN', 'mean': 'Mean', 'br': 'Iterative BR',
        'rf': 'Iterative RF', '': ''})

    # Create a column predictor
    pos = [j+5 if j != 0 else 0 for j in pos]
    acc['predictor'] = [s[j:] for s, j in zip(acc.method, pos)]

    # Rename models with complete names
    acc.predictor = acc.predictor.map(d)

    # Plot boxplots
    ax.grid(axis='y')
    sns.boxplot(ax=ax, data=acc, y='test_accuracy', x='predictor',
                hue='Imputer')
    if not is_legend:
        ax.legend([], [], frameon=False)
    else:
        ax.legend(frameon=True, title='Imputer')
    ax.set_axisbelow(True)
    if 'balanced' in filename:
        ax.set_ylabel('Test balanced accuracy')
    else:
        ax.set_ylabel('Test accuracy')
    ax.set_xlabel('Predictor')

    if i == 0:
        ax.set_title('Majors $\emph{or}$ Traces dataset \n ' +
                     '(13,925 sample observations)')


# Plot majors_and_traces and majors_and_traces_restricted
for i, filename in enumerate(['test_accuracy', 'test_balanced_accuracy']):
    ax = axes[i, 1]
    is_legend = i == 0

    geochem = 'majors_and_traces'
    file_acc = f'../results/{geochem}/{filename}.csv'
    acc = pd.read_csv(file_acc, index_col=0)

    geochem_restricted = 'majors_and_traces_restricted'
    file_acc_restricted = f'../results/{geochem_restricted}/{filename}.csv'
    acc_r = pd.read_csv(file_acc_restricted, index_col=0)

    # Convert from wide to long format
    value_vars = list(acc.columns)
    acc['repeat'] = acc.index
    acc = pd.melt(acc, id_vars='repeat', value_vars=value_vars,
                  var_name='method', value_name='test_accuracy')

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
    df.predictor = df.predictor.map(d)

    # Plot boxplots
    ax.grid(axis='y')
    my_pal = {'majors and traces': 'dimgray', 'majors only': 'lightgray'}
    # my_pal = {'majors and traces': '#4878d0', 'majors only': 'gray'}
    sns.boxplot(ax=ax, data=df, y='test_accuracy', x='predictor',
                hue='Elements', palette=my_pal)
    if not is_legend:
        ax.legend([], [], frameon=False)
    else:
        ax.legend(frameon=True)
    ax.set_axisbelow(True)
    ax.locator_params(axis='both', nbins=10)
    if 'balanced' in filename:
        ax.set_ylabel('Test balanced accuracy')
    else:
        ax.set_ylabel('Test accuracy')
    ax.set_xlabel('Predictor')

    if i == 0:
        ax.set_title('Majors $\emph{and}$ Traces dataset \n' +
                     '(1,212 sample observations)')

plt.savefig('../figures/fig10.pdf', dpi=300, bbox_inches='tight')
plt.close()
plt.show()
