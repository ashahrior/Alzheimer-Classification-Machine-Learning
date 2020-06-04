from sklearn.tree import DecisionTreeClassifier

criterion = ['gini', 'entropy']
splitter = ['best', 'random']
max_features = [1, 1.0, 'auto', 'sqrt', 'log2', None]
class_weight = ['balanced', None]

parameters_list = [criterion, splitter, max_features, class_weight]

#combo_list = list(itertools.product(*parameters_list))
# there are total 48 combinations originally

combos = [
    ['gini', 'best', 1, 'balanced'], ['gini', 'best', 1, None], ['gini', 'best', 1.0, 'balanced'], ['gini', 'best', 1.0, None], ['gini', 'best', 'auto', 'balanced'], ['gini', 'best', 'auto', None], ['gini', 'best', 'sqrt', 'balanced'], ['gini', 'best', 'sqrt', None], ['gini', 'best', 'log2', 'balanced'], ['gini', 'best', 'log2', None], ['gini', 'best', None, 'balanced'], ['gini', 'best', None, None], ['gini', 'random', 1, 'balanced'], ['gini', 'random', 1, None], ['gini', 'random', 1.0, 'balanced'], ['gini', 'random', 1.0, None], ['gini', 'random', 'auto', 'balanced'], ['gini', 'random', 'auto', None], ['gini', 'random', 'sqrt', 'balanced'], ['gini', 'random', 'sqrt', None], ['gini', 'random', 'log2', 'balanced'], ['gini', 'random', 'log2', None], ['gini', 'random', None, 'balanced'], ['gini', 'random', None, None], ['entropy', 'best', 1, 'balanced'], ['entropy', 'best', 1, None], ['entropy', 'best', 1.0, 'balanced'], ['entropy', 'best', 1.0, None], ['entropy', 'best', 'auto', 'balanced'], ['entropy', 'best', 'auto', None], ['entropy', 'best', 'sqrt', 'balanced'], ['entropy', 'best', 'sqrt', None], ['entropy', 'best', 'log2', 'balanced'], ['entropy', 'best', 'log2', None], ['entropy', 'best', None, 'balanced'], ['entropy', 'best', None, None], ['entropy', 'random', 1, 'balanced'], ['entropy', 'random', 1, None], ['entropy', 'random', 1.0, 'balanced'], ['entropy', 'random', 1.0, None], ['entropy', 'random', 'auto', 'balanced'], ['entropy', 'random', 'auto', None], ['entropy', 'random', 'sqrt', 'balanced'], ['entropy', 'random', 'sqrt', None], ['entropy', 'random', 'log2', 'balanced'], ['entropy', 'random', 'log2', None], ['entropy', 'random', None, 'balanced'], ['entropy', 'random', None, None]
    ]

headers = ['CRITERION','SPLITTER','MAX_FEATURES','CLASS_WEIGHT']