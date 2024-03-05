# from StringIO import StringIO

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
# from sklearn.tree.export import export_graphviz
from sklearn.feature_selection import mutual_info_classif

X = [[1,0,0], [0,0,0], [0,0,1], [0,1,0]]

y = [1,0,1,1]

clf = DecisionTreeClassifier()
clf.fit(X, y)

feat_importance = clf.tree_.compute_feature_importances(normalize=False)
print("feat importance = " + str(feat_importance))

# out = StringIO()
# out = export_graphviz(clf, out_file='test/tree.dot')
