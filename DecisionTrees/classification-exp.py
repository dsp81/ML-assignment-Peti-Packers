import numpy as np
import matplotlib.pyplot as plt
from base import DecisionTree
import metrics as me
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import pandas as pd

np.random.seed(42)

X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

X_df = pd.DataFrame(X, columns=['feature_1', 'feature_2'])
y_df = pd.Series(y, name='target')

plt.scatter(X_df['feature_1'], X_df['feature_2'], c=y_df)

X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.3, random_state=42)

decision_tree = DecisionTree(criterion="information_gain", max_depth=5)
decision_tree.fit(X_train, y_train)
y_pred = decision_tree.predict(X_test)

# Reset indices before calculating accuracy
y_pred.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)

accuracy = me.accuracy(y_pred, y_test)
precision = me.precision(y_pred, y_test, cls=1)
recall = me.recall(y_pred, y_test, cls=1)

print("Results for Q2a:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Per-class Precision: {precision:.2f}")
print(f"Per-class Recall: {recall:.2f}")
print("\n")

depths_to_evaluate = [3,4,5,6]
outer_folds = 5
outer_fold_size = len(X_df) // outer_folds

best_depths = []
best_accuracies = []

for i in range(outer_folds):
    test_start = i * outer_fold_size
    test_end = (i + 1) * outer_fold_size
    X_test_outer, y_test_outer = X_df.iloc[test_start:test_end], y_df.iloc[test_start:test_end]
    X_train_outer, y_train_outer = pd.concat([X_df.iloc[:test_start], X_df.iloc[test_end:]]), pd.concat([y_df.iloc[:test_start], y_df.iloc[test_end:]])

    fold_accuracies = []

    for depth in depths_to_evaluate:
        decision_tree = DecisionTree(criterion="information_gain", max_depth=depth)
        decision_tree.fit(X_train_outer, y_train_outer)

        y_pred_outer = decision_tree.predict(X_test_outer)

        #Reset indices before calculating accuracy
        y_pred_outer.reset_index(   drop=True, inplace=True)
        y_test_outer.reset_index(drop=True, inplace=True)

        accuracy_outer = me.accuracy(y_pred_outer, y_test_outer)
        fold_accuracies.append(accuracy_outer)

    best_depth_index = np.argmax(fold_accuracies)
    best_depth = depths_to_evaluate[best_depth_index]
    best_accuracy = fold_accuracies[best_depth_index]

    best_depths.append(best_depth)
    best_accuracies.append(best_accuracy)

overall_best_depth_index = np.argmax(best_accuracies)
overall_best_depth = best_depths[overall_best_depth_index]
overall_best_accuracy = best_accuracies[overall_best_depth_index]

print("Results for Q2b:")
print("Overall Best Depth:", overall_best_depth)
print("Overall Best Accuracy:", overall_best_accuracy)
