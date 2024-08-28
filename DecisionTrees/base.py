from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from utils import *

np.random.seed(42)


@dataclass
class DecisionTree:
    # criterion won't be used for regression
    criterion: Literal["information_gain", "gini_index"]
    max_depth: int  # The maximum depth the tree can grow to

    def __init__(self, criterion, max_depth):
        self.criterion = criterion
        self.max_depth = max_depth
        self.root = None
        self.is_real = None

    @dataclass
    class DecisionNode:
        def __init__(self, attribute=None, value=None, result=None, threshold=None, left=None, right=None, children=None):
            self.attribute = attribute
            self.value = value
            self.result = result
            self.threshold = threshold
            self.left = left
            self.right = right
            self.children = children or {}

    def make_tree(self, X: pd.DataFrame, y: pd.Series, depth) -> 'DecisionNode':
        if y.nunique() == 1 or X.shape[1] == 1 or depth > self.max_depth:
            if self.is_real['label']:  # real output
                branch_result = y.mean()
            else:
                branch_result = y.value_counts().idxmax()  # discrete output
            return self.DecisionNode(result=branch_result)

        if list(self.is_real.values())[0]:  # real input
            best_attr, mean = opt_split_attributeR(
                X, y, self.criterion, X.columns, self.is_real)
            node = self.DecisionNode(attribute=best_attr)
            X_left, X_right, y_left, y_right = split_dataR(
                X, y, best_attr, mean)
            node.left = self.make_tree(X_left, y_left, depth + 1)
            node.right = self.make_tree(X_right, y_right, depth + 1)
            node.threshold = mean
            return node

        else:  # discrete input
            best_attr = opt_split_attributeD(
                X, y, self.criterion, X.columns, self.is_real)
            node = self.DecisionNode(attribute=best_attr)
            for value in X[best_attr].unique():
                new_X, new_Y = split_dataD(X, y, best_attr, value)
                child_node = self.make_tree(new_X, new_Y, depth + 1)
                node.children[value] = child_node
            return node

    def predict(self, X: pd.DataFrame) -> pd.Series:
  
        result = []
        for _, row in X.iterrows():
            node = self.root
            while node.result is None:
                if list(self.is_real.values())[0]:
                    value = row[node.attribute]
                    if value < node.threshold:
                        node = node.left
                    else:
                        node = node.right
                else:
                    value = row[node.attribute]
                    # Ensure consistent typing between training and prediction
                    if isinstance(next(iter(node.children)), np.int64):
                        value = np.int64(value)
                    elif isinstance(next(iter(node.children)), int):
                        value = int(value)
                    elif isinstance(next(iter(node.children)), float):
                        value = float(value)
                    elif isinstance(next(iter(node.children)), str):
                        value = str(value)
                    node = node.children.get(value, None)
                    if node is None:
                        # Handle missing value in the children, possibly a case not seen during training
                        result.append(np.nan)  # or some other default value
                        break
            if node is not None and node.result is not None:
                result.append(node.result)
        return pd.Series(result)


    def plot(self, node=None, indent="") -> None:
        """
        Function to plot the tree
        """
        if node is None:
            node = self.root

        if node.result is not None:
            print(indent + "Result:", node.result)
        else:
            if list(self.is_real.values())[0]:
                print(indent + f"If {node.attribute} < {node.threshold}")
                if node.left:
                    self.plot(node.left, indent + "    ")
                print(indent + f"If {node.attribute} > {node.threshold}")
                if node.right:
                    self.plot(node.right, indent + "    ")
            else:
                for value, child_node in node.children.items():
                    print(indent + f"If {node.attribute} == {value}")
                    self.plot(child_node, indent + "    ")

    def fit(self, X: pd.DataFrame, y: pd.Series, depth=0) -> None:
        self.is_real = real_or_not(X, y)
        self.root = self.make_tree(X, y, 0)
        return self.root



