#Decision Tree Implementation and Analysis

#Overview

This repository contains the implementation and analysis of a custom Decision Tree algorithm as part of an academic assignment. The tasks involve building a Decision Tree from scratch, running experiments to evaluate its performance, and comparing it with scikit-learn's Decision Tree implementation. The repository also includes performance metrics, visualization, and timing analysis.

#File Descriptions

tree/

base.py: Contains the implementation of the custom Decision Tree class, supporting both discrete and real features with various splitting criteria.

utils.py: Includes utility functions needed for the Decision Tree operations.

__init__.py: Initialization file for the Decision Tree module (do not edit).

metrics.py: Implements performance metrics functions used to evaluate the Decision Tree's accuracy, precision, and recall.

usage.py: A script to test the Decision Tree implementation on a generated dataset. It splits the dataset into training (70%) and testing (30%) sets and outputs performance metrics.

classification-exp.py: Runs experiments using the Decision Tree on the generated dataset, including 5-fold cross-validation and nested cross-validation to determine the optimal depth of the tree.

auto-efficiency.py: Applies the Decision Tree to the automotive efficiency problem and compares the performance with the scikit-learn Decision Tree.

experiments.py: Contains code to generate synthetic datasets and perform runtime complexity analysis for tree learning and prediction. The results are compared against the theoretical
time complexity for different types of Decision Trees.

Asst#<task-name>_<Q#>.md: Markdown files answering subjective questions, including visualizations, timing analysis, and plot displays.

Decision Tree graphs.pdf: A PDF containing graphs related to timing analysis and complexity experiments (Q4).
