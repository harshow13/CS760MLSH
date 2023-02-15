import numpy as np
#import pandas as pd

##define node class
class Node:
    def __init__(self, feature, threshold, left, right, label):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.label = label

##entropy function
def entropy(y):
    if y.size ==0:
        return 0
    else:
        p1 = np.mean(y)##since y is binary, p1 is the probability of 1
        if p1 == 0 or p1 == 1:
            return 0
        else:
            return -p1 * np.log2(p1) - (1 - p1) * np.log2(1 - p1)

##information gain ratio function
def information_gain_ratio(X, y, feature, threshold):
    y1 = y[X[:, feature] >= threshold]
    y2 = y[X[:, feature] < threshold]
    p1 = len(y1) / len(y)
    p2 = len(y2) / len(y)
    info_gain =  entropy(y) - p1 * entropy(y1) - p2 * entropy(y2)
    if p1==0 or p1==1:
        split_entropy = 0
    else:
        split_entropy = -p1 * np.log2(p1) - p2 * np.log2(p2)

    if split_entropy == 0:
        return -1 ##if split_entropy is 0, then the candiate split cannot be used
    else:
        return info_gain / split_entropy

##find the best split
def find_best_split(X, y):
    best_feature = None
    best_threshold = None
    max_ig_ratio = 0
    for feature in range(X.shape[1]):##for each feature
        thresholds = np.unique(X[:, feature]) ## considering all unique values in the training data as potential split thresholds
        for threshold in thresholds:
            ig_ratio = information_gain_ratio(X, y, feature, threshold)
            if ig_ratio > max_ig_ratio:##taking the split that maximizes the information gain ratio
                max_ig_ratio = ig_ratio
                best_feature = feature
                best_threshold = threshold
    return best_feature, best_threshold, max_ig_ratio

##split the data
def data_split(X, y, feature, threshold):
    X1 = X[X[:, feature] >= threshold]
    X2 = X[X[:, feature] < threshold]
    y1 = y[X[:, feature] >= threshold]
    y2 = y[X[:, feature] < threshold]
    return X1, y1, X2, y2

##build the tree
def tree_build(X, y):
    feature, threshold, ig_ratio = find_best_split(X, y) ##finding the best split
    #checking the stopping criteria
    if len(y)==0 or ig_ratio ==0:##if node is empty or the information gain ratio is 0 (this also checks if ALL the candidate splits entropy are zero), then the node is a leaf node
        if np.mean(y) >= 0.5:
            label = 1
        else:
            label = 0
        return Node(feature = None, threshold = None, left = None, right = None, label = label)
    else:##recursively build the left and right subtrees and return the node
        X1, y1, X2, y2 = data_split(X, y, feature, threshold)
        left = tree_build(X1, y1)##since xi>=c is the left branch
        right = tree_build(X2, y2)
        return Node(feature = feature, threshold = threshold, left = left, right = right, label = None)

##predict the label (individually for each data point)
def predict(tree, x):
    if tree.label is not None:
        return tree.label
    if x[tree.feature] >= tree.threshold:
        return predict(tree.left, x)
    else:
        return predict(tree.right, x)

##reading text file with space delimeter and labels in last column to numpy array
def read_data(file_path):
    data = np.loadtxt(file_path, delimiter=' ')
    X = data[:, :-1]
    y = data[:, -1]
    return X, y 

##main testing function
def main_test(trainfile, testfile):
    X_train, y_train = read_data(trainfile)
    X_test, y_test = read_data(testfile)
    tree = tree_build(X_train, y_train)
    y_pred = np.array([predict(tree, x) for x in X_test])
    accuracy = np.mean(y_pred == y_test)
    return accuracy

##plotting the tree
def plot_tree(tree, spacing=""):
    if tree.label is not None:
        print (spacing + "Predict y =", tree.label)
        return
    print (spacing + "Is X" + str(tree.feature) + " >= " + str(tree.threshold) + "? ")
    print (spacing + '--> True:')
    plot_tree(tree.left, spacing + "   ")
    print (spacing + '--> False:')
    plot_tree(tree.right, spacing + "   ")

##count number of nodes in a tree
def count_nodes(tree):
    if tree.label is not None:
        return 1
    return 1 + count_nodes(tree.left) + count_nodes(tree.right)

##calcuate error
def calc_error(tree, X, y):
    y_pred = np.array([predict(tree, x) for x in X])
    accuracy = np.mean(y_pred == y)
    error = 1 - accuracy
    return error