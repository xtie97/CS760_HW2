from sklearn import tree
import os 
import graphviz 
import random 
import numpy as np 
import math 
from DecisionTree import * 
import matplotlib.pylab as plt

if __name__ == '__main__':
    trainx, trainy = read_txt('./data/D3leaves.txt')
    clf = self_DecisionTreeClassifier() 
    clf.fit(trainx, trainy)
    predy = clf.predict(trainx)
    clf.show_tree()
    print('# of nodes: {}'.format(clf.count_node())) 
   
