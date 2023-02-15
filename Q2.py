from sklearn import tree
import os 
import graphviz 
import random 
import numpy as np 
import math 
from DecisionTree import * 
import matplotlib.pylab as plt

if __name__ == '__main__':
    trainx = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])
    trainx1 = trainx[:, 0]
    trainx2 = trainx[:, 1]
    trainy = np.array([0, 0, 1, 1]) 
    plt.scatter(trainx1[trainy==0], trainx2[trainy==0], label='Class: 0')
    plt.scatter(trainx1[trainy==1], trainx2[trainy==1], label='Class: 1')
    plt.legend()
    plt.savefig('./images/Q2_data.png')
    plt.close()
    
    clf = self_DecisionTreeClassifier() 
    clf.fit(trainx, trainy)
    predy = clf.predict(trainx)
    clf.show_tree()
    print('# of nodes: {}'.format(clf.count_node())) 
   
