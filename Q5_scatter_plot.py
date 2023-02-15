from sklearn import tree
import os 
import graphviz 
import random 
import numpy as np 
import math 
from DecisionTree import * 
import matplotlib.pylab as plt

if __name__ == '__main__':
    # dataset: "D1.txt" 
    trainx, trainy = read_txt('./data/D1.txt')
    trainx = np.array(trainx)
    trainy = np.array(trainy) 
    plt.scatter(trainx[trainy==0, 0], trainx[trainy==0, 1], label='Class: 0')
    plt.scatter(trainx[trainy==1, 0], trainx[trainy==1, 1], label='Class: 1')
    plt.legend()
    plt.savefig('./images/D1_scatter.png')
    plt.close()     
    
    clf = self_DecisionTreeClassifier() 
    clf.fit(trainx, trainy)
    num_samp = 500 
    testx1 = np.linspace(trainx[:,0].min(), trainx[:,0].max(), num_samp)
    testx2 = np.linspace(trainx[:,1].min(), trainx[:,1].max(), num_samp)
    testx1, testx2 = np.meshgrid(testx1, testx2)
    testx1 = testx1.reshape(-1, 1) 
    testx2 = testx2.reshape(-1,1)
    predy = clf.predict( np.concatenate((testx1, testx2), 1) )
    plt.scatter(testx1[predy==0], testx2[predy==0], label='Class: 0')
    plt.scatter(testx1[predy==1], testx2[predy==1], label='Class: 1')
    plt.legend()
    plt.savefig('./images/D1_decision_boundary.png')
    plt.close()
    
    # dataset: "D2.txt" 
    trainx, trainy = read_txt('./data/D2.txt')
    trainx = np.array(trainx)
    trainy = np.array(trainy) 
    plt.scatter(trainx[trainy==0, 0], trainx[trainy==0, 1], label='Class: 0')
    plt.scatter(trainx[trainy==1, 0], trainx[trainy==1, 1], label='Class: 1')
    plt.legend()
    plt.savefig('./images/D2_scatter.png')
    plt.close()   
    
    clf = self_DecisionTreeClassifier() 
    clf.fit(trainx, trainy)
    testx1 = np.linspace(trainx[:,0].min(), trainx[:,0].max(), num_samp)
    testx2 = np.linspace(trainx[:,1].min(), trainx[:,1].max(), num_samp)
    testx1, testx2 = np.meshgrid(testx1, testx2)
    testx1 = testx1.reshape(-1, 1) 
    testx2 = testx2.reshape(-1,1)
    predy = clf.predict( np.concatenate((testx1, testx2), 1) )
    plt.scatter(testx1[predy==0], testx2[predy==0], label='Class: 0')
    plt.scatter(testx1[predy==1], testx2[predy==1], label='Class: 1')
    plt.legend()
    plt.savefig('./images/D2_decision_boundary.png')
    plt.close()
    
    
    
