from sklearn import tree
import os 
import graphviz 
import random 
import numpy as np 
import math 
from DecisionTree import * 
import matplotlib.pylab as plt

def compute_error(predy, testy):
    error = np.array(predy).reshape(-1,1) - np.array(testy).reshape(-1,1)
    error = np.sum(np.abs(error)>0)/len(testy)
    return error 
        
if __name__ == '__main__':
    # dataset: "D1.txt" 
    trainx, trainy, testx, testy = nested_sample('./data/Dbig.txt', 8092) 
    trainx = np.array(trainx)
    trainy = np.array(trainy) 
    num_train_list = [32, 128, 512, 2048, 8092]  
    
    # self-implemented 
    error_list = []
    for num_train in num_train_list:
        clf = self_DecisionTreeClassifier() 
        clf.fit(trainx[:num_train,:], trainy[:num_train])
        n_nodes = clf.count_node()
        predy = clf.predict(trainx[:num_train,:])
        error = compute_error(predy, trainy[:num_train])
        print('# of training samples: {}, # of nodes: {}, Train set error: {:.2f}%\n'.format(num_train, n_nodes, error*100))
        
        predy = clf.predict(testx)
        error = compute_error(predy, testy)
        print('# of training samples: {}, # of nodes: {}, Test set error: {:.2f}%\n'.format(num_train, n_nodes, error*100))
        error_list.append(error*100) 
         
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
        plt.savefig('./images/decision_boundary_{}.png'.format(num_train))
        plt.close()
        
    plt.semilogx(num_train_list, error_list, '^--')
    plt.xlabel('Number of training samples')
    plt.ylabel('Test set error (%)')
    plt.savefig('./images/learning_curve_self_implemented.png')
    plt.close()     
  
    # sklearn  
    error_list = []
    for num_train in num_train_list: 
        clf = tree.DecisionTreeClassifier(criterion='entropy', random_state=1)
        clf.fit(trainx[:num_train,:], trainy[:num_train])
        #dot_data = tree.export_graphviz(clf, out_file=None, filled=True,\
        #                            rounded=True, special_characters=True)
        #graph = graphviz.Source(dot_data) 
        #graph.render("Q7") 
        n_nodes = clf.tree_.node_count
        
        predy = clf.predict(trainx[:num_train,:])
        error = compute_error(predy, trainy[:num_train])
        print('# of training samples: {}, # of nodes: {}, Train set error: {:.2f}%\n'.format(num_train, n_nodes, error*100))
        
        predy = clf.predict(testx)
        error = compute_error(predy, testy)
        print('# of training samples: {}, # of nodes: {}, Test set error: {:.2f}%\n'.format(num_train, n_nodes, error*100))
        error_list.append(error*100) 
        
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
        plt.savefig('./images/decision_boundary_{}_sklearn.png'.format(num_train))
        plt.close()
        
    plt.semilogx(num_train_list, error_list, '^--')
    plt.xlabel('Number of training samples')
    plt.ylabel('Test set error (%)')
    plt.savefig('./images/learning_curve_sklearn.png')
    plt.close()   
    
