from sklearn import tree
import os 
import graphviz 
import random 
import numpy as np 
import math
from copy import deepcopy

def read_txt(txt_file: str):
    x = [] 
    y = []
    # read features and labels line by line, then output two lists
    with open(txt_file, "r") as data:
        lines = data.readlines()
        for line in lines:
            x.append([float(line.split(' ')[0]), float(line.split(' ')[1])])
            y.append(int(line.split()[2]))
    return x, y

def nested_sample(txt_file: str, num_train: int):
    # sample "num_train" training items 
    x, y = read_txt(txt_file)
    z = list(zip(x, y))
    random.seed(1)
    random.shuffle(z)
    x, y = zip(*z)
    return x[:num_train], y[:num_train], x[num_train:], y[num_train:]    

class treenode:
    def __init__(self, feat, thres, output, level):
        # feat: which feature is used to split the node
        # thres: what threshold is chosen
        # output: the class with majority at this instance of the decision tree
        self.feat = feat
        self.thres = thres 
        self.output = output
        self.level = level 
        # next node are stored as a dicticionary.
        # 0: >= thres, 1: < thres 
        self.nextnode = {}
        
    def add_next(self, splitid, obj):
        self.nextnode[splitid] = obj # obj: next treenode 
        
class self_DecisionTreeClassifier:
    def __init__(self):
        self.root = None
        
    def _compute_prob(self, y):
        # compute the frequency 
        y_classes = np.sort(np.unique(np.array(y))) 
        prob = [] 
        for y_class in y_classes:
            prob.append(np.sum(np.array(y)==y_class)/len(y))
        return prob

    def _entropy(self, y):
        # returns the entropy 
        prob = self._compute_prob(y)
        entropy = 0
        for p in prob:
            entropy += (-p)*math.log2(p)
        return entropy

    def _gain_ratio(self, x, y, feat_ind):
        # returns the gain ratio
        # mutual information = H(y) - H(y|x) = -plog(p) + p(x,y)log(p(y|x))
        # H(x) = -p(x)log(p(x)) 
        Hy = self._entropy(y) # entropy before splitting
        x = x[:, feat_ind] 
        x_ind = np.argsort(x)
        x = x[x_ind]
        y = y[x_ind]
        y_dif = np.abs(np.diff(y))        
        #thress = (x[np.where(y_dif > 0.5)[0]]+x[np.where(y_dif > 0.5)[0]+1])/2
        thress = x[np.where(y_dif > 0.5)[0]+1] 
        thress = np.unique(thress)
        infogan_max = 0
        select_thres = 0
        for thres in thress: 
            y1 = y[x>=thres]
            y2 = y[x<thres]
            Hy1 = self._entropy(y1) 
            Hy2 = self._entropy(y2) 
            Hyx = Hy1 * y1.shape[0]/y.shape[0] + Hy2 * y2.shape[0]/y.shape[0] 
            infogan = Hy - Hyx  # entropy after splitting upon the selected feature
            if infogan > infogan_max:
                infogan_max = infogan
                select_thres = thres 
                
        y1 = y[x>=select_thres]
        p = y1.shape[0]/y.shape[0]
        if infogan_max == 0:
            return 0, select_thres
        elif p==0 or p==1:
            return math.inf, select_thres
        else:
            Hx = (-p)*math.log2(p) -(1-p)*math.log2(1-p) 
            return infogan_max/Hx, select_thres
         
    def _decision_tree(self, x, y, level):
        # currently, the decision tree handles continuous attributes and binary classification
        x = np.array(x) 
        y = np.array(y)
        
        # If the node consists of only 1 class
        if len(np.unique(y)) == 1:
            output = np.unique(y)[0]
            return treenode(None, None, output, level) # reached a leaf node
        
        # Finding the best feature to split upon
        feats = np.arange(x.shape[1]) 
        max_gain = - math.inf
        select_feat = None
        select_thres = None 
        for feat in feats:
            current_gain, current_thres = self._gain_ratio(x, y, feat)
            if current_gain == math.inf:
                continue
            elif current_gain > max_gain:
                max_gain = current_gain
                select_feat = feat
                select_thres = current_thres
        if  max_gain == 0:
            return treenode(None, None, 0, level) # infogan = 0 
        prob = self._compute_prob(y)
        output = np.argmax(prob) 
        current_node = treenode(select_feat, select_thres, output, level)
        node1 = self._decision_tree(x[x[:, select_feat]>=select_thres,:], y[x[:, select_feat]>=select_thres], level+1)
        current_node.add_next(0, node1)
        node2 = self._decision_tree(x[x[:, select_feat]<select_thres,:], y[x[:, select_feat]<select_thres], level+1)
        current_node.add_next(1, node2)
        
        return current_node
    
    def fit(self, X, Y):
        # fit to the given training data
        self.root = self._decision_tree(X, Y, level=0)
        
    def _predict(self, data, node):
        # predicts the class for a given testing point and returns the answer
        # reached the leaf node
        if len(node.nextnode) == 0 :
            return node.output
        
        if data[node.feat] >= node.thres:
            splitid = 0
        else:
            splitid = 1 
             
        # Recursively call on the splits
        return self._predict(data, node.nextnode[splitid])    

    def predict(self, X):
        # This function returns Y predicted
        # X should be a 2-D np array
        X = np.array(X)
        Y = np.zeros((X.shape[0], 1), dtype='int8') 
        for i in range(X.shape[0]): 
            Y[i] = self._predict(X[i,], self.root)
        return Y
    
    def count_node(self):
        r = self.root
        count = 1
        node_list = [r.nextnode] 
        node_list_copy = deepcopy(node_list)
        while len(node_list) > 0:
            for node in node_list:
                for i in node:
                    count += 1 
                    if len(node[i].nextnode) != 0:
                        node_list_copy.append(node[i].nextnode)
                node_list_copy.pop(0)
            node_list = deepcopy(node_list_copy) 
        return count 
        
    def show_tree(self):
        r = self.root
        print("\n [Level: {}, Candidate split: X[{}] >= {}]".format(r.level, r.feat, r.thres))  
        node_list = [r.nextnode] 
        node_list_copy = deepcopy(node_list)
        truefalse = ['True', 'False']
        while len(node_list) > 0:
            for node in node_list:
                for i in node:
                    if node[i].feat == None:
                        print( "\n [Level: {}, Answer to the parent node: {}, Output: {}]".format(node[i].level, truefalse[i], node[i].output)) 
                    else:
                        print("\n [Level: {}, Answer to the parent node: {}, Candidate split: X[{}] >= {}]".format(node[i].level, truefalse[i], node[i].feat, node[i].thres))
                    
                    if len(node[i].nextnode) != 0:
                        node_list_copy.append(node[i].nextnode)
                node_list_copy.pop(0)
            node_list = deepcopy(node_list_copy) 
            
                   
              
