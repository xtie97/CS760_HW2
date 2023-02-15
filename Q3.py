from sklearn import tree
import os 
import graphviz 
import random 
import numpy as np 
import math 
from DecisionTree import read_txt

def compute_prob(y):
    # compute the frequency 
    y_classes = np.sort(np.unique(np.array(y))) 
    prob = [] 
    for y_class in y_classes:
        prob.append(np.sum(np.array(y)==y_class)/len(y))
    return prob

def entropy(y):
    # returns the entropy 
    prob = compute_prob(y)
    entropy = 0
    for p in prob:
        entropy += (-p)*math.log2(p)
    return entropy

def gain_ratio(x, y, feat_ind):
    # returns the gain ratio
    # mutual information = H(y) - H(y|x) = -plog(p) + p(x,y)log(p(y|x))
    # H(x) = -p(x)log(p(x)) 
    Hy = entropy(y) # entropy before splitting
    x = x[:, feat_ind] 
    x_ind = np.argsort(x)
    x = x[x_ind]
    y = y[x_ind]
    y_dif = np.abs(np.diff(y))        
    #thress = (x[np.where(y_dif > 0.5)[0]]+x[np.where(y_dif > 0.5)[0]+1])/2
    thress = x[np.where(y_dif > 0.5)[0]+1] 
    thress = np.unique(thress)
    infogan_ratio_list = {}
    infogan_list = {}
    for thres in thress: 
        y1 = y[x>=thres]
        y2 = y[x<thres]
        Hy1 = entropy(y1) 
        Hy2 = entropy(y2) 
        Hyx = Hy1 * y1.shape[0]/y.shape[0] + Hy2 * y2.shape[0]/y.shape[0] 
        infogan = Hy - Hyx  # entropy after splitting upon the selected feature
        
        y1 = y[x>=thres]
        p = y1.shape[0]/y.shape[0]
        if p==0 or p==1:
            infogan_list[thres] = infogan 
        else:
            Hx = (-p)*math.log2(p) -(1-p)*math.log2(1-p) 
            infogan_ratio_list[thres] = round(infogan / Hx, 3) 
    return infogan_ratio_list, infogan_list 

if __name__ == '__main__':
    trainx, trainy = read_txt('./data/Druns.txt')
    
    infogan_ratio_0, infogan_0 = gain_ratio(np.array(trainx), np.array(trainy), 0)
    infogan_ratio_1, infogan_1 = gain_ratio(np.array(trainx), np.array(trainy), 1) 
    print('Feature #1: InfoGan ratio: {}, InfoGan: {}\n'.format(infogan_ratio_0, infogan_0))
    print('Feature #2: InfoGan ratio: {}, InfoGan: {}\n'.format(infogan_ratio_1, infogan_1))

    
