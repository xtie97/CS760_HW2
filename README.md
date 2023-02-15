# CS760_HW2: Decision Tree 
## Only the binary classification and continuous attributes are considered. Hopefully, a more general version will come up in the future. 

*Question 2.2: 
'''
python Q2.py
'''

Question 2.3: python Q3.py

Question 2.4: python Q4.py

Question 2.5: python Q5_D1.py & python Q5_D2.py

Question 2.6: python Q6.py

Question 2.7: python Q7.py 
To reproduce the learning curve, please replace the default "thress = x[np.where(y_dif > 0.5)[0]+1]" at line 76 in "DecisionTree.py" with "thress = (x[np.where(y_dif > 0.5)[0]]+x[np.where(y_dif > 0.5)[0]+1])/2". Otherwise, the test set error is slightly worse.

Question 3 (sklearn): python Q7.py 

Question 4 (Lagrange interpolation): python lagrange.py 

All data files are stored in the "data" folder and the generated images are saved in the "images" folder. 
