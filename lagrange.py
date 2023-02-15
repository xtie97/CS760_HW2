import numpy as np
from scipy.interpolate import lagrange
import math 
from numpy.polynomial.polynomial import Polynomial
import matplotlib.pylab as plt

def lagrange_interp(a, b, num_train):
    np.random.seed(716)
    #num_train = 20  # 16, 18 | 20, 22, 30
    trainx = np.random.rand(100)
    num_test = 100 
    
    trainx = trainx[:num_train] * (b-a) + a 
    trainx = np.sort(trainx) 
    trainy = np.sin(trainx) 
    poly = lagrange(trainx, trainy)
    predy = Polynomial(poly.coef[::-1])(trainx)
    train_error = np.sqrt(np.mean((trainy - predy)**2)) 
    
    testx = np.random.rand(num_test) * (b-a) + a 
    testx = np.sort(testx)
    testy = np.sin(testx) 
    predy = Polynomial(poly.coef[::-1])(testx)
    test_error = np.sqrt(np.mean((testy - predy)**2)) 
    
    plt.scatter(testx, testy, label='data', c='r')
    plt.plot(testx, predy, label='Lagrange interpolation')
    plt.xlabel('x')
    plt.ylabel('y') 
    plt.legend()
    plt.savefig('./images/lagrange_num_train_{}.png'.format(num_train))
    plt.close()
    return trainx, trainy, testx, testy, train_error, test_error
     

if "__main__" == __name__ :
    train_error_list = []
    test_error_list = []
    for num_train in [18, 20, 22]:
        _, _, _, _, train_error, test_error = lagrange_interp(0, 10, num_train)
        train_error_list.append(train_error)
        test_error_list.append(test_error) 
    plt.plot([18, 20, 22], train_error_list, 'o--', label='train set error') 
    plt.plot([18, 20, 22], test_error_list, '^--', label='test set error') 
    print('Train set errors: {}'.format(np.around(train_error_list, 3)))
    print('Test set errors: {}'.format(np.around(test_error_list, 3)))
    plt.xlabel('Number of samples used for Lagrange Interpolation')
    plt.ylabel('Error')
    plt.legend() 
    plt.savefig('./images/lagrange_error.png')
    plt.close()
    
    sigma_list = np.array([1e-5, 1e-4, 1e-3, 1e-2])
    trainx, trainy, testx, testy, train_error, test_error = lagrange_interp(0, 10, 18)
    train_error_list = []
    test_error_list = []
    for sigma in sigma_list:
        trainx_noise = trainx + np.random.normal(scale=sigma, size=trainx.shape)
        poly = lagrange(trainx_noise, trainy)
        predy = Polynomial(poly.coef[::-1])(trainx)
        train_error = np.sqrt(np.mean((trainy - predy)**2)) 
        
        predy = Polynomial(poly.coef[::-1])(testx)
        test_error = np.sqrt(np.mean((testy - predy)**2)) 
        
        train_error_list.append(train_error)
        test_error_list.append(test_error) 
         
    plt.semilogx(sigma_list, train_error_list, 'o--', label='train set error') 
    plt.semilogx(sigma_list, test_error_list, '^--', label='test set error') 
    plt.rcParams.update({'font.size': 12})
    plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=12)  
    plt.xlabel('Standard Deviation of the zero-mean Gaussian noise')
    plt.ylabel('Error')
    plt.legend() 
    plt.savefig('./images/lagrange_sigma_error.png')
    plt.close()
