import numpy as np
import matplotlib.pyplot as plt

from math import exp, log
import os
import pickle

from skimage import io
from skimage.color import rgb2gray
from skimage.feature import hog

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from copy import deepcopy
import pandas as pd

IMAGE_DIR = "./data/"

def load_dataset(subset="train"):
    """
    1. subset = "train", "val", "test"
    2. About the dataset: in "train","val" subset, the first half of images are images of hands, the rest half are images of not-hand. 
    3. extract features from HoG
    """
    path = os.path.join(IMAGE_DIR, subset)
    name_list = os.listdir(path)
    print("Number of images in {}-set: {}".format(subset, len(name_list)))
    #HoG returns 324 features
    X = np.zeros(shape=(len(name_list), 324))

    if subset == "train" or subset == "val":
        #Make sure that we have equal number of positive and negative class images
        assert len(name_list)%2 == 0
        count = len(name_list)//2
        y = np.array(count*[1] + count*[0])
        for idx_true in range(count):
            img_name = os.path.join(path,str(idx_true)+".png")
            img = io.imread(img_name)
            img = rgb2gray(img)
            vec = hog(img)
            X[idx_true, :] = vec
        
        for idx in range(count):
            idx_false = idx + count
            img_name = os.path.join(path,str(idx_false)+".png")
            img = io.imread(img_name)
            img = rgb2gray(img)
            vec = hog(img)
            X[idx_false, :] = vec
        return X, y        
    else:
        for idx in range(len(name_list)):
            img_name = os.path.join(path, str(idx)+".png")
            img = io.imread(img_name)
            img = rgb2gray(img)
            vec = hog(img)
            X[idx, :] = vec
        return X


class LogisticRegression:
    """
    Logistic Regression
    """
    def __init__(self, eta0=0.1, eta1=1, m=64, max_epoch=1000, delta=0.0001):
        """
        m is the batch_size
        """
        self.__init = True # whether to initial the parameters
        self.__eta0 = eta0
        self.__eta1 = eta1
        self.__delta = delta
        self.__m = m
        self.__max_epoch = max_epoch
        
    def sigmoid(self, x):
        return 1.0 / (1 + exp(-x))
    
    def __init_param(self):
        """
        Weights initialized using a normal distribution here: you can change the distribution.
        """
        d = self.__dimension
        self.__wt = np.random.randn(1, d)
        self.__bias = np.random.randn()
        return self.__wt, self.__bias


    def visual_wt(self):
        '''visualize the weights
        '''
        pd.DataFrame(self.__wt[0]).hist(bins=15)


    def create_batch(self, n, m):
        '''Create batches of size m or m+1
        Parameters:
            n (int): the size of data set
            m (int): the size of the batch can be either m or m+1
        Return:
            (list): a list of batches, where each batch is a list
        '''
        permute = np.random.permutation(n)
        num_m_1 = n%m            # this number of batches have size of m+1
        num_m = n//m - num_m_1   # this number of batches have size of m
        batches = []
        index = 0
        for _ in range(num_m):
            batch = []
            for _ in range(m):
                batch.append(permute[index])
                index += 1
            batches.append(batch)
        for _ in range(num_m_1):
            batch = []
            for _ in range(m+1):
                batch.append(permute[index])
                index += 1
            batches.append(batch)
        return batches


    def loss_function(self, X_bar, y, thetaT):
        '''The loss function that we want to optimize
        Parameters:
            X_bar (ndarray, n=2): the training dataset
            thetaT (ndarray, n=1): the weights
        Return:
            (float)
        '''
        summation = 0
        for Xi, yi in zip(X_bar, y):
            summation += yi * thetaT @ Xi - log(1 + exp(thetaT @ Xi))
        return -summation/len(X_bar)


    def fit(self, X, y, X_val=None, y_val=None):
        """
        Recommended input:
        X: n x d array,
        y: n x 1 array or list
        """
        n, d = X.shape                              # X.shape = (8170, 324)
        self.__dimension = d

        if self.__init:
            self.__init_param()

        ### write your code here ###
        ones = np.array(np.ones(n))
        ones = np.expand_dims(ones, axis=0)
        X_bar = np.concatenate((X, ones.T), axis=1)  # X_bar.shape  = (8170, 325), i.e already transposed
        thetaT = np.append(self.__wt[0], self.__bias)          # thetaT.shape = (325,)

        for epoch in range(1, self.__max_epoch+1):
            eta = self.__eta0/(self.__eta1 + epoch)  # eta decreases
            batches = self.create_batch(n, self.__m)
            new_thetaT = deepcopy(thetaT)
            for batch in batches:                    # batch is a list of random index
                derivative = np.zeros(len(thetaT))
                for i in range(len(batch)):
                    Xi = X_bar[batch[i]]
                    yi = y[batch[i]]
                    prob = self.sigmoid(new_thetaT @ Xi)
                    derivative += (yi - prob) * Xi
                derivative = -derivative/len(batch)             # take the average and times -1
                new_thetaT = new_thetaT - eta*derivative          # update thetaT
            loss_old = self.loss_function(X_bar, y, thetaT)
            loss_new = self.loss_function(X_bar, y, new_thetaT)
            print(epoch, loss_old, loss_new, loss_new/loss_old)
            thetaT = deepcopy(new_thetaT)

            if loss_new > (1 - self.__delta)*loss_old:   # terminate condition
                self.__wt = thetaT[0:d]
                self.__bias = thetaT[d]
                break
            

    def get_thetaT(self):
        return self.__thetaT


    def predict_proba(self, X):
        """Predict the probility of the sample, 
           i.e. probability of the samples belonging to the positive class
        Attributes:
            X (ndarray, n=2): samples, for SGD, X is a batch (subset) of samples.
        Return:
            (list)
        """
        proba = []
        for Xi in X:
            proba.append(self.sigmoid(np.exp(self.__thetaT @ Xi)))
        return proba


    def predict(self, X):
        """
        Classify the sample
        """
        # return self.predict_proba(X) >= 0.5 # attention: this will give result in bool, you need to convert it to int for submission. 
    
    def get_param(self):
        """
        output:
            parameters: wt(1*d array), b(scalar)
        """
        return [self.__wt, self.__bias]
    
    def save_model(self, save_file):
        """
        save model to .pkl file
        """
        with open(save_file,"wb") as file:
            pickle.dump([self.__wt, self.__bias], file)

    def load_model(self, load_file):
        """
        load model from .pkl file
        """
        with open(load_file,"rb") as file:
            param = pickle.load(file)
        self.__wt = param[0]
        self.__bias = param[1]
        self.__init = False
        return self.__wt, self.__bias


def main():
    X_train, y_train = load_dataset("train")
    lr = LogisticRegression()
    lr.fit(X_train, y_train)


if __name__ == '__main__':
    main()
