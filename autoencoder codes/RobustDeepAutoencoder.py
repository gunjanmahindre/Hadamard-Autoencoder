import numpy as np
import xgboost as xgb
import numpy.linalg as nplin
import tensorflow as tf
from BasicAutoencoder import DeepAE as DAE
from shrink import l1shrink as SHR 
import PIL.Image as Image
import ImShow as I
import random
import pandas as pd
import matplotlib.pyplot as plt
import os
from math import sqrt
from sklearn.metrics import mean_absolute_error
np.random.seed(123)
class RDAE(object):
    """
    @author: Chong Zhou
    2.0 version.
    complete: 10/17/2016
    version changes: move implementation from theano to tensorflow.
    3.0
    complete: 2/12/2018
    changes: delete unused parameter, move shrink function to other file
    update: 03/15/2019
        update to python3 
    Des:
        X = L + S
        L is a non-linearly low rank matrix and S is a sparse matrix.
        argmin ||L - Decoder(Encoder(L))|| + ||S||_1
        Use Alternating projection to train model
    """
    def __init__(self, sess, layers_sizes, lambda_=1.0, error = 1.0e-7):
        """
        sess: a Tensorflow tf.Session object
        layers_sizes: a list that contain the deep ae layer sizes, including the input layer
        lambda_: tuning the weight of l1 penalty of S
        error: converge criterior for jump out training iteration
        """
        self.lambda_ = lambda_
        self.layers_sizes = layers_sizes
        self.error = error
        self.errors=[]
        self.AE = DAE.Deep_Autoencoder( sess = sess, input_dim_list = self.layers_sizes)

    def fit(self,X, sess, learning_rate=0.15, inner_iteration = 50,
            iteration=20, batch_size=50, verbose=False):
        ## The first layer must be the input layer, so they should have same sizes.
        assert X.shape[1] == self.layers_sizes[0]

        ## initialize L, S, mu(shrinkage operator)
        self.L = np.zeros(X.shape)
        self.S = np.zeros(X.shape)
        self.hadamard_train = np.array(hadamard_train)
        self.cost = list()

        mu = (X.size) / (4.0 * nplin.norm(X,1))
        print ("shrink parameter:", self.lambda_ / mu)
        LS0 = self.L + self.S

        XFnorm = nplin.norm(X,'fro')
        if verbose:
            print ("X shape: ", X.shape)
            print ("L shape: ", self.L.shape)
            print ("S shape: ", self.S.shape)
            print ("mu: ", mu)
            print ("XFnorm: ", XFnorm)

        for it in range(iteration):
            if verbose:
                print ("Out iteration: " , it)
            ## alternating project, first project to L
            #self.L = X - self.S
            ## Using L to train the auto-encoder
            self.cost.append(self.AE.fit(X = X, sess = sess, S =self.S, h = self.hadamard_train,
                                    iteration = inner_iteration,
                                    learning_rate = learning_rate,
                                    batch_size = batch_size,
                                    verbose = verbose))
            ## get optmized L
            self.L = self.AE.getRecon(X = X, sess = sess)
            ## alternating project, now project to S
            self.S = SHR.shrink(self.lambda_/np.min([mu,np.sqrt(mu)]), (X - self.L).reshape(X.size)).reshape(X.shape)

            ## break criterion 1: the L and S are close enough to X
            c1 = nplin.norm(X - self.L - self.S, 'fro') / XFnorm
            ## break criterion 2: there is no changes for L and S 
            c2 = np.min([mu,np.sqrt(mu)]) * nplin.norm(LS0 - self.L - self.S) / XFnorm

            if verbose:
                print ("c1: ", c1)
                print ("c2: ", c2)

            if c1 < self.error and c2 < self.error :
                print ("early break")
                break
            ## save L + S for c2 check in the next iteration
            LS0 = self.L + self.S
            
        return self.L , self.S, self.cost
    
    def transform(self, X, sess):
        #L = X - self.S
        return self.AE.transform(X = X, sess = sess)
    
    def getRecon(self, X, sess):
        return self.AE.getRecon(X, sess = sess)

if __name__ == "__main__":
    data1 = pd.read_csv('./g1v_bb.csv')
    data1 = np.array(data1)
    data2 = pd.read_csv('./g2v_bb.csv')
    data2 = np.array(data2)
    data3 = pd.read_csv('./g3v_bb.csv')
    data3 = np.array(data3)
    data4 = pd.read_csv('./g4v_pl.csv')
    data4 = np.array(data4)
    data5 = pd.read_csv('./g5v_pl.csv')
    data5 = np.array(data5)
    data6 = pd.read_csv('./g6v_pl.csv')
    data6 = np.array(data6)
    data_test = pd.read_csv('./fb_test.csv')
    data_test_original = data_test.copy()
    data_test = np.array(data_test)
    [Rn, Cn] = data1.shape
    uT = []
    for i in range(Cn):
        for j in range(i + 1, Cn):
            uT.append(data1[i, j])
    fraction = 99
    # calculate entries to be deleted
    rem_num = ((len(uT)) * fraction / 100)  # total number of entries to be removed
    # has to be an integer value
    rem_num = int(rem_num)
    # select random elements from the upper triangle:
    ind = np.random.choice(len(uT), rem_num, replace=False)
    # make these indices -1
    for i in ind:
        uT[i] = 0  # now place these values back in the upper triangle:
    p = 0
    for i in range(Cn):
        for j in range(i + 1, Cn):
            data1[i, j] = uT[p]
            if data1[i, j] == 0:
                data1[j, i] = 0
            p += 1

    hadamard_train = np.ones(data1.shape)
    hadamard_train = np.where(data1 == 0, 0 , hadamard_train)
    hadamard_train = pd.DataFrame(hadamard_train)
    [R1, C1] = data_test.shape
    uT = []
    for i in range(C1):
        for j in range(i + 1, C1):
            uT.append(data_test[i, j])
    fraction = 99
    # calculate entries to be deleted
    rem_num = ((len(uT)) * fraction / 100)  # total number of entries to be removed
    # has to be an integer value
    rem_num = int(rem_num)
    # select random elements from the upper triangle:
    ind = np.random.choice(len(uT), rem_num, replace=False)
    # make these indices -1
    for i in ind:
        uT[i] = 0
    # now place these values back in the upper triangle:
    p = 0
    for i in range(C1):
        for j in range(i + 1, C1):
            data_test[i, j] = uT[p]
            if data_test[i, j] == 0:
                data_test[j, i] = 0
            p += 1
    hadamard_test = np.ones(data_test.shape)
    hadamard_test = np.where(data_test == 0,0, hadamard_test)
    hadamard_test = pd.DataFrame(hadamard_test)
    data_test = pd.DataFrame(data_test)
    data_test = np.array(data_test)

with tf.Session() as sess:

        rae = RDAE(sess = sess, lambda_= 500000, layers_sizes=[744,2])

        L, S, cost = rae.fit(data1 ,sess = sess, learning_rate=0.001, batch_size =1
                ,inner_iteration =10,iteration=1, verbose=True)

        [Rn, Cn] = data2.shape
        uT = []
        for i in range(Cn):
            for j in range(i + 1, Cn):
                uT.append(data2[i, j])
        fraction = 99
        # calculate entries to be deleted
        rem_num = ((len(uT)) * fraction / 100)  # total number of entries to be removed
        # has to be an integer value
        rem_num = int(rem_num)
        # select random elements from the upper triangle:
        ind = np.random.choice(len(uT), rem_num, replace=False)
        # make these indices -1
        for i in ind:
            uT[i] = 0
            # now place these values back in the upper triangle:
        p = 0
        for i in range(Cn):
            for j in range(i + 1, Cn):
                data2[i, j] = uT[p]
                if data2[i, j] == 0:
                    data2[j, i] = 0
                p += 1
        hadamard_train = np.ones(data2.shape)
        hadamard_train = np.where(data2 == 0, 0, hadamard_train)
        hadamard_train = pd.DataFrame(hadamard_train)

        L, S, cost = rae.fit(data2, sess=sess, learning_rate=0.001, batch_size=1, inner_iteration=10,iteration=1, verbose=True)


        [Rn, Cn] = data3.shape
        uT = []
        for i in range(Cn):
            for j in range(i + 1, Cn):
                uT.append(data3[i, j])
        fraction = 99

        # calculate entries to be deleted
        rem_num = ((len(uT)) * fraction / 100)  # total number of entries to be removed
        # has to be an integer value
        rem_num = int(rem_num)
        # select random elements from the upper triangle:
        ind = np.random.choice(len(uT), rem_num, replace=False)
        # make these indices -1
        for i in ind:
            uT[i] = 0
        # now place these values back in the upper triangle:
        p = 0
        for i in range(Cn):
            for j in range(i + 1, Cn):
                data3[i, j] = uT[p]
                if data3[i, j] == 0:
                    data3[j, i] = 0
                p += 1
        hadamard_train = np.ones(data3.shape)
        hadamard_train = np.where(data3 == 0, 0, hadamard_train)
        hadamard_train = pd.DataFrame(hadamard_train)

        L, S, cost = rae.fit(data3, sess=sess, learning_rate=0.001, batch_size=1,inner_iteration=10, iteration=1, verbose=True)


        [Rn, Cn] = data4.shape
        uT = []
        for i in range(Cn):
            for j in range(i + 1, Cn):
                uT.append(data4[i, j])
        fraction = 99

        # calculate entries to be deleted
        rem_num = ((len(uT)) * fraction / 100)  # total number of entries to be removed
        # has to be an integer value
        rem_num = int(rem_num)
        # select random elements from the upper triangle:
        ind = np.random.choice(len(uT), rem_num, replace=False)
        # make these indices -1
        for i in ind:
            uT[i] = 0
        # now place these values back in the upper triangle:
        p = 0
        for i in range(Cn):
            for j in range(i + 1, Cn):
                data4[i, j] = uT[p]
                if data4[i, j] == 0:
                    data4[j, i] = 0
                p += 1
        hadamard_train = np.ones(data4.shape)
        hadamard_train = np.where(data4 == 0, 0, hadamard_train)
        hadamard_train = pd.DataFrame(hadamard_train)

        L, S, cost = rae.fit(data4, sess=sess, learning_rate=0.001, batch_size=1
                             , inner_iteration=10, iteration=1, verbose=True)

        [Rn, Cn] = data5.shape
        uT = []
        for i in range(Cn):
            for j in range(i + 1, Cn):
                uT.append(data5[i, j])
        fraction = 99

        # calculate entries to be deleted
        rem_num = ((len(uT)) * fraction / 100)  # total number of entries to be removed
        # has to be an integer value
        rem_num = int(rem_num)
        # select random elements from the upper triangle:
        ind = np.random.choice(len(uT), rem_num, replace=False)
        # make these indices -1
        for i in ind:
            uT[i] = 0
        # now place these values back in the upper triangle:
        p = 0
        for i in range(Cn):
            for j in range(i + 1, Cn):
                data5[i, j] = uT[p]
                if data5[i, j] == 0:
                    data5[j, i] = 0
                p += 1
        hadamard_train = np.ones(data5.shape)
        hadamard_train = np.where(data5 == 0, 0, hadamard_train)
        hadamard_train = pd.DataFrame(hadamard_train)

        L, S, cost = rae.fit(data5, sess=sess, learning_rate=0.001, batch_size=1
                             , inner_iteration=10, iteration=1, verbose=True)


        [Rn, Cn] = data6.shape
        uT = []
        for i in range(Cn):
            for j in range(i + 1, Cn):
                uT.append(data6[i, j])
        fraction = 99

        # calculate entries to be deleted
        rem_num = ((len(uT)) * fraction / 100)  # total number of entries to be removed
        # has to be an integer value
        rem_num = int(rem_num)
        # select random elements from the upper triangle:
        ind = np.random.choice(len(uT), rem_num, replace=False)
        # make these indices -1
        for i in ind:
            uT[i] = 0
        # now place these values back in the upper triangle:
        p = 0
        for i in range(Cn):
            for j in range(i + 1, Cn):
                data6[i, j] = uT[p]
                if data6[i, j] == 0:
                    data6[j, i] = 0
                p += 1
        hadamard_train = np.ones(data6.shape)
        hadamard_train = np.where(data6 == 0, 0, hadamard_train)
        hadamard_train = pd.DataFrame(hadamard_train)

        L, S, cost = rae.fit(data6, sess=sess, learning_rate=0.001, batch_size=16
                             , inner_iteration=10, iteration=1, verbose=True)
        '''
        hadamard_train = np.ones(data_test.shape)
        hadamard_train = np.where(data_test == 0, 0, hadamard_train)
        hadamard_train = pd.DataFrame(hadamard_train)

        L, S, cost = rae.fit(data_test, sess=sess, learning_rate=0.001, batch_size=1
                             , inner_iteration=10, iteration=1, verbose=True)

        L, S, cost = rae.fit(data_test, sess=sess, learning_rate=0.001, batch_size=1
                             , inner_iteration=10, iteration=1, verbose=True)

        L, S, cost = rae.fit(data_test, sess=sess, learning_rate=0.001, batch_size=16
                             , inner_iteration=10, iteration=1, verbose=True)
        L, S, cost = rae.fit(data_test, sess=sess, learning_rate=0.001, batch_size=16
                             , inner_iteration=10, iteration=1, verbose=True)
        '''
        h = rae.transform(data_test, sess=sess)
        R = rae.getRecon(data_test, sess=sess)
        R = pd.DataFrame(R)
        data_test_original = pd.DataFrame(data_test_original)

        for i in range(len(R)):
            for j in R.columns:
                if i != j:
                    if R.iloc[i, j] <= 1.0:
                        R.iloc[i, j] = 1
                    else:
                        continue

        for i in range(len(R)):
            for j in R.columns:
                if i == j:
                    R.iloc[i, j] = 0
                else:
                    continue

        #R = np.round(R)

        R = pd.DataFrame(R)
        data_test = pd.DataFrame(data_test)
        R[hadamard_test == 1.0] = data_test[hadamard_test == 1.0]
        data_test_original = pd.DataFrame(data_test_original)
        R = pd.DataFrame(R)
        R.to_csv('./R_99_zeroshot.csv')
        data_test_original.to_csv('./Original_fb.csv')

        [r,c] = data_test.shape
        hop = np.zeros((r*c))
        ori = np.zeros((r*c))
        meane = []
        abse = []

##################################################################
	#  mean  and absolute hop error calculation----------------
        p = 0
        for i in range(r):
            for j in range(c):
                print(i,j)
                #R = np.array(R)
                #data_test = np.array(data_test_original)
                hop[p] = (R.iloc[i,j])
                ori[p] = (data_test_original.iloc[i,j])
                p = p+1
        x = np.round(hop-ori)
        ori = np.array(ori)
        hadamard_test = np.array(hadamard_test)
        list1 = np.where(hadamard_test == 0)
        #list1 = np.array(list1)
        #print(list1)
        import itertools
        b = list(itertools.chain(*list1))
        #b = list1.flatten()
        b = len(b)
        print(b)

        def findElements(lst1, lst2):
            return [lst1[i] for i in lst2]
        a = findElements(ori, list1)
        print(a)
        print(x)
        xx = (np.sum(abs(x)))/(np.sum(a))
        xx = xx*100
        print(xx)
        #yy = (np.sum(abs(x)))/(r*c)
        yy = (np.sum(abs(x))) / b
        print(yy)
        meane.append(xx)
        abse.append(yy)


