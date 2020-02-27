# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 10:20:08 2019
@author: Sophie
"""
import os
path = os.getcwd()
%load_ext watermark
%load_ext autoreload
%autoreload 2
import sys
import numpy as np
import pandas as pd
from math import ceil
from tqdm import trange
from subprocess import call
from itertools import islice
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix, dok_matrix
%watermark -a 'Ethen' -d -t -v -p numpy,pandas,scipy,sklearn,tqdm
names = ['user_id', 'item_id', 'rating', 'timestamp']
#df = pd.read_csv(file_path, sep = '\t', names = names)#\t
df = pd.read_table('C:/Users/Sophie/Desktop/ml-1m/ratings.dat', sep = '::', names = names)#
print('data dimension: \n', df.shape)
#df.head()
print(df)
def create_matrix(data, users_col, items_col, ratings_col, threshold = None):
    
    if threshold is not None:
        data = data[data[ratings_col] >= threshold]
        data[ratings_col] = 1
    
    for col in (items_col, users_col, ratings_col):
        data[col] = data[col].astype('category')
    ratings = csr_matrix((data[ratings_col],
                          (data[users_col].cat.codes, data[items_col].cat.codes)))
    ratings.eliminate_zeros()
    return ratings, data
items_col = 'item_id'
users_col = 'user_id'
ratings_col = 'rating'
threshold = 3
X, df = create_matrix(df, users_col, items_col, ratings_col, threshold)
X
def create_train_test(ratings, test_size = 0.2, seed = 1234):
    
    assert test_size < 1.0 and test_size > 0.0
    # Dictionary Of Keys based sparse matrix is more efficient
    # for constructing sparse matrices incrementally compared with csr_matrix
    train = ratings.copy().todok()
    test = dok_matrix(train.shape)
    
    # for all the users assign randomly chosen interactions
    # to the test and assign those interactions to zero in the training;
    # when computing the interactions to go into the test set, 
    # remember to round up the numbers (e.g. a user has 4 ratings, if the
    # test_size is 0.2, then 0.8 ratings will go to test, thus we need to
    # round up to ensure the test set gets at least 1 rating)
    rstate = np.random.RandomState(seed)
    for u in range(ratings.shape[0]):
        split_index = ratings[u].indices
        n_splits = ceil(test_size * split_index.shape[0])
        test_index = rstate.choice(split_index, size = n_splits, replace = False)
        test[u, test_index] = ratings[u, test_index]
        train[u, test_index] = 0
    
    train, test = train.tocsr(), test.tocsr()
    return train, test
X_train, X_test = create_train_test(X, test_size = 0.2, seed = 1234)
X_train
class BPR:
   
    def __init__(self, learning_rate = 0.01, n_factors = 15, n_iters = 10, 
                 batch_size = 1000, reg = 0.01, seed = 1234, verbose = True):
        self.reg = reg
        self.seed = seed
        self.verbose = verbose
        self.n_iters = n_iters
        self.n_factors = n_factors
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        # to avoid re-computation at predict
        self._prediction = None
        
    def fit(self, ratings):
       
        indptr = ratings.indptr
        indices = ratings.indices
        n_users, n_items = ratings.shape
        
        
        batch_size = self.batch_size
        if n_users < batch_size:
            batch_size = n_users
            sys.stderr.write('WARNING: Batch size is greater than number of users,'
                             'switching to a batch size of {}\n'.format(n_users))
        batch_iters = n_users // batch_size
       
        rstate = np.random.RandomState(self.seed)
        self.user_factors = rstate.normal(size = (n_users, self.n_factors))
        self.item_factors = rstate.normal(size = (n_items, self.n_factors))
        
        loop = range(self.n_iters)
        if self.verbose:
            loop = trange(self.n_iters, desc = self.__class__.__name__)
        
        for _ in loop:
            for _ in range(batch_iters):
                sampled = self._sample(n_users, n_items, indices, indptr)
                sampled_users, sampled_pos_items, sampled_neg_items = sampled
                self._update(sampled_users, sampled_pos_items, sampled_neg_items)
        return self
    
    def _sample(self, n_users, n_items, indices, indptr):
        sampled_pos_items = np.zeros(self.batch_size, dtype = np.int)
        sampled_neg_items = np.zeros(self.batch_size, dtype = np.int)
        sampled_users = np.random.choice(
            n_users, size = self.batch_size, replace = False)
        for idx, user in enumerate(sampled_users):
            pos_items = indices[indptr[user]:indptr[user + 1]]
            pos_item = np.random.choice(pos_items)
            neg_item = np.random.choice(n_items)
            while neg_item in pos_items:
                neg_item = np.random.choice(n_items)
            sampled_pos_items[idx] = pos_item
            sampled_neg_items[idx] = neg_item
        return sampled_users, sampled_pos_items, sampled_neg_items
                
    def _update(self, u, i, j):
        
        user_u = self.user_factors[u]
        item_i = self.item_factors[i]
        item_j = self.item_factors[j]
        
        r_uij = np.sum(user_u * (item_i - item_j), axis = 1)
        sigmoid = np.exp(-r_uij) / (1.0 + np.exp(-r_uij))
        
         sigmoid_tiled = np.tile(sigmoid, (self.n_factors, 1)).T
        grad_u = sigmoid_tiled * (item_j - item_i) + self.reg * user_u
        grad_i = sigmoid_tiled * -user_u + self.reg * item_i
        grad_j = sigmoid_tiled * user_u + self.reg * item_j
        self.user_factors[u] -= self.learning_rate * grad_u
        self.item_factors[i] -= self.learning_rate * grad_i
        self.item_factors[j] -= self.learning_rate * grad_j
        return self
    def predict(self):
        
        if self._prediction is None:
            self._prediction = self.user_factors.dot(self.item_factors.T)
        return self._prediction
    def _predict_user(self, user):
        
        user_pred = self.user_factors[user].dot(self.item_factors.T)
        return user_pred
    def recommend(self, ratings, N = 5):
       
        n_users = ratings.shape[0]
        recommendation = np.zeros((n_users, N), dtype = np.uint32)
        for user in range(n_users):
            top_n = self._recommend_user(ratings, user, N)
            recommendation[user] = top_n
        return recommendation
    def _recommend_user(self, ratings, user, N):
        
        scores = self._predict_user(user)
         liked = set(ratings[user].indices)
        count = N + len(liked)
        if count < scores.shape[0]:
             ids = np.argpartition(scores, -count)[-count:]
            best_ids = np.argsort(scores[ids])[::-1]
            best = ids[best_ids]
        else:
            best = np.argsort(scores)[::-1]
        top_n = list(islice((rec for rec in best if rec not in liked), N))
        return top_n
    
    def get_similar_items(self, N = 5, item_ids = None):
       
         normed_factors = normalize(self.item_factors)
        knn = NearestNeighbors(n_neighbors = N + 1, metric = 'euclidean')
        knn.fit(normed_factors)
       
        if item_ids is not None:
            normed_factors = normed_factors[item_ids]
        _, items = knn.kneighbors(normed_factors)
        similar_items = items[:, 1:].astype(np.uint32)
        return similar_items
    
bpr_params = {'reg': 0.01,
              'learning_rate': 0.1,
              'n_iters': 160,
              'n_factors': 15,
              'batch_size': 100}
bpr = BPR(**bpr_params)
bpr.fit(X_train)
def auc_score(model, ratings):
    
    auc = 0.0
    n_users, n_items = ratings.shape
    for user, row in enumerate(ratings):
        y_pred = model._predict_user(user)
        y_true = np.zeros(n_items)
        y_true[row.indices] = 1
        auc += roc_auc_score(y_true, y_pred)
    auc /= n_users
    return auc
print(auc_score(bpr, X_train))
print(auc_score(bpr, X_test))