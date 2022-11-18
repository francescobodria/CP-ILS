import pandas as pd
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import os

def load_bb(X_train, X_test, dataset_name, bb_name):

    if bb_name == 'xgb':
        from xgboost import XGBClassifier
        clf_xgb = pickle.load(open(f'./blackboxes/{dataset_name}_xgboost.p','rb'))
        def predict(x, return_proba=False):
            if return_proba:
                return clf_xgb.predict_proba(x)[:,1].ravel()
            else: return clf_xgb.predict(x).ravel().ravel()
        y_train_pred = predict(X_train)
        y_test_pred = predict(X_test)
        return y_train_pred, y_test_pred, clf_xgb

    elif bb_name == 'rf':
        from sklearn.ensemble import RandomForestClassifier
        clf_rf = pickle.load(open(f'./blackboxes/{dataset_name}_rf.p','rb'))
        def predict(x, return_proba=False):
            if return_proba:
                return clf_rf.predict_proba(x)[:,1].ravel()
            else: return clf_rf.predict(x).ravel().ravel()
        y_test_pred = predict(X_test, return_proba=True)
        y_train_pred = predict(X_train, return_proba=True)
        return y_train_pred, y_test_pred, clf_rf

    elif bb_name == 'svc':
        from sklearn.svm import SVC
        clf_svc = pickle.load(open(f'./blackboxes/{dataset_name}_svc.p','rb'))
        def predict(x, return_proba=False):
            if return_proba:
                return clf_svc.predict_proba(x)[:,1].ravel()
            else: return clf_svc.predict(x).ravel().ravel()
        y_train_pred = predict(X_train.values, return_proba=True)
        y_test_pred = predict(X_test.values, return_proba=True)
        return y_train_pred, y_test_pred, clf_svc

    elif bb_name == 'nn':
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras.callbacks import EarlyStopping
        clf_nn = keras.Sequential([
            keras.layers.Dense(units=10, activation='relu'),
            keras.layers.Dense(units=5, activation='relu'),
            keras.layers.Dense(units=1, activation='sigmoid'),
        ])
        clf_nn.compile(optimizer='adam', 
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

        clf_nn.load_weights(f'./blackboxes/{dataset_name}_tf_nn')
        clf_nn.trainable = False
        def predict(x, return_proba=False):
            if return_proba:
                return clf_nn.predict(x.values).ravel()
            else: return np.round(clf_nn.predict(x.values).ravel()).astype(int).ravel()
        y_test_pred = predict(X_test, return_proba=True)
        y_train_pred = predict(X_train, return_proba=True)
        return y_train_pred, y_test_pred, clf_nn
        