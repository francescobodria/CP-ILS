import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import pandas as pd
pd.set_option('display.max_columns', 500)
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import time
import os

rnd = 42
np.random.seed(rnd)
torch.manual_seed(rnd)

from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

df = pd.read_csv('../data/adult_clear.csv')
df = df[df["workclass"] != "?"]
df = df[df["occupation"] != "?"]
df = df[df["native-country"] != "?"]
df.replace(['Divorced', 'Married-AF-spouse',
            'Married-civ-spouse', 'Married-spouse-absent',
            'Never-married', 'Separated', 'Widowed'],
           ['notmarried', 'married', 'married', 'married',
            'notmarried', 'notmarried', 'notmarried'], inplace=True)
df['education'].replace(['Preschool', '10th', '11th', '12th', '1st-4th', '5th-6th', '7th-8th', '9th'], 'dropout',
                        inplace=True)
df['education'].replace(['HS-Grad', 'HS-grad'], 'HighGrad', inplace=True)
df['education'].replace(['Some-college', 'Assoc-acdm', 'Assoc-voc'], 'CommunityCollege', inplace=True)
df = df[df.race == 'White']
# excludes 10 observations
df = df[df['workclass'] != 'Never-worked']
# excludes 14 observations
df = df[df['occupation'] != 'Armed-Forces']
# excludes 21 observations
df = df[df['workclass'] != 'Without-pay']
df.drop(['fnlwgt', 'educational-num', 'relationship', 'race', 'capital-gain', 'capital-loss'],
            axis=1, inplace=True)
df['workclass'].replace(['Local-gov', 'State-gov', 'Federal-gov'], 'Gov', inplace=True)
df['workclass'].replace(['Private', 'Self-emp-not-inc', 'Self-emp-inc'], 'Private', inplace=True)
df['occupation'].replace(
    ['Craft-repair', 'Machine-op-inspct', 'Handlers-cleaners', 'Transport-moving', 'Adm-clerical',
     'Farming-fishing'], 'BlueCollar', inplace=True)
df['occupation'].replace(['Other-service', 'Protective-serv', 'Tech-support', 'Priv-house-serv'], 'Services',
                         inplace=True)
df['occupation'].replace(['Exec-managerial'], 'ExecManagerial', inplace=True)
df['occupation'].replace(['Prof-specialty'], 'ProfSpecialty', inplace=True)
df['education'].replace(['Prof-school'], 'ProfSchool', inplace=True)
df['native-country'].replace(['Cambodia', 'Canada', 'China', 'Columbia', 'Cuba', 'Dominican-Republic', 'Ecuador', \
                              'El-Salvador', 'England', 'France', 'Germany', 'Greece', 'Guatemala', 'Haiti',\
                              'Holand-Netherlands', 'Honduras', 'Hong', 'Hungary', 'India', 'Iran', 'Ireland',\
                              'Italy', 'Jamaica', 'Japan', 'Mexico', 'Nicaragua', 'Outlying-US(Guam-USVI-etc)',\
                              'Peru', 'Philippines', 'Poland', 'Portugal', 'Puerto-Rico', 'Scotland', 'South',\
                              'Taiwan','Thailand', 'Trinadad&Tobago', 'Vietnam','Yugoslavia'], 'Non_US', inplace=True)
df.rename(columns={'hours-per-week': 'hoursPerWeek'}, inplace=True)
df.rename(columns={'marital-status': 'marital_status'}, inplace=True)
df.rename(columns={'native-country': 'native_country'}, inplace=True)
columns_titles = ["age","hoursPerWeek","education","marital_status","occupation","gender","native_country","income"]
df=df.reindex(columns=columns_titles)
df = df[~df.duplicated()]
X = df.copy()
std = MinMaxScaler(feature_range=(-1,1))
X.iloc[:,:2] = std.fit_transform(X.values[:,:2])
X.loc[:,'age']=X.loc[:,'age'].values.astype(float)
X.loc[:,'hoursPerWeek']=X.loc[:,'hoursPerWeek'].values.astype(float)
hot_enc = OneHotEncoder(handle_unknown='ignore')
hot_enc.fit(X.iloc[:,[2,3,4,5,6]])
X[np.hstack(hot_enc.categories_).tolist()]=hot_enc.transform(X.iloc[:,[2,3,4,5,6]]).toarray().astype(int)
X.drop(['education'], axis=1, inplace=True)
X.drop(['marital_status'], axis=1, inplace=True)
X.drop(['occupation'], axis=1, inplace=True)
X.drop(['gender'], axis=1, inplace=True)
X.drop(['native_country'], axis=1, inplace=True)
X.drop(['income'], axis=1, inplace=True)
y = df["income"].apply(lambda x: ">50K" in x).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=rnd)

### XGBOOST

from xgboost import XGBClassifier
clf_xgb = XGBClassifier(n_estimators=60, reg_lambda=3, use_label_encoder=False, eval_metric='logloss')
clf_xgb.fit(X_train, y_train)
pickle.dump(clf_xgb,open('../blackboxes/adult_dice_xgboost.p','wb'))

clf_xgb = pickle.load(open('../blackboxes/adult_dice_xgboost.p','rb'))
def predict(x, return_proba=False):
    if return_proba:
        return clf_xgb.predict_proba(x)[:,1].ravel()
    else: return clf_xgb.predict(x).ravel().ravel()
y_train_pred = predict(X_train)
y_test_pred = predict(X_test)
print('XGBOOST')
print('train acc:',np.mean(np.round(y_train_pred)==y_train))
print('test acc:',np.mean(np.round(y_test_pred)==y_test))

### RF

from sklearn.ensemble import RandomForestClassifier

clf_rf = RandomForestClassifier(max_depth=7,random_state=rnd)
clf_rf.fit(X_train, y_train)

pickle.dump(clf_rf,open('../blackboxes/adult_dice_rf.p','wb'))
clf_rf = pickle.load(open('../blackboxes/adult_dice_rf.p','rb'))

def predict(x, return_proba=False):
    if return_proba:
        return clf_rf.predict_proba(x)[:,1].ravel()
    else: return clf_rf.predict(x).ravel().ravel()
y_test_pred = predict(X_test, return_proba=True)
y_train_pred = predict(X_train, return_proba=True)
print('RF')
print('train acc:',np.mean(np.round(y_train_pred)==y_train))
print('test acc:',np.mean(np.round(y_test_pred)==y_test))

# SVC

from sklearn.svm import SVC
clf_svc = SVC(gamma='auto', probability=True)
clf_svc.fit(X_train, y_train)
pickle.dump(clf_svc,open('../blackboxes/adult_dice_svc.p','wb'))
clf_svc = pickle.load(open('../blackboxes/adult_dice_svc.p','rb'))
def predict(x, return_proba=False):
    if return_proba:
        return clf_svc.predict_proba(x)[:,1].ravel()
    else: return clf_svc.predict(x).ravel().ravel()
y_train_pred = predict(X_train, return_proba=True)
y_test_pred = predict(X_test, return_proba=True)
print('SVC')
print('train acc:',np.mean(np.round(y_train_pred)==y_train))
print('test acc:',np.mean(np.round(y_test_pred)==y_test))

### NN tf

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping

BATCH_SIZE = 1024
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(2048).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE)

clf_nn = keras.Sequential([
    keras.layers.Dense(units=10, activation='relu'),
    keras.layers.Dense(units=5, activation='relu'),
    keras.layers.Dense(units=1, activation='sigmoid'),
])

early_stopping = EarlyStopping(patience=5)

clf_nn.compile(optimizer='adam', 
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = clf_nn.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=1000,
    callbacks=[early_stopping],
    verbose=0
    )

def plot_metric(history, metric):
    train_metrics = history.history[metric]
    val_metrics = history.history['val_'+metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics)
    plt.plot(epochs, val_metrics)
    plt.title('Training and validation '+ metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.grid()
    plt.legend(["train_"+metric, 'val_'+metric])
    plt.show()

plot_metric(history, 'loss')
clf_nn.save_weights('../blackboxes/adult_dice_tf_nn')

from sklearn.metrics import accuracy_score
clf_nn.load_weights('../blackboxes/adult_dice_tf_nn')
clf_nn.trainable = False

def predict(x, return_proba=False):
    if return_proba:
        return clf_nn.predict(x).ravel()
    else: return np.round(clf_nn.predict(x).ravel()).astype(int).ravel()

print('NN')
print(accuracy_score(np.round(predict(X_train, return_proba = True)),y_train))
print(accuracy_score(np.round(predict(X_test, return_proba = True)),y_test))
print('---------------')

for black_box in ['svc']:

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=rnd)

    from scipy.spatial.distance import euclidean, cdist

    # DICE

    import dice_ml

    dataset = pd.DataFrame(np.hstack((y_test.values.reshape(-1,1).astype(int),X_test.values)), columns=['income']+list(X_test.columns))
    d = dice_ml.Data(dataframe=dataset, continuous_features=['age', 'hoursPerWeek'], outcome_name='income')

    if black_box == 'nn':
        X = df.copy()
        std = MinMaxScaler(feature_range=(-1,1))
        X.iloc[:,:2] = std.fit_transform(X.values[:,:2])
        hot_enc = OneHotEncoder(handle_unknown='ignore')
        hot_enc.fit(X.iloc[:,[2,3,4,5,6]])
        X.drop(['income'], axis=1, inplace=True)
        y = df["income"].apply(lambda x: ">50K" in x).astype(int)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=rnd)
        dataset = pd.DataFrame(np.hstack((y_test.values.reshape(-1,1).astype(int),X_test.values)), columns=['income']+list(X_test.columns))
        X_train = np.hstack((X_train.values[:,:2].astype(float),hot_enc.transform(pd.DataFrame(X_train.values[:,[2,3,4,5,6]],columns=list(df.columns)[2:-1])).toarray().astype(int)))  
        dataset.loc[:,'age']=dataset.loc[:,'age'].values.astype(float)
        dataset.loc[:,'hoursPerWeek']=dataset.loc[:,'hoursPerWeek'].values.astype(float)
        d = dice_ml.Data(dataframe=dataset, continuous_features=['age', 'hoursPerWeek'], outcome_name='income')
        m = dice_ml.Model(model=clf_nn, backend='TF2')
    elif black_box == 'rf':
        X_train = X_train.values
        m = dice_ml.Model(model=clf_rf, backend='sklearn')
    elif black_box == 'svc':
        X_train = X_train.values
        m = dice_ml.Model(model=clf_svc, backend='sklearn')
    elif black_box == 'xgb':
        X_train = X_train.values
        m = dice_ml.Model(model=clf_xgb, backend='sklearn')

    # initiate DiCE
    exp = dice_ml.Dice(d, m, method='random')

    from scipy.spatial.distance import cdist

    d_dist_dice = []
    d_count_dice = []
    d_impl_dice = []
    d_adv_dice = []
    num_dice = []
    div_dist_dice = []
    div_count_dice = []

    import errno
    import os
    import signal
    import functools

    class TimeoutError(Exception):
        pass

    def timeout(seconds=120, error_message=os.strerror(errno.ETIME)):
        def decorator(func):
            def _handle_timeout(signum, frame):
                raise TimeoutError(error_message)

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                signal.signal(signal.SIGALRM, _handle_timeout)
                signal.alarm(seconds)
                try:
                    result = func(*args, **kwargs)
                finally:
                    signal.alarm(0)
                return result

            return wrapper

        return decorator
    
    @timeout(120)
    def compute_cfs_dice(query_instances): 
        dice_exp = exp.generate_counterfactuals(query_instances, 
                                                total_CFs=4, 
                                                desired_class="opposite", 
                                                verbose=False,
                                                permitted_range={'age':[-1, 1], 'hoursPerWeek':[-1, 1]})
        q_cfs_dice = dice_exp.cf_examples_list[0].final_cfs_df.values
        return q_cfs_dice

    for i in [13]:#tqdm(range(100)):
        query_instances = dataset.iloc[i:i+1,1:]
        print(query_instances)
        print(query_instances.columns)
        try:
            q_cfs_dice = compute_cfs_dice(query_instances)
        except:
            continue
        if black_box == 'nn':
            q_cfs_dice = np.hstack((q_cfs_dice[:,:2].astype(float),hot_enc.transform(pd.DataFrame(q_cfs_dice[:,[2,3,4,5,6]],columns=list(df.columns)[2:-1])).toarray().astype(int),q_cfs_dice[:,-1].reshape(-1,1).astype(float)))
            query_instances = np.hstack((query_instances.values[:,:2].astype(float),hot_enc.transform(pd.DataFrame(query_instances.values[:,[2,3,4,5,6]],columns=list(df.columns)[2:-1])).toarray().astype(int)))
            pred = predict(query_instances)
        else:
            query_instances = query_instances.values
            pred = predict(query_instances)
        
        r = np.argsort(cdist(q_cfs_dice[:,:-1],X_train),axis=1)[:,:10]
        d_adv_dice.append(np.mean(np.array([np.mean(predict(X_train[r,:][i,:])==pred) for i in range(q_cfs_dice.shape[0])])))

        d_dist_dice.append(np.min(cdist(q_cfs_dice[:,:-1],query_instances)))
        d_count_dice.append(np.min(np.sum(q_cfs_dice[:,:-1]!=query_instances,axis=1)))
        d_impl_dice.append(np.min(cdist(q_cfs_dice[:,:-1],X_train)))
        num_dice.append(len(q_cfs_dice))
        div_dist_dice.append(1/(q_cfs_dice.shape[0]**2)*np.sum(cdist(q_cfs_dice[:,:-1],q_cfs_dice[:,:-1])))
        div_count_dice.append(20/(q_cfs_dice.shape[0]**2)*np.sum(cdist(q_cfs_dice[:,:-1], q_cfs_dice[:,:-1],metric='hamming')))
        print(q_cfs_dice)
    
    #with open('../results/adult_results_dice.txt','a') as f:
        #f.write('dice '+black_box+'\n')
        #f.write(str(np.round(np.mean(d_dist_dice),5))+','+str(np.round(np.std(d_dist_dice),5))+'\n')
        #f.write(str(np.round(np.mean(d_count_dice),5))+','+str(np.round(np.std(d_count_dice),5))+'\n')
        #f.write(str(np.round(np.mean(d_adv_dice),5))+','+str(np.round(np.std(d_adv_dice),5))+'\n')
        #f.write(str(np.round(np.mean(d_impl_dice),5))+','+str(np.round(np.std(d_impl_dice),5))+'\n')
        #f.write(str(np.round(np.mean(num_dice),5))+','+str(np.round(np.std(num_dice),5))+'\n')
        #f.write(str(np.round(np.mean(div_dist_dice),5))+','+str(np.round(np.std(div_dist_dice),5))+'\n')
        #f.write(str(np.round(np.mean(div_count_dice),5))+','+str(np.round(np.std(div_count_dice),5))+'\n')
        #f.write('success_rate: '+str(len(d_dist_dice)/100)+'\n')









