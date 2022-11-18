import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import pandas as pd
pd.set_option('display.max_columns', None)
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

rnd = 42
np.random.seed(rnd)
torch.manual_seed(rnd)

from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from collections import defaultdict

def get_features_map(feature_names, real_feature_names):
    features_map = defaultdict(dict)
    i = 0
    j = 0

    while i < len(feature_names) and j < len(real_feature_names):
        if feature_names[i] == real_feature_names[j]:
            features_map[j][feature_names[i].replace('%s=' % real_feature_names[j], '')] = i
            i += 1
            j += 1
        elif feature_names[i].startswith(real_feature_names[j]):
            features_map[j][feature_names[i].replace('%s=' % real_feature_names[j], '')] = i
            i += 1
        else:
            j += 1
    return features_map

def get_real_feature_names(rdf, numeric_columns, class_name):
    if isinstance(class_name, list):
        real_feature_names = [c for c in rdf.columns if c in numeric_columns and c not in class_name]
        real_feature_names += [c for c in rdf.columns if c not in numeric_columns and c not in class_name]
    else:
        real_feature_names = [c for c in rdf.columns if c in numeric_columns and c != class_name]
        real_feature_names += [c for c in rdf.columns if c not in numeric_columns and c != class_name]
    return real_feature_names

def one_hot_encoding(df, class_name):
    if not isinstance(class_name, list):
        dfX = pd.get_dummies(df[[c for c in df.columns if c != class_name]], prefix_sep='=')
        class_name_map = {v: k for k, v in enumerate(sorted(df[class_name].unique()))}
        dfY = df[class_name].map(class_name_map)
        df = pd.concat([dfX, dfY], axis=1)
        df =df.reindex(dfX.index)
        feature_names = list(dfX.columns)
        class_values = sorted(class_name_map)
    else: # isinstance(class_name, list)
        dfX = pd.get_dummies(df[[c for c in df.columns if c not in class_name]], prefix_sep='=')
        # class_name_map = {v: k for k, v in enumerate(sorted(class_name))}
        class_values = sorted(class_name)
        dfY = df[class_values]
        df = pd.concat([dfX, dfY], axis=1)
        df = df.reindex(dfX.index)
        feature_names = list(dfX.columns)
    return df, feature_names, class_values

def remove_missing_values(df):
    for column_name, nbr_missing in df.isna().sum().to_dict().items():
        if nbr_missing > 0:
            if column_name in df._get_numeric_data().columns:
                mean = df[column_name].mean()
                df[column_name].fillna(mean, inplace=True)
            else:
                mode = df[column_name].mode().values[0]
                df[column_name].fillna(mode, inplace=True)
    return df

def get_numeric_columns(df):
    numeric_columns = list(df._get_numeric_data().columns)
    return numeric_columns

class_name = 'default'
# Load and transform dataset 
df = pd.read_csv('../data/german_credit.csv', skipinitialspace=True, na_values='?', keep_default_na=True)
df.columns = [c.replace('=', '') for c in df.columns]

df = remove_missing_values(df)
numeric_columns = get_numeric_columns(df)
rdf = df
df, feature_names, class_values = one_hot_encoding(df, class_name)
real_feature_names = get_real_feature_names(rdf, numeric_columns, class_name)
rdf = rdf[real_feature_names + (class_values if isinstance(class_name, list) else [class_name])]
features_map = get_features_map(feature_names, real_feature_names)
std = MinMaxScaler(feature_range=(-1,1))
df.iloc[:,[0,1,4]] = std.fit_transform(df.values[:,[0,1,4]])
hot_enc = OneHotEncoder(handle_unknown='ignore')
for i in [2,3,5,6]:
    df.iloc[:,i] = [df.columns[i]+'='+df.iloc[:,i].values.astype(str)[j] for j in range(len(df))] 
hot_enc.fit(df.iloc[:,[2,3,5,6]])
df[np.hstack(hot_enc.categories_).tolist()]=hot_enc.transform(df.iloc[:,[2,3,5,6]]).toarray().astype(int)
df = df.drop(list(df.columns[[2,3,5,6]]), axis=1, inplace=False)
y = df['default'].astype(int)
df.drop(['default'], axis=1, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, stratify=y, random_state=rnd)

### XGBOOST

from xgboost import XGBClassifier
#clf_xgb = XGBClassifier(n_estimators=10, reg_lambda=1, use_label_encoder=False, eval_metric='logloss')
#clf_xgb.fit(X_train.values, y_train)
#pickle.dump(clf_xgb,open('../blackboxes/german_xgboost.p','wb'))

clf_xgb = pickle.load(open('../blackboxes/german_xgboost.p','rb'))
def predict(x, return_proba=False):
    if return_proba:
        return clf_xgb.predict_proba(x)[:,1].ravel()
    else: return clf_xgb.predict(x).ravel().ravel()
y_train_pred = predict(X_train.values)
y_test_pred = predict(X_test.values)
print('XGBOOST')
print('train acc:',np.mean(np.round(y_train_pred)==y_train))
print('test acc:',np.mean(np.round(y_test_pred)==y_test))

### RF
from sklearn.ensemble import RandomForestClassifier

#clf_rf = RandomForestClassifier(max_depth=7,random_state=rnd)
#clf_rf.fit(X_train, y_train)

#pickle.dump(clf_rf,open('../blackboxes/german_rf.p','wb'))

clf_rf = pickle.load(open('../blackboxes/german_rf.p','rb'))

def predict(x, return_proba=False):
    if return_proba:
        return clf_rf.predict_proba(x)[:,1].ravel()
    else: return clf_rf.predict(x).ravel().ravel()

y_test_pred = predict(X_test, return_proba=True)
y_train_pred = predict(X_train, return_proba=True)
print('RF')
print('train acc:',np.mean(np.round(y_train_pred)==y_train))
print('test acc:',np.mean(np.round(y_test_pred)==y_test))

### SVC

#from sklearn.svm import SVC
#clf_svc = SVC(gamma='auto', probability=True)
#clf_svc.fit(X_train, y_train)
#pickle.dump(clf_svc,open('../blackboxes/german_svc.p','wb'))

clf_svc = pickle.load(open('../blackboxes/german_svc.p','rb'))

def predict(x, return_proba=False):
    if return_proba:
        return clf_svc.predict_proba(x)[:,1].ravel()
    else: return clf_svc.predict(x).ravel().ravel()

y_train_pred = predict(X_train, return_proba=True)
y_test_pred = predict(X_test, return_proba=True)
print('SVC')
print('train acc:',np.mean(np.round(y_train_pred)==y_train))
print('test acc:',np.mean(np.round(y_test_pred)==y_test))

# ### NN tf

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
    callbacks=[early_stopping],
    epochs=10000,
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

#plot_metric(history, 'loss')
clf_nn.save_weights('../blackboxes/german_tf_nn')

from sklearn.metrics import accuracy_score
clf_nn.load_weights('../blackboxes/german_tf_nn')
clf_nn.trainable = False

def predict(x, return_proba=False):
    if return_proba:
        return clf_nn.predict(x).ravel()
    else: return np.round(clf_nn.predict(x).ravel()).astype(int).ravel()

print('NN')
print(accuracy_score(np.round(predict(X_train, return_proba = True)),y_train))
print(accuracy_score(np.round(predict(X_test, return_proba = True)),y_test))
print('---------------')

for black_box in ['nn']:

    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, stratify=y, random_state=rnd)

    if black_box=='xgb':
        def predict(x, return_proba=False):
            if return_proba:
                return clf_xgb.predict_proba(x)[:,1].ravel()
            else: return clf_xgb.predict(x).ravel().ravel()
        y_test_pred = predict(X_test, return_proba=True)
        y_train_pred = predict(X_train, return_proba=True)
    elif black_box=='rf':
        def predict(x, return_proba=False):
            if return_proba:
                return clf_rf.predict_proba(x)[:,1].ravel()
            else: return clf_rf.predict(x).ravel().ravel()
        y_test_pred = predict(X_test, return_proba=True)
        y_train_pred = predict(X_train, return_proba=True)
    elif black_box=='svc':
        def predict(x, return_proba=False):
            if return_proba:
                return clf_svc.predict_proba(x)[:,1].ravel()
            else: return clf_svc.predict(x).ravel().ravel()
        y_test_pred = predict(X_test, return_proba=True)
        y_train_pred = predict(X_train, return_proba=True)
    elif black_box=='nn':
        def predict(x, return_proba=False):
            if return_proba:
                return clf_nn.predict(x).ravel()
            else: return np.round(clf_nn.predict(x).ravel()).astype(int).ravel()
        y_test_pred = predict(X_test, return_proba=True)
        y_train_pred = predict(X_train, return_proba=True)

    import dice_ml

    dataset = pd.concat((y_train,df),axis=1)
    d = dice_ml.Data(dataframe=dataset, continuous_features=['duration_in_month', 'credit_amount', 'age'], outcome_name='default')

    if black_box == 'nn':
        df_nn = pd.read_csv('../data/german_credit.csv', skipinitialspace=True, na_values='?', keep_default_na=True)
        df_nn.columns = [c.replace('=', '') for c in df_nn.columns]
        df_nn = remove_missing_values(df_nn)
        y = df_nn['default'].astype(int)
        df_nn.drop(['default'], axis=1, inplace=True)
        l = list(df_nn.columns)
        l.remove('duration_in_month')
        l.remove('age')
        l.remove('credit_amount')
        df_nn = df_nn.loc[:,['duration_in_month','credit_amount','age']+l]
        std = MinMaxScaler(feature_range=(-1,1))
        df_nn.iloc[:,:3] = std.fit_transform(df_nn.values[:,:3])
        hot_enc = OneHotEncoder(handle_unknown='ignore')
        hot_enc.fit(df_nn.values[:,3:])
        X_train, X_test, y_train, y_test = train_test_split(df_nn, y, test_size=0.2, stratify=y, random_state=rnd)
        dataset = pd.DataFrame(np.hstack((y_train.values.reshape(-1,1).astype(int),X_train.values)), columns=['default']+list(X_train.columns))
        dataset.loc[:,'duration_in_month']=dataset.loc[:,'duration_in_month'].values.astype(float)
        dataset.loc[:,'age']=dataset.loc[:,'age'].values.astype(float)
        dataset.loc[:,'credit_amount']=dataset.loc[:,'credit_amount'].values.astype(float)
        dataset.iloc[:,4:] = dataset.iloc[:,4:].values.astype(str)
        d = dice_ml.Data(dataframe=dataset, continuous_features=['duration_in_month','age','credit_amount'], outcome_name='default')
        m = dice_ml.Model(model=clf_nn, backend='TF2')
    elif black_box == 'xgb':
        m = dice_ml.Model(model=clf_xgb, backend='sklearn')
    elif black_box == 'svc':
        m = dice_ml.Model(model=clf_svc, backend='sklearn')
    elif black_box == 'rf':
        m = dice_ml.Model(model=clf_rf, backend='sklearn')

    exp = dice_ml.Dice(d, m, method='random')

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
                                                total_CFs = 4, 
                                                desired_class = "opposite", 
                                                verbose = False,
                                                permitted_range = {'age' : [-1, 1], 
                                                                   'credit_amount' : [-1, 1],
                                                                   'duration_in_month' : [-1, 1]})

        q_cfs_dice = dice_exp.cf_examples_list[0].final_cfs_df.values
        return q_cfs_dice

    from scipy.spatial.distance import cdist, euclidean, hamming

    if black_box != 'nn':
        X_train = X_train.values
    else:
        X_train = np.hstack((X_train.values[:,:3].astype(float),hot_enc.transform(X_train.values[:,3:]).toarray().astype(int))) 

    for idx in tqdm(range(100)):
        query_instances = dataset.iloc[idx:idx+1,1:]
        try: 
            q_cfs_dice = compute_cfs_dice(query_instances)
        except:
            continue
        if black_box == 'nn':
            q_cfs_dice = np.hstack((q_cfs_dice[:,:3].astype(float),hot_enc.transform(q_cfs_dice[:,3:-1]).toarray().astype(int),q_cfs_dice[:,-1].reshape(-1,1).astype(float)))
            query_instances = np.hstack((query_instances.values[:,:3].astype(float),hot_enc.transform(query_instances.values[:,3:]).toarray().astype(int)))
            pred = predict(query_instances)
        else:
            pred = predict(query_instances)
            query_instances = query_instances.values
        
        d_dist_dice.append(np.min(cdist(q_cfs_dice[:,3:-1],query_instances[:,3:],metric='hamming') + cdist(q_cfs_dice[:,:3],query_instances[:,:3],metric='euclidean')))
        d_count_dice.append(np.min(np.sum(q_cfs_dice[:,:-1]!=query_instances,axis=1)))
        d_impl_dice.append(np.min(cdist(q_cfs_dice[:,3:-1],X_train[:,3:],metric='hamming') + cdist(q_cfs_dice[:,:3],X_train[:,:3],metric='euclidean')))
        r = np.argsort(cdist(q_cfs_dice[:,:-1],X_train),axis=1)[:,:10]
        d_adv_dice.append(np.mean(np.array([np.mean(predict(X_train[r,:][i,:])==pred) for i in range(q_cfs_dice.shape[0])])))
        num_dice.append(len(q_cfs_dice))
        div_dist_dice.append(1/(q_cfs_dice.shape[0]**2)*np.sum(cdist(q_cfs_dice[:,:-1], q_cfs_dice[:,:-1])))
        div_count_dice.append(72/(q_cfs_dice.shape[0]**2)*np.sum(cdist(q_cfs_dice[:,:-1], q_cfs_dice[:,:-1],metric='hamming')))

    with open('../results/german_results_dice.txt','a') as f:
        f.write('dice '+black_box+'\n')
        f.write(str(np.round(np.mean(d_dist_dice),5))+','+str(np.round(np.std(d_dist_dice),5))+'\n')
        f.write(str(np.round(np.mean(d_count_dice),5))+','+str(np.round(np.std(d_count_dice),5))+'\n')
        f.write(str(np.round(np.mean(d_impl_dice),5))+','+str(np.round(np.std(d_impl_dice),5))+'\n')
        f.write(str(np.round(np.mean(d_adv_dice),5))+','+str(np.round(np.std(d_adv_dice),5))+'\n')
        f.write(str(np.round(np.mean(num_dice),5))+','+str(np.round(np.std(num_dice),5))+'\n')
        f.write(str(np.round(np.mean(div_dist_dice),5))+','+str(np.round(np.std(div_dist_dice),5))+'\n')
        f.write(str(np.round(np.mean(div_count_dice),5))+','+str(np.round(np.std(div_count_dice),5))+'\n')
        f.write('success_rate: '+str(len(d_dist_dice)/100)+'\n')

