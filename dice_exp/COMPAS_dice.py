import pandas as pd
import pickle
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import os

rnd = 384
np.random.seed(rnd)
torch.manual_seed(rnd)

from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

df = pd.read_csv('./data/compas-scores-two-years.csv')
df = df[(df.days_b_screening_arrest>=-30)*(df.days_b_screening_arrest<=30)]
df = df.drop(['name','id','dob','age_cat','c_case_number','compas_screening_date','c_jail_in','c_jail_out','c_offense_date',
              'c_arrest_date','r_case_number','r_charge_degree','r_days_from_arrest','r_offense_date','r_charge_desc','r_jail_in',
              'r_jail_out','violent_recid','vr_case_number','vr_charge_degree','vr_offense_date','vr_charge_desc','v_screening_date',
              'in_custody','out_custody','v_type_of_assessment','type_of_assessment','screening_date','is_recid','first','last','c_charge_desc'],axis=1,inplace=False)
cat_columns = ['sex','race','c_charge_degree','score_text','v_score_text','event','is_violent_recid']
idx = np.ravel([np.where(np.array(list(df.columns)) == i) for i in cat_columns])
df = df.loc[:,cat_columns+list(np.delete(df.columns, idx))]
df.loc[:,'event'] = df.event.replace({0:'ev_0',1:'ev_1'})
df.loc[:,'is_violent_recid'] = df.is_violent_recid.replace({0:'not_violent_recid',1:'violent_recid'})
df.loc[:,'c_charge_degree'] = df.c_charge_degree.replace({'M':'charge_degree_M','F':'charge_degree_F'})
df.loc[:,'score_text'] = df.score_text.replace({'Low':'sc_low','Medium':'sc_medium','High':'sc_high'})
df.loc[:,'v_score_text'] = df.v_score_text.replace({'Low':'v_score_low','Medium':'v_score_medium','High':'v_score_high'})
df = df[~df.duplicated()]
y = df["two_year_recid"].astype(int)
df = df.drop(['two_year_recid'],axis=1,inplace=False)

X = df.copy()
std = MinMaxScaler(feature_range=(-1,1))
X.iloc[:,7:] = std.fit_transform(X.values[:,7:])
hot_enc = OneHotEncoder(handle_unknown='ignore')
hot_enc.fit(X.iloc[:,:7])
X[np.hstack(hot_enc.categories_).tolist()]=hot_enc.transform(X.iloc[:,:7]).toarray().astype(int)
X.drop(['sex','race','c_charge_degree','score_text','v_score_text','event','is_violent_recid'], axis=1, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=rnd)

### XGBOOST

from xgboost import XGBClassifier
clf_xgb = XGBClassifier(n_estimators=60, use_label_encoder=False, eval_metric='logloss', random_state=rnd)
clf_xgb.fit(X_train, y_train)
pickle.dump(clf_xgb,open('./BlackBoxes/compas_xgboost.p','wb'))

clf_xgb = pickle.load(open('./BlackBoxes/compas_xgboost.p','rb'))
def predict(x, return_proba=False):
    if return_proba:
        return clf_xgb.predict_proba(x)[:,1].ravel()
    else: return clf_xgb.predict(x).ravel().ravel()
y_train_pred = predict(X_train)
y_test_pred = predict(X_test)
print('XGBOOST')
print('train acc:',np.mean(np.round(y_train_pred)==y_train))
print('test acc:',np.mean(np.round(y_test_pred)==y_test))

# ### RF

from sklearn.ensemble import RandomForestClassifier

clf_rf = RandomForestClassifier(max_depth=12,random_state=rnd)
clf_rf.fit(X_train, y_train)

pickle.dump(clf_rf,open('./BlackBoxes/compas_rf.p','wb'))
clf_rf

clf_rf = pickle.load(open('./BlackBoxes/compas_rf.p','rb'))

def predict(x, return_proba=False):
    if return_proba:
        return clf_rf.predict_proba(x)[:,1].ravel()
    else: return clf_rf.predict(x).ravel().ravel()

y_test_pred = predict(X_test, return_proba=True)
y_train_pred = predict(X_train, return_proba=True)
print('RF')
print('train acc:',np.mean(np.round(y_train_pred)==y_train))
print('test acc:',np.mean(np.round(y_test_pred)==y_test))

# ### SVC

from sklearn.svm import SVC
clf_svc = SVC(gamma='auto', probability=True)
clf_svc.fit(X_train, y_train)
pickle.dump(clf_svc,open('./BlackBoxes/compas_svc.p','wb'))

clf_svc = pickle.load(open('./BlackBoxes/compas_svc.p','rb'))

def predict(x, return_proba=False):
    if return_proba:
        return clf_svc.predict_proba(x)[:,1].ravel()
    else: return clf_svc.predict(x).ravel().ravel()

y_train_pred = predict(X_train, return_proba=True)
y_test_pred = predict(X_test, return_proba=True)
print('SVC')
print('train acc:',np.mean(np.round(y_train_pred)==y_train))
print('test acc:',np.mean(np.round(y_test_pred)==y_test))

### NN

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
clf_nn.save_weights('./blackboxes/compas_tf_nn')

from sklearn.metrics import accuracy_score
clf_nn.load_weights('./blackboxes/compas_tf_nn')
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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=rnd)

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

    dataset = pd.concat((y_train,X_train),axis=1)
    d = dice_ml.Data(dataframe=dataset, continuous_features=['age', 'juv_fel_count', 'decile_score', 'juv_misd_count', 'juv_other_count', 'priors_count', 'days_b_screening_arrest', 'c_days_from_compas', 'decile_score.1', 'v_decile_score', 'priors_count.1', 'start', 'end'], 
    outcome_name='two_year_recid')
    
    if black_box == 'nn':
        X = df.copy()
        std = MinMaxScaler(feature_range=(-1,1))
        X.iloc[:,7:] = std.fit_transform(X.values[:,7:])
        hot_enc = OneHotEncoder(handle_unknown='ignore')
        hot_enc.fit(X.iloc[:,:7])
        #X[np.hstack(hot_enc.categories_).tolist()]=hot_enc.transform(X.iloc[:,:7]).toarray().astype(int)
        #X.drop(['sex','race','c_charge_degree','score_text','v_score_text','event','is_violent_recid'], axis=1, inplace=True)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=rnd)

        dataset = pd.DataFrame(np.hstack((y_train.values.reshape(-1,1).astype(int),X_train.values)), columns=['two_year_recid']+list(X_train.columns))
        for i in range(0, 8):
            dataset.iloc[:,i] = dataset.iloc[:,i].astype('str')
        for i in range(8, dataset.values.shape[1]):
            dataset.iloc[:,i] = dataset.iloc[:,i].astype('float')
        d = dice_ml.Data(dataframe=dataset, continuous_features=['age', 'juv_fel_count', 'decile_score', 'juv_misd_count', 'juv_other_count', 'priors_count', 'days_b_screening_arrest', 'c_days_from_compas', 'decile_score.1', 'v_decile_score', 'priors_count.1', 'start', 'end'], 
        outcome_name='two_year_recid')
        m = dice_ml.Model(model=clf_nn, backend='TF2')
    elif black_box == 'xgb':
        m = dice_ml.Model(model=clf_xgb, backend='sklearn')
    elif black_box == 'svc':
        m = dice_ml.Model(model=clf_svc, backend='sklearn')
    elif black_box == 'rf':
        m = dice_ml.Model(model=clf_rf, backend='sklearn')

    exp = dice_ml.Dice(d, m, method='random')

    d_dist_dice    = []
    d_count_dice   = []
    d_impl_dice    = []
    d_adv_dice     = []
    num_dice       = []
    div_dist_dice  = []
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
                                                permitted_range={'age': [-1,1],
                                                                 'juv_fel_count': [-1,1],
                                                                 'decile_score':[-1,1], 
                                                                 'juv_misd_count':[-1,1], 
                                                                 'juv_other_count':[-1,1],
                                                                 'priors_count':[-1,1],
                                                                 'days_b_screening_arrest':[-1,1],
                                                                 'c_days_from_compas':[-1,1],
                                                                 'decile_score.1':[-1,1],
                                                                 'v_decile_score':[-1,1],
                                                                 'priors_count.1':[-1,1],
                                                                 'start':[-1,1],
                                                                 'end':[-1,1]})

        q_cfs_dice = dice_exp.cf_examples_list[0].final_cfs_df.values
        return q_cfs_dice

    from scipy.spatial.distance import cdist, euclidean, hamming

    for idx in tqdm(range(100)):
        query_instances = dataset.iloc[idx:idx+1,1:]
        try: 
            q_cfs_dice = compute_cfs_dice(query_instances)
        except:
            continue
        if black_box == 'nn':
            q_cfs_dice = np.hstack((q_cfs_dice[:,7:-1].astype(float),hot_enc.transform(q_cfs_dice[:,:7]).toarray().astype(int),q_cfs_dice[:,-1].reshape(-1,1).astype(float)))
            query_instances = np.hstack((query_instances.values[:,7:].astype(float),hot_enc.transform(query_instances.values[:,:7]).toarray().astype(int)))
            X_train_enc = np.hstack((X_train.values[:,7:].astype(float),hot_enc.transform(X_train.values[:,:7]).toarray().astype(int))) 
            d_dist_dice.append(np.min(cdist(q_cfs_dice[:,13:-1],query_instances[:,13:],metric='hamming') + cdist(q_cfs_dice[:,:13],query_instances[:,:13],metric='euclidean')))
            d_count_dice.append(np.min(np.sum(q_cfs_dice[:,:-1]!=query_instances,axis=1)))
            pred = predict(query_instances)
            d_impl_dice.append(np.min(cdist(q_cfs_dice[:,13:-1],X_train_enc[:,13:],metric='hamming') + cdist(q_cfs_dice[:,:13],X_train_enc[:,:13],metric='euclidean')))
            r = np.argsort(cdist(q_cfs_dice[:,13:-1],X_train_enc[:,13:],metric='hamming') + cdist(q_cfs_dice[:,:13],X_train_enc[:,:13],metric='euclidean'),axis=1)[:,:10]
            d_adv_dice.append(np.mean(np.array([np.mean(predict(X_train_enc[r,:][i,:])==pred) for i in range(q_cfs_dice.shape[0])])))
        else:
            d_dist_dice.append(np.min(cdist(q_cfs_dice[:,13:-1],query_instances.values[:,13:],metric='hamming') + cdist(q_cfs_dice[:,:13],query_instances.values[:,:13],metric='euclidean')))
            d_count_dice.append(np.min(np.sum(q_cfs_dice[:,:-1]!=query_instances.values,axis=1)))
            pred = predict(query_instances)
            d_impl_dice.append(np.min(cdist(q_cfs_dice[:,13:-1],X_train.values[:,13:],metric='hamming') + cdist(q_cfs_dice[:,:13],X_train.values[:,:13],metric='euclidean')))
            r = np.argsort(cdist(q_cfs_dice[:,13:-1],X_train.values[:,13:],metric='hamming') + cdist(q_cfs_dice[:,:13],X_train.values[:,:13],metric='euclidean'),axis=1)[:,:10]
            d_adv_dice.append(np.mean(np.array([np.mean(predict(X_train.values[r,:][i,:])==pred) for i in range(q_cfs_dice.shape[0])])))
        num_dice.append(len(q_cfs_dice))
        div_dist_dice.append(1/(q_cfs_dice.shape[0]**2)*np.sum(cdist(q_cfs_dice[:,:-1], q_cfs_dice[:,:-1])))
        div_count_dice.append(33/(q_cfs_dice.shape[0]**2)*np.sum(cdist(q_cfs_dice[:,:-1], q_cfs_dice[:,:-1],metric='hamming')))

    with open('./results/compas_results_dice.txt','a') as f:
        f.write('dice '+black_box+'\n')
        f.write(str(np.round(np.mean(d_dist_dice),5))+','+str(np.round(np.std(d_dist_dice),5))+'\n')
        f.write(str(np.round(np.mean(d_count_dice),5))+','+str(np.round(np.std(d_count_dice),5))+'\n')
        f.write(str(np.round(np.mean(d_impl_dice),5))+','+str(np.round(np.std(d_impl_dice),5))+'\n')
        f.write(str(np.round(np.mean(d_adv_dice),5))+','+str(np.round(np.std(d_adv_dice),5))+'\n')
        f.write(str(np.round(np.mean(num_dice),5))+','+str(np.round(np.std(num_dice),5))+'\n')
        f.write(str(np.round(np.mean(div_dist_dice),5))+','+str(np.round(np.std(div_dist_dice),5))+'\n')
        f.write(str(np.round(np.mean(div_count_dice),5))+','+str(np.round(np.std(div_count_dice),5))+'\n')
        f.write('success_rate: '+str(len(d_dist_dice)/100)+'\n')


