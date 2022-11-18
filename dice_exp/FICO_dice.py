import tensorflow as tf
tf.compat.v1.enable_eager_execution()
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import numpy as np
import os
import pandas as pd
import pickle
pd.set_option('display.max_columns', None)
import time
import warnings
warnings.filterwarnings("ignore")

import torch
random_seed = 42
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv('../data/heloc_dataset.csv')
df['RiskPerformance']=(df.RiskPerformance=='Bad')+0

scaler = MinMaxScaler((-1,1))
X = scaler.fit_transform(df.values[:,1:])
y = df['RiskPerformance'].values

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# # BlackBoxes

# ### XGBOOST

from xgboost import XGBClassifier

clf_xgb = XGBClassifier(n_estimators=60, reg_lambda=3, use_label_encoder=False, eval_metric='logloss')

#clf_xgb.fit(X_train, Y_train)
#clf_xgb.save_model('./BlackBoxes/fico_xgboost')

clf_xgb.load_model('../blackboxes/fico_xgboost')
y_train_pred = clf_xgb.predict(X_train)
y_test_pred = clf_xgb.predict(X_test)
print('XGBOOST')
print('train acc:',np.mean(np.round(y_train_pred)==Y_train))
print('test acc:',np.mean(np.round(y_test_pred)==Y_test))

# ### Random Forest

from sklearn.ensemble import RandomForestClassifier

#clf_rf = RandomForestClassifier(random_state=random_seed)
#clf_rf.fit(X_train, Y_train)

#pickle.dump(clf_rf,open('../blackboxes/fico_rf.p','wb'))

clf_rf = pickle.load(open('../blackboxes/fico_rf.p','rb'))
y_train_pred = clf_rf.predict(X_train)
y_test_pred = clf_rf.predict(X_test)
print('RF')
print('train acc:',np.mean(np.round(y_train_pred)==Y_train))
print('test acc:',np.mean(np.round(y_test_pred)==Y_test))

# ### SVC

#from sklearn.svm import SVC
#clf_svc = SVC(gamma='auto', probability=True)
#clf_svc.fit(X_train, Y_train)

#pickle.dump(clf_svc,open('../blackboxes/fico_svc.p','wb'))

clf_svc = pickle.load(open('../blackboxes/fico_svc.p','rb'))
y_train_pred = clf_svc.predict(X_train)
y_test_pred = clf_svc.predict(X_test)
print('SVC')
print('train acc:',np.mean(np.round(y_train_pred)==Y_train))
print('test acc:',np.mean(np.round(y_test_pred)==Y_test))

# ### NN tf

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping

BATCH_SIZE = 1024
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).shuffle(2048).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).batch(BATCH_SIZE)

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
    epochs=500,
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

#plot_metric(history, 'loss')
clf_nn.save_weights('../blackboxes/fico_tf_nn')

from sklearn.metrics import accuracy_score
clf_nn.load_weights('../blackboxes/fico_tf_nn')
clf_nn.trainable = False
print('NN')
print(accuracy_score(np.round(clf_nn.predict(X_train)),Y_train))
print(accuracy_score(np.round(clf_nn.predict(X_test)),Y_test))
print('---------------')

# # Latent Space

for black_box in ['xgb', 'rf', 'svc', 'nn']:

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.33, random_state=42)

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

    from scipy.spatial.distance import euclidean, cdist

    # # DICE

    import dice_ml

    dataset = pd.DataFrame(np.hstack((Y_train.reshape(-1,1),X_train)), columns=list(df.columns))
    d = dice_ml.Data(dataframe=dataset, continuous_features=list(dataset.columns[1:]), outcome_name='RiskPerformance')

    if black_box == 'nn':
        m = dice_ml.Model(model=clf_nn, backend='TF2')
    elif black_box == 'rf':
        m = dice_ml.Model(model=clf_rf, backend='sklearn')
    elif black_box == 'svc':
        m = dice_ml.Model(model=clf_svc, backend='sklearn')
    elif black_box == 'xgb':
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
                                                verbose=False)
        q_cfs_dice = dice_exp.cf_examples_list[0].final_cfs_df.values
        return q_cfs_dice

    for i in tqdm(range(100)):
        query_instances = dataset.iloc[i:i+1,1:]
        pred = predict(query_instances)
        try:
            q_cfs_dice = compute_cfs_dice(query_instances)
            d_dist_dice.append(np.min(cdist(q_cfs_dice[:,:-1],query_instances.values)))
            d_count_dice.append(np.min(np.sum(q_cfs_dice[:,:-1]!=query_instances.values,axis=1)))
            d_impl_dice.append(np.min(cdist(q_cfs_dice[:,:-1],X_train)))
            r = np.argsort(cdist(q_cfs_dice[:,:-1],X_train),axis=1)[:,:10]
            d_adv_dice.append(np.mean(np.array([np.mean(predict(X_train[r,:][i,:])==pred) for i in range(q_cfs_dice.shape[0])])))
            num_dice.append(len(q_cfs_dice))
            div_dist_dice.append(1/(q_cfs_dice.shape[0]**2)*np.sum(cdist(q_cfs_dice[:,:-1],q_cfs_dice[:,:-1])))
            div_count_dice.append(23/(q_cfs_dice.shape[0]**2)*np.sum(cdist(q_cfs_dice[:,:-1], q_cfs_dice[:,:-1],metric='hamming')))
        except:
            continue
    
    with open('../results/fico_results_dice.txt','a') as f:
        f.write('dice '+black_box+'\n')
        f.write(str(np.round(np.mean(d_dist_dice),5))+','+str(np.round(np.std(d_dist_dice),5))+'\n')
        f.write(str(np.round(np.mean(d_count_dice),5))+','+str(np.round(np.std(d_count_dice),5))+'\n')
        f.write(str(np.round(np.mean(d_impl_dice),5))+','+str(np.round(np.std(d_impl_dice),5))+'\n')
        f.write(str(np.round(np.mean(d_adv_dice),5))+','+str(np.round(np.std(d_adv_dice),5))+'\n')
        f.write(str(np.round(np.mean(num_dice),5))+','+str(np.round(np.std(num_dice),5))+'\n')
        f.write(str(np.round(np.mean(div_dist_dice),5))+','+str(np.round(np.std(div_dist_dice),5))+'\n')
        f.write(str(np.round(np.mean(div_count_dice),5))+','+str(np.round(np.std(div_count_dice),5))+'\n')
        f.write('success_rate: '+str(len(d_dist_dice)/100)+'\n')









