import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
pd.set_option('display.max_columns', None)
import time
import warnings
warnings.filterwarnings("ignore")
from scipy.spatial.distance import hamming, euclidean, cdist

import torch
random_seed = 42
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

dataset_name = 'adult'
# number of prototypes to test
n = 10
# also train the latent space, set to false if already trained
latent_train = True
# Latent Space Parameters
latent_dim = 5
batch_size = 1024
sigma = 1
max_epochs = 1000
early_stopping = 3
learning_rate = 1e-3
if dataset_name == 'adult':
    idx_cat = [2,3,4,5,6]
elif dataset_name == 'fico':
    idx_cat = None
elif dataset_name == 'german':
    idx_cat = np.arange(3,71,1).tolist()
elif dataset_name == 'compas':
    idx_cat = list(range(13,33,1))

# LOAD Dataset
from exp.data_loader import load_tabular_data

X_train, X_test, Y_train, Y_test = load_tabular_data(dataset_name)

# load Black Boxes

# XGB
from xgboost import XGBClassifier
clf_xgb = XGBClassifier(n_estimators=60, reg_lambda=3, use_label_encoder=False, eval_metric='logloss')
clf_xgb.fit(X_train, y_train)
pickle.dump(clf_xgb,open(f'./blackboxes/{dataset_name}_xgboost.p','wb'))
clf_xgb = pickle.load(open(f'./blackboxes/{dataset_name}_xgboost.p','rb'))
y_train_pred = clf_xgb.predict(X_train.values)
y_test_pred = clf_xgb.predict(X_test.values)
print('XGB')
print('train acc:',np.mean(np.round(y_train_pred)==Y_train))
print('test acc:',np.mean(np.round(y_test_pred)==Y_test))

#RF
from sklearn.ensemble import RandomForestClassifier
clf_rf = RandomForestClassifier(random_state=random_seed)
clf_rf.fit(X_train, y_train)
pickle.dump(clf_rf,open(f'./blackboxes/{dataset_name}_rf.p','wb'))
clf_rf = pickle.load(open(f'./blackboxes/{dataset_name}_rf.p','rb'))
y_train_pred = clf_rf.predict(X_train)
y_test_pred = clf_rf.predict(X_test)
print('RF')
print('train acc:',np.mean(np.round(y_train_pred)==Y_train))
print('test acc:',np.mean(np.round(y_test_pred)==Y_test))

#SVC
from sklearn.svm import SVC
clf_svc = SVC(gamma='auto', probability=True)
clf_svc.fit(X_train, y_train)
pickle.dump(clf_svc,open(f'./blackboxes/{dataset_name}_svc.p','wb'))
clf_svc = pickle.load(open(f'./blackboxes/{dataset_name}_svc.p','rb'))
y_train_pred = clf_svc.predict(X_train)
y_test_pred = clf_svc.predict(X_test)
print('SVC')
print('train acc:',np.mean(np.round(y_train_pred)==Y_train))
print('test acc:',np.mean(np.round(y_test_pred)==Y_test))

#NN
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
clf_nn.save_weights(f'./blackboxes/{dataset_name}_tf_nn')
from sklearn.metrics import accuracy_score
clf_nn.load_weights(f'./blackboxes/{dataset_name}_tf_nn')
clf_nn.trainable = False
def predict(x, return_proba=False):
    if return_proba:
        return clf_nn.predict(x).ravel()
    else: return np.round(clf_nn.predict(x).ravel()).astype(int).ravel()
print('NN')
print(accuracy_score(np.round(predict(X_train, return_proba = True)),y_train))
print(accuracy_score(np.round(predict(X_test, return_proba = True)),y_test))
print('---------------')

results = {'xgb':{},'rf':{},'svc':{},'nn':{}}

for black_box in ['xgb','rf','svc','nn']:

    if black_box=='xgb':
        def predict(x, return_proba=False):
            if return_proba:
                return clf_xgb.predict_proba(x)[:,1].ravel()
            else: return clf_xgb.predict(x).ravel().ravel()
        y_test_pred = predict(X_test, return_proba=True)
        y_train_pred = predict(X_train, return_proba=True)
        y_train_bb = predict(X_train, return_proba=False)
        y_test_bb = predict(X_test, return_proba=False)
    elif black_box=='rf':
        def predict(x, return_proba=False):
            if return_proba:
                return clf_rf.predict_proba(x)[:,1].ravel()
            else: return clf_rf.predict(x).ravel().ravel()
        y_test_pred = predict(X_test, return_proba=True)
        y_train_pred = predict(X_train, return_proba=True)
        y_train_bb = predict(X_train, return_proba=False)
        y_test_bb = predict(X_test, return_proba=False)
    elif black_box=='svc':
        def predict(x, return_proba=False):
            if return_proba:
                return clf_svc.predict_proba(x)[:,1].ravel()
            else: return clf_svc.predict(x).ravel().ravel()
        y_test_pred = predict(X_test, return_proba=True)
        y_train_pred = predict(X_train, return_proba=True)
        y_train_bb = predict(X_train, return_proba=False)
        y_test_bb = predict(X_test, return_proba=False)
    elif black_box=='nn':
        def predict(x, return_proba=False):
            if return_proba:
                return clf_nn.predict(x).ravel()
            else: return np.round(clf_nn.predict(x).ravel()).astype(int).ravel()
        y_test_pred = predict(X_test, return_proba=True)
        y_train_pred = predict(X_train, return_proba=True)
        y_train_bb = predict(X_train, return_proba=False)
        y_test_bb = predict(X_test, return_proba=False)

    results[black_box]['proto_select'] = {} 
    results[black_box]['proto_dash'] = {} 
    results[black_box]['proto_mmd'] = {} 
    results[black_box]['proto_latent'] = {} 

    # Baseline 1-KNN
    from sklearn.neighbors import KNeighborsClassifier
    neigh = KNeighborsClassifier(n_neighbors=1)
    neigh.fit(X_train, y_train_bb)
    results[black_box]['1knn_baseline'] = accuracy_score(neigh.predict(X_test),y_test_bb)

    for n in [3,4,5,6,7,8,9,10,12,14,16,18,20]:

        print(f'model:{black_box} n:{n}')

        # Proto Select
        from alibi.prototypes import ProtoSelect
        from alibi.utils.kernel import EuclideanDistance
        summariser = ProtoSelect(kernel_distance=EuclideanDistance(), eps=5)
        summariser = summariser.fit(X=X_train.values, y=y_train_bb)
        summary = summariser.summarise(num_prototypes=n)
        proto_select = pd.DataFrame(summary.prototypes, columns=X_train.columns)
        num_proto = len(proto_select)
        proto_pred = predict(proto_select)
        from scipy.spatial.distance import cdist
        knn_1 = np.argmin(cdist(proto_select.values, X_test),axis=0)
        d = {}
        for i in range(num_proto):
            d[i]=proto_pred[i]
        knn_1 = [d[x] for x in knn_1]
        
        results[black_box]['proto_select']['n_'+str(n)] = {} 
        results[black_box]['proto_select']['n_'+str(n)]['proto'] = proto_select
        results[black_box]['proto_select']['n_'+str(n)]['perc_pos'] = np.mean(proto_pred)
        results[black_box]['proto_select']['n_'+str(n)]['acc_1knn'] = accuracy_score(knn_1,y_test_bb)
        results[black_box]['proto_select']['n_'+str(n)]['avg_dist'] = np.mean(cdist(proto_select.values,proto_select.values))

        # ProtoDASH
        from protodash import ProtodashExplainer, get_Gaussian_Data
        explainer = ProtodashExplainer()
        (W, S, _) = explainer.explain(X_train.values, X_train.values, m=n) 
        proto_dash = X_train.iloc[S, :].copy()
        num_proto = len(proto_dash)
        proto_pred = predict(proto_dash)
        from scipy.spatial.distance import cdist
        knn_1 = np.argmin(cdist(proto_dash.values, X_test),axis=0)
        d = {}
        for i in range(num_proto):
            d[i]=proto_pred[i]
        knn_1 = [d[x] for x in knn_1]

        results[black_box]['proto_dash']['n_'+str(n)] = {} 
        results[black_box]['proto_dash']['n_'+str(n)]['proto'] = proto_dash
        results[black_box]['proto_dash']['n_'+str(n)]['perc_pos'] = np.mean(proto_pred)
        results[black_box]['proto_dash']['n_'+str(n)]['acc_1knn'] = accuracy_score(knn_1,y_test_bb)
        results[black_box]['proto_dash']['n_'+str(n)]['avg_dist'] = np.mean(cdist(proto_dash.values,proto_dash.values))

        # MMD Critic
        from mmd.mmd_critic import Dataset, select_prototypes, select_criticisms
        gamma = 0.026
        num_prototypes = n
        num_criticisms = n
        kernel_type = 'local'
        # kernel_type = 'global'
        # regularizer = None
        regularizer = 'logdet'
        # regularizer = 'iterative'
        d_train = Dataset(torch.tensor(X_train.values, dtype=torch.float), torch.tensor(y_train_bb,dtype=torch.long))
        if kernel_type == 'global':
            d_train.compute_rbf_kernel(gamma)
        elif kernel_type == 'local':
            d_train.compute_local_rbf_kernel(gamma)
        else:
            raise KeyError('kernel_type must be either "global" or "local"')
        prototype_indices = select_prototypes(d_train.K, num_prototypes)
        prototypes = d_train.X[prototype_indices]
        prototype_labels = d_train.y[prototype_indices]
        sorted_by_y_indices = prototype_labels.argsort()
        prototypes_sorted = prototypes[sorted_by_y_indices]
        prototype_labels = prototype_labels[sorted_by_y_indices]
        # Criticisms
        criticism_indices = select_criticisms(d_train.K, prototype_indices, num_criticisms, regularizer)
        criticisms = d_train.X[criticism_indices]
        criticism_labels = d_train.y[criticism_indices]
        sorted_by_y_indices = criticism_labels.argsort()
        criticisms_sorted = criticisms[sorted_by_y_indices]
        criticism_labels = criticism_labels[sorted_by_y_indices]
        proto_mmd = X_train.iloc[prototype_indices.sort()[0].tolist()]
        crit_mmd = X_train.iloc[criticism_indices.sort()[0].tolist()]
        proto_pred = predict(proto_mmd)

        num_proto = len(proto_mmd)
        from scipy.spatial.distance import cdist
        knn_1 = np.argmin(cdist(proto_mmd.values, X_test),axis=0)
        d = {}
        for i in range(num_proto):
            d[i]=proto_pred[i]
        knn_1 = [d[x] for x in knn_1]

        results[black_box]['proto_mmd']['n_'+str(n)] = {} 
        results[black_box]['proto_mmd']['n_'+str(n)]['proto'] = proto_mmd
        results[black_box]['proto_mmd']['n_'+str(n)]['crit'] = crit_mmd
        results[black_box]['proto_mmd']['n_'+str(n)]['perc_pos'] = np.mean(proto_pred)
        results[black_box]['proto_mmd']['n_'+str(n)]['acc_1knn'] = accuracy_score(knn_1,y_test_bb)
        results[black_box]['proto_mmd']['n_'+str(n)]['avg_dist'] = np.mean(cdist(proto_mmd.values,proto_mmd.values))

        # --- Latent ---
        X_train_latent = np.hstack((X_train,y_train_pred.reshape(-1,1)))
        X_test_latent = np.hstack((X_test,y_test_pred.reshape(-1,1)))
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import TensorDataset, DataLoader
        # Latent Space Training
        latent_dim = 6
        batch_size = 1024
        sigma = 1
        max_epochs = 1000
        early_stopping = 3
        learning_rate = 1e-3
        idx_cat = list(range(2,17))
        similarity_KLD = torch.nn.KLDivLoss(reduction='batchmean')
        def compute_similarity_Z(Z, sigma):
            D = 1 - F.cosine_similarity(Z[:, None, :], Z[None, :, :], dim=-1)
            M = torch.exp((-D**2)/(2*sigma**2))
            return M / (torch.ones([M.shape[0],M.shape[1]])*(torch.sum(M, axis = 0)-1)).transpose(0,1)
        def compute_similarity_X(X, sigma, idx_cat=None):
            D_class = torch.cdist(X[:,-1].reshape(-1,1),X[:,-1].reshape(-1,1))
            X = X[:, :-1]
            if idx_cat:
                X_cat = X[:, idx_cat]
                X_cont = X[:, np.delete(range(X.shape[1]),idx_cat)]
                h = X_cat.shape[1]
                m = X.shape[1]
                D_cont = 1 - F.cosine_similarity(X[:, None, :], X[None, :, :], dim=-1)
                D_cat = torch.cdist(X_cat, X_cat, p=0)/h
                D = h/m * D_cat + ((m-h)/m) * D_cont + D_class
            else:
                D_features = 1 - F.cosine_similarity(X[:, None, :], X[None, :, :], dim=-1) 
                D = D_features + D_class
            M = torch.exp((-D**2)/(2*sigma**2))
            return M / (torch.ones([M.shape[0],M.shape[1]])*(torch.sum(M, axis = 0)-1)).transpose(0,1)
        def loss_function(X, Z, idx_cat, sigma=1):
            Sx = compute_similarity_X(X, sigma, idx_cat)
            Sz = compute_similarity_Z(Z, sigma)
            loss = similarity_KLD(torch.log(Sx), Sz)
            return loss
        class LinearModel(nn.Module):
            def __init__(self, input_shape, latent_dim):
                super(LinearModel, self).__init__()
                # encoding components
                self.fc1 = nn.Linear(input_shape, latent_dim)
            def encode(self, x):
                x = self.fc1(x)
                return x
            def forward(self, x):
                z = self.encode(x)
                return z
        # Create Model
        model = LinearModel(X_train_latent.shape[1], latent_dim=latent_dim)
        model.load_state_dict(torch.load(f'./models/adult_latent_{black_box}_{latent_dim}.pt'))
        with torch.no_grad():
            model.eval()
            Z_train = model(torch.tensor(X_train_latent).float()).cpu().detach().numpy()
            Z_test = model(torch.tensor(X_test_latent).float()).cpu().detach().numpy()
        
        # Latent Clustering
        from sklearn.cluster import SpectralClustering
        Z_train_0 = Z_train[y_train_bb==0]
        Z_train_1 = Z_train[y_train_bb==1]

    
        clustering_0 = SpectralClustering(n_clusters=int(n//(1/(y_train_bb==0/len(y_train_bb)))),assign_labels='discretize').fit(Z_train_0)
        clustering_1 = SpectralClustering(n_clusters=int(n-n//(1/(y_train_bb==0/len(y_train_bb)))),assign_labels='discretize').fit(Z_train_1)
        centers = []
        for i in range(int(n//(1/(y_train_bb==0/len(y_train_bb))))):
            centers.append(np.mean(Z_train_0[clustering_0.labels_==i],axis=0))
        for i in range(int(n-n//(1/(y_train_bb==0/len(y_train_bb))))):
            centers.append(np.mean(Z_train_1[clustering_1.labels_==i],axis=0))
        centers = np.stack(centers)
        from scipy.spatial.distance import cdist
        idx = np.argmin(cdist(centers,Z_train),axis=1)
        proto_latent_clustering = pd.DataFrame(X_train_latent[idx,:-1],columns=X_train.columns)
        proto_pred = predict(proto_latent_clustering)
        from scipy.spatial.distance import cdist
        knn_1 = np.argmin(cdist(proto_latent_clustering.values, X_test_latent[:,:-1]),axis=0)
        d = {}
        for i in range(n):
            d[i]=proto_pred[i]
        knn_1 = [d[x] for x in knn_1]

        results[black_box]['proto_latent']['n_'+str(n)] = {} 
        results[black_box]['proto_latent']['n_'+str(n)]['proto'] = proto_latent_clustering
        results[black_box]['proto_latent']['n_'+str(n)]['perc_pos'] = np.mean(proto_pred)
        results[black_box]['proto_latent']['n_'+str(n)]['acc_1knn'] = accuracy_score(knn_1,y_test_bb)
        results[black_box]['proto_latent']['n_'+str(n)]['avg_dist'] = np.mean(cdist(proto_latent_clustering.values,proto_latent_clustering.values))

        pickle.dump(results,open('results_proto_adult.pickle','wb'))
    