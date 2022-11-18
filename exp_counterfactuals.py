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

dataset_name = 'german'
# number of counterfactuals to test
n = 100
# also train the latent space, set to false if already trained
latent_train = True
# Latent Space Parameters
latent_dim = 3
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
elif dataset_name == 'diva':
    idx_cat = list(range(58))

# LOAD Dataset
from exp.data_loader import load_tabular_data

X_train, X_test, Y_train, Y_test = load_tabular_data(dataset_name)

# load Black Boxes

# XGB
from xgboost import XGBClassifier
clf_xgb = XGBClassifier(n_estimators=60, reg_lambda=3, use_label_encoder=False, eval_metric='logloss')
clf_xgb.fit(X_train.values, Y_train.values)
clf_xgb.save_model(f'./blackboxes/{dataset_name}_xgboost')
clf_xgb.load_model(f'./blackboxes/{dataset_name}_xgboost')
y_train_pred = clf_xgb.predict(X_train.values)
y_test_pred = clf_xgb.predict(X_test.values)
print('XGB')
print('train acc:',np.mean(np.round(y_train_pred)==Y_train))
print('test acc:',np.mean(np.round(y_test_pred)==Y_test))

#RF
from sklearn.ensemble import RandomForestClassifier
clf_rf = RandomForestClassifier(random_state=random_seed)
clf_rf.fit(X_train, Y_train)
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
clf_svc.fit(X_train, Y_train)
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
print(accuracy_score(np.round(clf_nn.predict(X_train)),Y_train))
print(accuracy_score(np.round(clf_nn.predict(X_test)),Y_test))

d = {}

for black_box in ['xgb', 'rf', 'svc', 'nn']:
    
    d[dataset_name]={bb_name:{}}
    print(dataset_name)
    print(black_box)

    X_train, X_test, Y_train, Y_test = load_tabular_data(dataset_name)

    if black_box=='xgb':
        def predict(x, return_proba=False):
            if return_proba:
                return clf_xgb.predict_proba(x)[:,1].ravel()
            else: return clf_xgb.predict(x).ravel().ravel()
        y_test_pred = predict(X_test.values, return_proba=True)
        y_train_pred = predict(X_train.values, return_proba=True)
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

    X_train = np.hstack((X_train,y_train_pred.reshape(-1,1)))
    X_test = np.hstack((X_test,y_test_pred.reshape(-1,1)))

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import TensorDataset, DataLoader

    if latent_train:
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
        model = LinearModel(X_train.shape[1], latent_dim=latent_dim)
        
        train_dataset = TensorDataset(torch.tensor(X_train).float())
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 
        test_dataset = TensorDataset(torch.tensor(X_test).float())
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) 
    
        def check_and_clear(dir_name):
            if not os.path.exists(dir_name):
                os.mkdir(dir_name)
            else:
                os.system('rm -r ' + dir_name)
                os.mkdir(dir_name)
        
        check_and_clear('./models/weights')
        
        model_params = list(model.parameters())
        optimizer = torch.optim.Adam(model_params, lr=learning_rate)
        
        # record training process
        epoch_train_losses = []
        epoch_test_losses = []
        
        #validation parameters
        epoch = 1
        best = np.inf
        
        # progress bar
        pbar = tqdm(bar_format="{postfix[0]} {postfix[1][value]:03d} {postfix[2]} {postfix[3][value]:.5f} {postfix[4]} {postfix[5][value]:.5f} {postfix[6]} {postfix[7][value]:d}",
                postfix=["Epoch:", {'value':0}, "Train Sim Loss", {'value':0}, "Test Sim Loss", {'value':0}, "Early Stopping", {"value":0}])
        
        # start training
        while epoch <= max_epochs:
        
            # ------- TRAIN ------- #
            # set model as training mode
            model.train()
            batch_loss = []
            
            for batch, (X_batch,) in enumerate(train_loader):
                optimizer.zero_grad()
                Z_batch = model(X_batch)  #
                loss  = loss_function(X_batch, Z_batch, idx_cat, sigma) 
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            
            # save result
            epoch_train_losses.append(np.mean(batch_loss))
            pbar.postfix[3]["value"] = np.mean(batch_loss)
        
            # -------- VALIDATION --------
            
            # set model as testing mode
            model.eval()
            batch_loss = []
            
            with torch.no_grad():
                for batch, (X_batch,) in enumerate(test_loader):
                    Z_batch = model(X_batch)
                    loss = loss_function(X_batch, Z_batch, idx_cat, sigma)
                    batch_loss.append(loss.item())
            
            # save information
            epoch_test_losses.append(np.mean(batch_loss))
            pbar.postfix[5]["value"] = np.mean(batch_loss)
            pbar.postfix[1]["value"] = epoch
            
            if epoch_test_losses[-1] < best:
                wait = 0
                best = epoch_test_losses[-1]
                best_epoch = epoch
                torch.save(model.state_dict(), f'./models/weights/LinearTransparent_{dataset_name}.pt')
            else:
                wait += 1
            pbar.postfix[7]["value"] = wait
            if wait == early_stopping:
                break    
            epoch += 1
            pbar.update()
        
        model.load_state_dict(torch.load(f'./models/weights/LinearTransparent_{dataset_name}.pt'))
        with torch.no_grad():
            model.eval()
            Z_train = model(torch.tensor(X_train).float()).cpu().detach().numpy()
            Z_test = model(torch.tensor(X_test).float()).cpu().detach().numpy()
    
        torch.save(model.state_dict(), f'./models/{dataset_name}_latent_{black_box}_{latent_dim}.pt')
    
    model.load_state_dict(torch.load(f'./models/{dataset_name}_latent_{black_box}_{latent_dim}.pt'))
    with torch.no_grad():
       model.eval()
       Z_train = model(torch.tensor(X_train).float()).cpu().detach().numpy()
       Z_test = model(torch.tensor(X_test).float()).cpu().detach().numpy()
    
    plt.scatter(Z_train[:,0], Z_train[:,1], c=y_train_pred, cmap='coolwarm')
    plt.grid()
    
    w = model.fc1.weight.detach().numpy()
    b = model.fc1.bias.detach().numpy()
    y_contrib = model.fc1.weight.detach().numpy()[:,-1]
    
    def compute_cf(q, indexes):
       q_pred = predict(q[:-1].reshape(1,-1),return_proba=True)
       q_cf = q.copy()
       q_cf_preds = []
       q_cf_preds.append(float(predict(q_cf[:-1].reshape(1,-1),return_proba=True)))
       if q_pred > 0.5:
           m = -0.1
       else:
           m = +0.1
       while np.round(q_pred) == np.round(q_cf_preds[-1]):
           v = np.array(model(torch.tensor(q_cf).float()).detach().numpy()+m*y_contrib)
           c_l = [v[l] - np.sum(q_cf*w[l,:]) - b[l] for l in range(latent_dim)]
           M = []
           for l in range(latent_dim):
               M.append([np.sum(w[k,indexes]*w[l,indexes]) for k in range(latent_dim)])
           M = np.vstack(M)
           lambda_k = np.linalg.solve(M, c_l)
           delta_i = [np.sum(lambda_k*w[:,i]) for i in indexes]
           q_cf[indexes] += delta_i
           #q_cf = np.clip(q_cf,-1,1)
           if float(predict(q_cf[:-1].reshape(1,-1),return_proba=True)) in q_cf_preds:
               return q_cf
           q_cf_preds.append(float(predict(q_cf[:-1].reshape(1,-1),return_proba=True)))
           q_cf[-1] = q_cf_preds[-1]
       return q_cf
    
    from itertools import combinations
    from scipy.spatial.distance import cdist
    
    d_dist = []
    d_impl = []
    d_count = []
    d_adv = []
    num = []
    div_dist = []
    div_count = []
    
    for idx in tqdm(range(n)):
       q = X_test[idx,:].copy()
       q_pred = predict(q[:-1].reshape(1,-1),return_proba=False)
       q_cfs = []
       l_i = []
       l_f = []
    
       for indexes in list(combinations(list(range(X_train.shape[1]-1)),1)):    
           q_cf = compute_cf(q, list(indexes))
           q_cf_pred = predict(q_cf[:-1].reshape(1,-1),return_proba=True)
           if q_pred:
               if q_cf_pred<0.5:
                   q_cfs.append(q_cf)
           else:
               if q_cf_pred>0.5:
                   q_cfs.append(q_cf) 
    
       for indexes in list(combinations(list(range(X_train.shape[1]-1)),2)):    
           q_cf = compute_cf(q, list(indexes))
           q_cf_pred = predict(q_cf[:-1].reshape(1,-1),return_proba=True)
           if q_pred:
               if q_cf_pred<0.5:
                   q_cfs.append(q_cf)
           else:
               if q_cf_pred>0.5:
                   q_cfs.append(q_cf) 
           l_i.append([list(indexes),q_cf_pred])
       r = np.argsort(np.stack(np.array(l_i,dtype=object)[:,1]).ravel())[-10:]
       l_i = np.array(l_i,dtype=object)[r,0]
    
       while len(l_i[0])<6:
           for e in l_i:
               for i in list(np.delete(range(X_train.shape[1]-1),e)):
                   q_cf = compute_cf(q, e+[i])
                   q_cf_pred = predict(q_cf[:-1].reshape(1,-1),return_proba=True)
                   if q_pred:
                       if q_cf_pred<0.5:
                           q_cfs.append(q_cf)
                   else:
                       if q_cf_pred>0.5:
                           q_cfs.append(q_cf) 
                   l_f.append([e+[i],q_cf_pred])
           r = np.argsort(np.stack(np.array(l_f,dtype=object)[:,1]).ravel())[-10:]
           l_f = np.array(l_f,dtype=object)[r,0]
           l_i = l_f.copy()
           l_f = []
       
       if len(q_cfs)<1:
           continue
       else:
           q_cfs = np.vstack(q_cfs)
           if dataset_name == 'fico':
               d_dist.append(np.min(cdist(q_cfs[:,:-1],q[:-1].reshape(1,-1))))
               d_impl.append(np.min(cdist(q_cfs[:,:-1],X_train[:,:-1])))
               d_count.append(np.min(np.sum(q_cfs[:,:-1]!=q[:-1],axis=1)))
               r = np.argsort(cdist(q_cfs[:,:-1],X_train[:,:-1]),axis=1)[:,:10]
               d_adv.append(np.mean(np.array([np.mean(predict(X_train[r,:-1][i,:])==q_pred) for i in range(q_cfs.shape[0])])))
               num.append(len(q_cfs))
               div_dist.append(1/(q_cfs.shape[0]**2)*np.sum(cdist(q_cfs[:,:-1],q_cfs[:,:-1])))
               div_count.append((X_train.shape[1]-1)/(q_cfs.shape[0]**2)*np.sum(cdist(q_cfs[:,:-1], q_cfs[:,:-1],metric='hamming')))
           elif dataset_name == 'adult':
               d_dist.append(np.min(cdist(q_cfs[:,[2,3,4,5,6]],q[[2,3,4,5,6]].reshape(1,-1),metric='hamming') + cdist(q_cfs[:,[0,1]],q[[0,1]].reshape(1,-1),metric='euclidean')))
               d_impl.append(np.min(cdist(q_cfs[:,[2,3,4,5,6]],X_train[:,[2,3,4,5,6]],metric='hamming') + cdist(q_cfs[:,[0,1]],X_train[:,[0,1]],metric='euclidean')))
               d_count.append(np.min(np.sum(q_cfs[:,:-1]!=q[:-1],axis=1)))
               r = np.argsort(cdist(q_cfs[:,[2,3,4,5,6]],X_train[:,[2,3,4,5,6]],metric='hamming') + cdist(q_cfs[:,[0,1]],X_train[:,[0,1]],metric='euclidean'),axis=1)[:,:10]
               d_adv.append(np.mean(np.array([np.mean(predict(X_train[r,:-1][i,:])==q_pred) for i in range(q_cfs.shape[0])])))
               num.append(len(q_cfs))
               div_dist.append(np.mean(cdist(q_cfs[:,[2,3,4,5,6]],q_cfs[:,[2,3,4,5,6]],metric='hamming') + cdist(q_cfs[:,[0,1]],q_cfs[:,[0,1]],metric='euclidean')))
               div_count.append(X_train.shape[1]-1/(q_cfs.shape[0]**2)*np.sum(cdist(q_cfs[:,:-1], q_cfs[:,:-1],metric='hamming')))
           elif dataset_name == 'compas':
               d_dist.append(np.min(cdist(q_cfs[:,13:-1],q[13:-1].reshape(1,-1),metric='hamming') + cdist(q_cfs[:,:13],q[:13].reshape(1,-1),metric='euclidean')))
               d_count.append(np.min(np.sum(q_cfs[:,:-1]!=q[:-1],axis=1)))
               d_impl.append(np.min(cdist(q_cfs[:,13:-1],X_train[:,13:-1],metric='hamming') + cdist(q_cfs[:,:13],X_train[:,:13],metric='euclidean')))
               r = np.argsort(cdist(q_cfs[:,13:-1],X_train[:,13:-1],metric='hamming') + cdist(q_cfs[:,:13],X_train[:,:13],metric='euclidean'),axis=1)[:,:10]
               d_adv.append(np.mean(np.array([np.mean(predict(X_train[r,:-1][i,:])==q_pred) for i in range(q_cfs.shape[0])])))
               num.append(len(q_cfs))
               div_dist.append(np.mean(cdist(q_cfs[:,13:-1],q_cfs[:,13:-1],metric='hamming') + cdist(q_cfs[:,:13],q_cfs[:,:13],metric='euclidean')))
               div_count.append(X_train.shape[1]-1/(q_cfs.shape[0]**2)*np.sum(cdist(q_cfs[:,:-1], q_cfs[:,:-1],metric='hamming')))
           elif dataset_name == 'german':
               d_dist.append(np.min(cdist(q_cfs[:,3:-1],q[3:-1].reshape(1,-1),metric='hamming') + cdist(q_cfs[:,:3],q[:3].reshape(1,-1),metric='euclidean')))
               d_impl.append(np.min(cdist(q_cfs[:,3:-1],X_train[:,3:-1],metric='hamming') + cdist(q_cfs[:,:3],X_train[:,:3],metric='euclidean')))
               d_count.append(np.min(np.sum(q_cfs[:,:-1]!=q[:-1],axis=1)))
               num.append(len(q_cfs))
               div_dist.append(np.mean(cdist(q_cfs[:,3:-1],q_cfs[:,3:-1],metric='hamming') + cdist(q_cfs[:,:3],q_cfs[:,:3],metric='euclidean')))
               div_count.append(X_train.shape[1]-1/(q_cfs.shape[0]**2)*np.sum(cdist(q_cfs[:,:-1], q_cfs[:,:-1],metric='hamming')))

    d[dataset_name][bb_name][str(latent_dim)]['d_dist_mean'] = np.mean(np.array(d_dist))
    d[dataset_name][bb_name][str(latent_dim)]['d_dist_std'] = np.std(np.array(d_dist))
    d[dataset_name][bb_name][str(latent_dim)]['d_count_mean'] = np.mean(np.array(d_count))
    d[dataset_name][bb_name][str(latent_dim)]['d_count_std'] = np.std(np.array(d_count))
    d[dataset_name][bb_name][str(latent_dim)]['d_impl_mean'] = np.mean(np.array(d_impl)) 
    d[dataset_name][bb_name][str(latent_dim)]['d_impl_std'] = np.std(np.array(d_impl))
    d[dataset_name][bb_name][str(latent_dim)]['d_advs_mean'] = np.mean(np.array(d_adv)) 
    d[dataset_name][bb_name][str(latent_dim)]['d_advs_std'] = np.std(np.array(d_adv))
    d[dataset_name][bb_name][str(latent_dim)]['div_count_mean'] = np.mean(np.array(div_dist)) 
    d[dataset_name][bb_name][str(latent_dim)]['div_count_std'] = np.std(np.array(div_dist))
    d[dataset_name][bb_name][str(latent_dim)]['div_dist_mean'] = np.mean(np.array(div_count)) 
    d[dataset_name][bb_name][str(latent_dim)]['div_dist_std'] = np.std(np.array(div_count))
    pickle.dump(d, open(f'./{dataset_name}_{black_box}_results.p','wb'))

    # ---- Growing Sphere -----
    from growingspheres import counterfactuals as cf

    d_dist_GS = []
    d_count_GS = []
    d_impl_GS = []
    d_adv_GS = []

    for idx in tqdm(range(n)):
        q = X_test[idx,:-1].reshape(1,-1).copy()
        pred = int(predict(q))
        CF = cf.CounterfactualExplanation(q, predict, method='GS')
        CF.fit(n_in_layer=2000, first_radius=0.1, dicrease_radius=10, sparse=True, verbose=False)
        q_cf_GS = CF.enemy
        if dataset_name == 'adult':
            d_dist_GS.append(float(cdist(q_cf_GS[[2,3,4,5,6]].reshape(1,-1),q[:,[2,3,4,5,6]],metric='hamming') + cdist(q_cf_GS[[0,1]].reshape(1,-1),q[:,[0,1]],metric='euclidean')))
            d_count_GS.append(np.sum(q_cf_GS!=q))
            d_impl_GS.append(np.min(cdist(q_cf_GS[[2,3,4,5,6]].reshape(1,-1),X_train[:,[2,3,4,5,6]],metric='hamming') + cdist(q_cf_GS[[0,1]].reshape(1,-1),X_train[:,[0,1]],metric='euclidean')))
            r = np.argsort(cdist(q_cf_GS[[2,3,4,5,6]].reshape(1,-1),X_train[:,[2,3,4,5,6]],metric='hamming') + cdist(q_cf_GS[[0,1]].reshape(1,-1),X_train[:,[0,1]],metric='euclidean'),axis=1)[:,:10]
            d_adv_GS.append(np.mean(np.array([np.mean(predict(X_train[r,:-1][i,:])==pred) for i in range(q_cf_GS.reshape(1,-1).shape[0])])))
        elif dataset_name == 'fico':
            d_dist_GS.append(euclidean(q_cf_GS,q))
            d_count_GS.append(np.sum(q_cf_GS!=q))
            d_impl_GS.append(np.min(cdist(q_cf_GS.reshape(1,-1),X_train[:,:-1])))
            r = np.argsort(cdist(q_cf_GS.reshape(1,-1),X_train[:,:-1],metric='euclidean'),axis=1)[:,:10]
            d_adv_GS.append(np.mean(np.array([np.mean(predict(X_train[r,:-1][i,:])==pred) for i in range(q_cf_GS.reshape(1,-1).shape[0])])))
        elif dataset_name == 'german':
            d_dist_GS.append(float(cdist(q_cf_GS[3:].reshape(1,-1),q[:,3:],metric='hamming') + cdist(q_cf_GS[:3].reshape(1,-1),q[:,:3],metric='euclidean')))
            d_count_GS.append(np.sum(q_cf_GS!=q.ravel()))
            d_impl_GS.append(np.min(cdist(q_cf_GS[3:].reshape(1,-1),X_train[:,3:-1],metric='hamming') + cdist(q_cf_GS[:3].reshape(1,-1),X_train[:,:3],metric='euclidean')))
            r = np.argsort(cdist(q_cf_GS[3:].reshape(1,-1),X_train[:,3:-1],metric='hamming') + cdist(q_cf_GS[:3].reshape(1,-1),X_train[:,:3],metric='euclidean'),axis=1)[:,:10]
            d_adv_GS.append(np.mean(np.array([np.mean(predict(X_train[r,:-1][i,:])==pred) for i in range(q_cf_GS.reshape(1,-1).shape[0])])))
        elif dataset_name == 'compas':
            d_dist_GS.append(float(cdist(q_cf_GS[13:].reshape(1,-1),q[:,13:],metric='hamming') + cdist(q_cf_GS[:13].reshape(1,-1),q[:,:13],metric='euclidean')))
            d_count_GS.append(np.sum(q_cf_GS!=q))
            d_impl_GS.append(np.min(cdist(q_cf_GS[13:].reshape(1,-1),X_train[:,13:-1],metric='hamming') + cdist(q_cf_GS[:13].reshape(1,-1),X_train[:,:13],metric='euclidean')))
            r = np.argsort(cdist(q_cf_GS[13:].reshape(1,-1),X_train[:,13:-1],metric='hamming') + cdist(q_cf_GS[:13].reshape(1,-1),X_train[:,:13],metric='euclidean'),axis=1)[:,:10]
            d_adv_GS.append(np.mean(np.array([np.mean(predict(X_train[r,:-1][i,:])==pred) for i in range(q_cf_GS.reshape(1,-1).shape[0])])))

    d[dataset_name][bb_name][str(latent_dim)]['d_dist_GS_mean'] = np.mean(d_dist_GS)
    d[dataset_name][bb_name][str(latent_dim)]['d_dist_GS_std'] = np.std(d_dist_GS)
    d[dataset_name][bb_name][str(latent_dim)]['d_count_GS_mean'] = np.mean(d_count_GS)
    d[dataset_name][bb_name][str(latent_dim)]['d_count_GS_std'] = np.std(d_count_GS)
    d[dataset_name][bb_name][str(latent_dim)]['d_impl_GS_mean'] = np.mean(d_impl_GS)
    d[dataset_name][bb_name][str(latent_dim)]['d_impl_GS_std'] = np.std(d_impl_GS)
    d[dataset_name][bb_name][str(latent_dim)]['d_advs_GS_mean'] = np.mean(d_count_GS)
    d[dataset_name][bb_name][str(latent_dim)]['d_advs_GS_std'] = np.std(d_count_GS)
    d[dataset_name][bb_name][str(latent_dim)]['GS_success_rate'] = len(d_dist_GS)/n
    pickle.dump(d, open(f'./{dataset_name}_{black_box}_results.p','wb'))

    # ---- Watcher ----
    from scipy.spatial.distance import cdist, euclidean
    from scipy.optimize import minimize
    from scipy import stats

    d_dist_watch = []
    d_count_watch = []
    d_impl_watch = []
    d_adv_watch = []

    for i in tqdm(range(n)):
        # initial conditions
        lamda = 0.1 
        x0 = np.zeros([1,X_train.shape[1]-1]) # initial guess for cf
        q = X_test[i:i+1,:-1].copy()
        pred = predict(q,return_proba=False)

        def dist_mad(cf, eg):
            manhat = [cdist(eg.T, cf.reshape(1,-1).T, metric='cityblock')[i][i] for i in range(len(eg.T))]
            #mad = stats.median_absolute_deviation(X_train)
            return sum(manhat)

        def loss_function_mad(x_dash):
            target = 1-pred
            if target == 0:
                L = lamda*(predict(x_dash.reshape(1,-1),return_proba=True)-1)**2 + dist_mad(x_dash.reshape(1,-1), q)
            else:
                L = lamda*(1-predict(x_dash.reshape(1,-1),return_proba=True)-1)**2 + dist_mad(x_dash.reshape(1,-1), q) 
            return L

        res = minimize(loss_function_mad, x0, method='nelder-mead', options={'maxiter':100, 'xatol': 1e-6})
        cf = res.x.reshape(1, -1)

        i = 0
        r = 1
        while pred == predict(cf):
            lamda += 0.1
            x0 = cf 
            res = minimize(loss_function_mad, x0, method='nelder-mead', options={'maxiter':100, 'xatol': 1e-6})
            cf = res.x.reshape(1, -1)
            i += 1
            if i == 100:
                r = 0
                break

        if r == 1:
            if dataset_name == 'adult':
                d_dist_watch.append(float(cdist(cf[:,[2,3,4,5,6]],q[:,[2,3,4,5,6]],metric='hamming') + cdist(cf[:,[0,1]],q[:,[0,1]],metric='euclidean')))
                d_count_watch.append(np.sum(cf!=q))
                d_impl_watch.append(np.min(cdist(cf[:,[2,3,4,5,6]],X_train[:,[2,3,4,5,6]],metric='hamming') + cdist(cf[:,[0,1]],X_train[:,[0,1]],metric='euclidean')))
                r = np.argsort(cdist(cf[:,[2,3,4,5,6]],X_train[:,[2,3,4,5,6]],metric='hamming') + cdist(cf[:,[0,1]],X_train[:,[0,1]],metric='euclidean'),axis=1)[:,:10]
                d_adv_watch.append(np.mean(np.array([np.mean(predict(X_train[r,:-1][i,:])==pred) for i in range(cf.shape[0])])))
            elif dataset_name == 'fico':
                d_dist_watch.append(euclidean(cf,q))
                d_count_watch.append(np.sum(cf!=q))
                d_impl_watch.append(np.min(cdist(cf.reshape(1,-1),X_train[:,:-1])))
                r = np.argsort(cdist(cf.reshape(1,-1),X_train[:,:-1],metric='euclidean'),axis=1)[:,:10]
                d_adv_watch.append(np.mean(np.array([np.mean(predict(X_train[r,:-1][i,:])==pred) for i in range(q_cf_GS.reshape(1,-1).shape[0])])))
            elif dataset_name == 'german':
                d_dist_watch.append(float(cdist(cf[:,3:],q[:,3:],metric='hamming') + cdist(cf[:,:3],q[:,:3],metric='euclidean')))
                d_count_watch.append(np.sum(cf!=q.ravel()))
                d_impl_watch.append(np.min(cdist(cf[:,3:],X_train[:,3:-1],metric='hamming') + cdist(cf[:,:3],X_train[:,:3],metric='euclidean')))
                r = np.argsort(cdist(cf[:,3:],X_train[:,3:-1],metric='hamming') + cdist(cf[:,:3],X_train[:,:3],metric='euclidean'),axis=1)[:,:10]
                d_adv_watch.append(np.mean(np.array([np.mean(predict(X_train[r,:-1][i,:])==pred) for i in range(q_cf_GS.reshape(1,-1).shape[0])])))
            elif dataset_name == 'compas':
                d_dist_watch.append(float(cdist(cf[:,13:],q[:,13:],metric='hamming') + cdist(cf[:,:13],q[:,:13],metric='euclidean')))
                d_count_watch.append(np.sum(q_cf_GS!=q))
                d_impl_watch.append(np.min(cdist(cf[:,13:],X_train[:,13:-1],metric='hamming') + cdist(cf[:,:13],X_train[:,:13],metric='euclidean')))
                r = np.argsort(cdist(cf[:,13:],X_train[:,13:-1],metric='hamming') + cdist(cf[:,:13],X_train[:,:13],metric='euclidean'),axis=1)[:,:10]
                d_adv_watch.append(np.mean(np.array([np.mean(predict(X_train[r,:-1][i,:])==pred) for i in range(q_cf_GS.reshape(1,-1).shape[0])])))

    d[dataset_name][bb_name][str(latent_dim)]['d_dist_watch_mean'] = np.mean(d_dist_watch)
    d[dataset_name][bb_name][str(latent_dim)]['d_dist_watch_std'] = np.std(d_dist_watch)
    d[dataset_name][bb_name][str(latent_dim)]['d_count_watch_mean'] = np.mean(d_count_watch)
    d[dataset_name][bb_name][str(latent_dim)]['d_count_watch_std'] = np.std(d_count_watch)
    d[dataset_name][bb_name][str(latent_dim)]['d_impl_watch_mean'] = np.mean(d_impl_watch)
    d[dataset_name][bb_name][str(latent_dim)]['d_impl_watch_std'] = np.std(d_impl_watch)
    d[dataset_name][bb_name][str(latent_dim)]['d_advs_watch_mean'] = np.mean(d_count_watch)
    d[dataset_name][bb_name][str(latent_dim)]['d_advs_watch_std'] = np.std(d_count_watch)
    d[dataset_name][bb_name][str(latent_dim)]['watch_success_rate'] = len(d_dist_watch)/n
    pickle.dump(d, open(f'./{dataset_name}_{black_box}_results.p','wb'))
