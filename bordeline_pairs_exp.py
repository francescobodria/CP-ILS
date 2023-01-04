from tqdm import tqdm
import numpy as np
import pandas as pd
import pickle
import time
import os
import warnings
warnings.filterwarnings("ignore")


import torch
random_seed = 42
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

# number of samples to test
n = 2

#for dataset_name in ['adult','german','compass']:
dataset_name = 'german'

# LOAD Dataset
from exp.data_loader import load_tabular_data
X_train, X_test, Y_train, Y_test, df = load_tabular_data(dataset_name)

# fit and save Black Boxes

# XGB
from xgboost import XGBClassifier
clf_xgb = XGBClassifier(n_estimators=60, reg_lambda=3, use_label_encoder=False, eval_metric='logloss')
clf_xgb.fit(X_train, Y_train)
clf_xgb.save_model(f'./blackboxes/{dataset_name}_xgboost')
clf_xgb.load_model(f'./blackboxes/{dataset_name}_xgboost')
y_train_pred = clf_xgb.predict(X_train)
y_test_pred = clf_xgb.predict(X_test)
print('XGB')
print('train acc:',np.mean(np.round(y_train_pred)==Y_train))
print('test acc:',np.mean(np.round(y_test_pred)==Y_test))

#SVC
from sklearn.svm import SVC
clf_svc = SVC(gamma='auto', probability=True)
# clf_svc.fit(X_train, Y_train)
# pickle.dump(clf_svc,open(f'./blackboxes/{dataset_name}_svc.p','wb'))
clf_svc = pickle.load(open(f'./blackboxes/{dataset_name}_svc.p','rb'))
y_train_pred = clf_svc.predict(X_train)
y_test_pred = clf_svc.predict(X_test)
print('SVC')
print('train acc:',np.mean(np.round(y_train_pred)==Y_train))
print('test acc:',np.mean(np.round(y_test_pred)==Y_test))

# NN
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
class FFNN(pl.LightningModule):
    def __init__(self, input_shape):
        super(FFNN,self).__init__()
        self.fc1 = nn.Linear(input_shape,10)
        self.fc2 = nn.Linear(10,5)
        self.out = nn.Linear(5,1)
        self.lr = 1e-3
        self.loss = nn.BCEWithLogitsLoss()
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)
    def predict(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.sigmoid(self.out(x))
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.loss(logits,y)
        self.log("train_loss", loss, on_epoch=True)
        return loss
    def validation_step(self, valid_batch, batch_idx):
        x, y = valid_batch
        logits = self.forward(x)
        loss = self.loss(logits,y)
        self.log("test_loss", loss, on_epoch=True)
clf_nn = FFNN(X_train.shape[1])
# -------- training ----------
# class Data(pl.LightningDataModule):
#     def prepare_data(self):      
#         self.train_data = TensorDataset(torch.tensor(X_train.values).float(),torch.tensor(Y_train.values).reshape(-1,1).float())
#         self.test_data = TensorDataset(torch.tensor(X_test.values).float(),torch.tensor(Y_test.values).reshape(-1,1).float())

#     def train_dataloader(self):
#         return DataLoader(self.train_data, batch_size=1024, shuffle=True)

#     def val_dataloader(self):
#         return DataLoader(self.test_data, batch_size=1024, shuffle=False)
# data = Data()
# logger = TensorBoardLogger("NN_logs", name=f"NN_{dataset_name}", version=0)
# es = EarlyStopping(monitor="train_loss", mode="min")
# class LitProgressBar(TQDMProgressBar):
#     def init_validation_tqdm(self):
#         bar = tqdm(disable=True)
#         return bar
# bar = LitProgressBar()
# trainer = pl.Trainer(logger=logger, max_epochs=5000, callbacks=[es,bar], enable_checkpointing=False)
# trainer.fit(clf_nn,data)
# torch.save(clf_nn.state_dict(), f'./blackboxes/{dataset_name}_nn.pt')
# ----------------------
clf_nn.load_state_dict(torch.load(f'./blackboxes/{dataset_name}_nn.pt'))        
clf_nn.trainable = False
from sklearn.metrics import accuracy_score
with torch.no_grad():
    clf_nn.eval()
    print('NN')
    print('train_acc: ',accuracy_score(np.round(clf_nn.predict(torch.tensor(X_train.values).float()).detach().numpy()),Y_train))
    print('test_acc: ',accuracy_score(np.round(clf_nn.predict(torch.tensor(X_test.values).float()).detach().numpy()),Y_test))

# create saving dataset
d = {}

# black box selection
for black_box in ['xgb','svc','nn']:

    print(f'{dataset_name} {black_box} \n')
    
    d[dataset_name]={black_box:{}}

    if black_box=='xgb':
        def predict(x, return_proba=False):
            if return_proba:
                return clf_xgb.predict_proba(x)[:,1].ravel()
            else: return clf_xgb.predict(x).ravel().ravel()
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
            if type(x)==pd.core.frame.DataFrame:
                x = x.values
            clf_nn.eval()
            if return_proba:
                with torch.no_grad():
                    return clf_nn.predict(torch.tensor(x).float()).detach().numpy().ravel()
            else: return np.round(clf_nn.predict(torch.tensor(x).float()).detach().numpy()).ravel()
        y_test_pred = predict(X_test, return_proba=True)
        y_train_pred = predict(X_train, return_proba=True)

    X_train_latent = np.hstack((X_train,y_train_pred.reshape(-1,1)))
    X_test_latent = np.hstack((X_test,y_test_pred.reshape(-1,1)))

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import TensorDataset, DataLoader

    # Latent Space
    if dataset_name == 'adult':
        latent_dim = 5
        idx_cat = [2,3,4,5,6]
        batch_size = 1024
        sigma = 3
        max_epochs = 1000
        early_stopping = 3
        learning_rate = 1e-3
    elif dataset_name == 'german':
        idx_cat = np.arange(3,71,1).tolist()
        latent_dim = 10
        batch_size = 1024
        sigma = 1
        max_epochs = 1000
        early_stopping = 3
        learning_rate = 1e-3
    elif dataset_name == 'compas':
        idx_cat = list(range(13,33,1))
        latent_dim = 15
        batch_size = 1024
        sigma = 1
        max_epochs = 1000
        early_stopping = 3
        learning_rate = 1e-3

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
    # Create Latent Model
    model = LinearModel(X_train_latent.shape[1], latent_dim=latent_dim)

    # --------------- ILS Training ---------------
    # train_dataset = TensorDataset(torch.tensor(X_train_latent).float())
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 
    # test_dataset = TensorDataset(torch.tensor(X_test_latent).float())
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) 
    # def check_and_clear(dir_name):
    #     if not os.path.exists(dir_name):
    #         os.mkdir(dir_name)
    #     else:
    #         os.system('rm -r ' + dir_name)
    #         os.mkdir(dir_name)
    # check_and_clear('./models/weights')
    # model_params = list(model.parameters())
    # optimizer = torch.optim.Adam(model_params, lr=learning_rate)
    # # record training process
    # epoch_train_losses = []
    # epoch_test_losses = []
    # #validation parameters
    # epoch = 1
    # best = np.inf
    # # progress bar
    # pbar = tqdm(bar_format="{postfix[0]} {postfix[1][value]:03d} {postfix[2]} {postfix[3][value]:.5f} {postfix[4]} {postfix[5][value]:.5f} {postfix[6]} {postfix[7][value]:d}",
    #             postfix=["Epoch:", {'value':0}, "Train Sim Loss", {'value':0}, "Test Sim Loss", {'value':0}, "Early Stopping", {"value":0}])
    # # start training
    # while epoch <= max_epochs:
    #     # set model as training mode
    #     model.train()
    #     batch_loss = []
    #     for batch, (X_batch,) in enumerate(train_loader):
    #         optimizer.zero_grad()
    #         Z_batch = model(X_batch)  #
    #         loss  = loss_function(X_batch, Z_batch, idx_cat, sigma) 
    #         loss.backward()
    #         optimizer.step()
    #         batch_loss.append(loss.item())
    #     # save result
    #     epoch_train_losses.append(np.mean(batch_loss))
    #     pbar.postfix[3]["value"] = np.mean(batch_loss)
    #     # set model as testing mode
    #     model.eval()
    #     batch_loss = []
    #     with torch.no_grad():
    #         for batch, (X_batch,) in enumerate(test_loader):
    #             Z_batch = model(X_batch)
    #             loss = loss_function(X_batch, Z_batch, idx_cat, sigma)
    #             batch_loss.append(loss.item())
    #     # save information
    #     epoch_test_losses.append(np.mean(batch_loss))
    #     pbar.postfix[5]["value"] = np.mean(batch_loss)
    #     pbar.postfix[1]["value"] = epoch
    #     # Early Stopping
    #     if epoch_test_losses[-1] < best:
    #         wait = 0
    #         best = epoch_test_losses[-1]
    #         best_epoch = epoch
    #         torch.save(model.state_dict(), f'./models/weights/LinearTransparent_{dataset_name}.pt')
    #     else:
    #         wait += 1
    #     pbar.postfix[7]["value"] = wait
    #     if wait == early_stopping:
    #         break    
    #     epoch += 1
    #     pbar.update()
    # model.load_state_dict(torch.load(f'./models/weights/LinearTransparent_{dataset_name}.pt'))
    # with torch.no_grad():
    #     model.eval()
    #     Z_train = model(torch.tensor(X_train_latent).float()).cpu().detach().numpy()
    #     Z_test = model(torch.tensor(X_test_latent).float()).cpu().detach().numpy()
    # torch.save(model.state_dict(), f'./models/{dataset_name}_latent_{black_box}_{latent_dim}.pt')
    # --------------- ILS Training ---------------

    model.load_state_dict(torch.load(f'./models/{dataset_name}_latent_{black_box}_{latent_dim}.pt'))
    with torch.no_grad():
        model.eval()
        Z_train = model(torch.tensor(X_train_latent).float()).cpu().detach().numpy()
        Z_test = model(torch.tensor(X_test_latent).float()).cpu().detach().numpy()
    
    # Borderline Pairs
    w = model.fc1.weight.detach().numpy()
    b = model.fc1.bias.detach().numpy()
    y_contrib = model.fc1.weight.detach().numpy()[:,-1]
    def compute_bp(q, indexes, step, idx_cat):
        q_pred = predict(q,return_proba=True)
        q_cex = q.copy()
        q_cex_preds = []
        q_cex_preds.append(float(predict(q_cex,return_proba=True)))
        q_cex = np.insert(q_cex,-1,q_pred).reshape(1,-1)
        if q_pred > 0.5:
            m = -step
        else:
            m = +step
        while np.round(q_pred) == np.round(q_cex_preds[-1]):
            # compute the vector to apply
            v = (model(torch.tensor(q_cex).float()).detach().numpy()+m*y_contrib).ravel()
            # compute the changes delta in the input space
            c_l = [v[l] - np.sum(q_cex*w[l,:]) - b[l] for l in range(latent_dim)]
            M = []
            for l in range(latent_dim):
                M.append([np.sum(w[k,indexes]*w[l,indexes]) for k in range(latent_dim)])
            M = np.vstack(M)
            lambda_k = np.linalg.solve(M, c_l)
            delta_i = [np.sum(lambda_k*w[:,i]) for i in indexes]
            q_ex = q_cex[:,:-1]
            prop_cex = q_cex.copy()
            prop_cex[:,indexes] += delta_i
            prop_cex[:,idx_cat] = np.round(prop_cex[:,idx_cat])
            # check changes or null effects in the prediction
            if float(predict(prop_cex[:,:-1],return_proba=True)) in q_cex_preds:
                return q_ex, q_cex[:,:-1]
            q_cex_preds.append(float(predict(prop_cex[:,:-1],return_proba=True)))
            prop_cex[:,-1] = q_cex_preds[-1]        
            if len(q_cex_preds) >= 3:
                if (q_cex_preds[-2]-q_cex_preds[-1])*np.sign(m)>=0:
                    return q_ex, q_cex[:,:-1]
            q_cex = prop_cex.copy()
        return q_ex, q_cex[:,:-1]
    from itertools import combinations
    from scipy.spatial.distance import cdist
    # maximum number of features to change
    k = 6
    # step of searching in the prediction direction
    m = 0.1
    d_dist_ILS = []
    d_count_ILS = [] 
    d_impl_ILS = []
    d_adv_ILS = []
    d_dist_pairs_ILS = []
    d_count_pairs_ILS = []
    d_pred_ex_ILS = []
    d_pred_pairs_ILS = []

    print(f'ILS {dataset_name} {black_box} \n')

    for idx in tqdm(range(n)): 
        q = X_test.iloc[idx:idx+1,:].copy()
        q_pred = predict(q,return_proba=False)
        q_cexs = []
        q_exs = []
        l_i = []
        l_f = []
        for indexes in list(combinations(list(range(X_train.shape[1])),1)):    
            q_ex, q_cex = compute_bp(q.values, list(indexes), m, idx_cat)
            q_cex_pred = predict(q_cex,return_proba=True)
            # counterfactual prediction check
            if q_pred:
                if q_cex_pred<0.5:
                    q_cexs.append(q_cex)
                    q_exs.append(q_ex)
            else:
                if q_cex_pred>0.5:
                    q_cexs.append(q_cex)
                    q_exs.append(q_ex)
        for indexes in list(combinations(list(range(X_train.shape[1])),2)):    
            q_ex, q_cex = compute_bp(q.values, list(indexes), m, idx_cat)
            q_cex_pred = predict(q_cex,return_proba=True)
            if q_pred:
                if q_cex_pred<0.5:
                    q_cexs.append(q_cex)
                    q_exs.append(q_ex)            
            else:
                if q_cex_pred>0.5:
                    q_cexs.append(q_cex)
                    q_exs.append(q_ex)
            l_i.append([list(indexes),q_cex_pred])
        # cerco solo nelle direzioni più promettenti
        r = np.argsort(np.stack(np.array(l_i,dtype=object)[:,1]).ravel())[-10:]
        l_i = np.array(l_i,dtype=object)[r,0]
        while len(l_i[0])<=k:
            for e in l_i:
                for i in list(np.delete(range(X_train.shape[1]),e)):
                    q_ex, q_cex = compute_bp(q.values, e+[i], m, idx_cat)
                    q_cex_pred = predict(q_cex,return_proba=True)
                    if q_pred:
                        if q_cex_pred<0.5:
                            q_cexs.append(q_cex)
                            q_exs.append(q_ex)            
                    else:
                        if q_cex_pred>0.5:
                            q_cexs.append(q_cex)
                            q_exs.append(q_ex) 
                    l_f.append([e+[i],q_cex_pred])
            r = np.argsort(np.stack(np.array(l_f,dtype=object)[:,1]).ravel())[-10:]
            l_f = np.array(l_f,dtype=object)[r,0]
            l_i = l_f.copy()
            l_f = []
        if len(q_cexs)>0:
            q_cexs = pd.DataFrame(np.vstack(q_cexs),columns=X_train.columns)
            q_exs = pd.DataFrame(np.vstack(q_exs),columns=X_train.columns)

            cont_idx = np.delete(range(len(X_train.columns)),idx_cat)
            q_cexs.iloc[:,cont_idx] = np.clip(q_cexs.values[:,cont_idx],-1,1)
            q_exs.iloc[:,cont_idx] = np.clip(q_exs.values[:,cont_idx],-1,1)
            q_cexs.iloc[:,idx_cat] = np.clip(q_cexs.values[:,idx_cat],0,100)
            q_exs.iloc[:,idx_cat] = np.clip(q_exs.values[:,idx_cat],0,100)

            d_dist_pairs = cdist(q_exs.iloc[:,cont_idx].values,q_cexs.iloc[:,cont_idx].values,metric='euclidean')+cdist(q_exs.iloc[:,idx_cat].values,q_cexs.iloc[:,idx_cat].values,metric='hamming')
            d_count_pairs = cdist(q_exs.values,q_cexs.values,metric='hamming')
            d_pred_pairs = cdist(predict(q_exs,return_proba=True).reshape(-1,1),predict(q_cexs,return_proba=True).reshape(-1,1))
            indices = np.stack(np.where((d_dist_pairs+d_count_pairs+d_pred_pairs)==np.min(d_dist_pairs+d_count_pairs+d_pred_pairs)))[:,0]
            q_ex = q_exs.iloc[indices[0]:indices[0]+1,:]
            q_cex = q_cexs.iloc[indices[1]:indices[1]+1,:]

            d_dist_ILS.append(float(cdist(q_cex.iloc[:,cont_idx].values,q.iloc[:,cont_idx].values,metric='euclidean')+cdist(q_cex.iloc[:,idx_cat].values,q.iloc[:,idx_cat].values,metric='hamming')))
            d_count_ILS.append(float(np.sum(q_cex.values!=q.values,axis=1)/len(X_train.columns)))
            d_impl_ILS.append(float(np.min(cdist(q_cex,X_train),axis=1)))
            r = np.argsort(cdist(q_cex.iloc[:,cont_idx].values,X_train.iloc[:,cont_idx].values,metric='euclidean')+cdist(q_cex.iloc[:,idx_cat].values,X_train.iloc[:,idx_cat].values,metric='hamming'))[:10].ravel()
            d_adv_ILS.append(np.mean(predict(X_train.iloc[r])==predict(q)))
            d_dist_pairs_ILS.append(float(cdist(q_ex.iloc[:,cont_idx].values,q_cex.iloc[:,cont_idx].values,metric='euclidean')+cdist(q_ex.iloc[:,idx_cat].values,q_cex.iloc[:,idx_cat].values,metric='hamming')))
            d_count_pairs_ILS.append(float(np.sum(q_cex.values!=q_ex.values,axis=1)/len(X_train.columns)))
            d_pred_ex_ILS.append(float(np.abs(0.5-predict(q_ex,return_proba=True))))
            d_pred_pairs_ILS.append(float(np.abs(predict(q_ex,return_proba=True)-predict(q_cex,return_proba=True))))

    d[dataset_name][black_box]['d_dist_ILS_mean'] = np.mean(np.array(d_dist_ILS))
    d[dataset_name][black_box]['d_dist_ILS_std'] = np.std(np.array(d_dist_ILS))
    d[dataset_name][black_box]['d_count_ILS_mean'] = np.mean(np.array(d_count_ILS))
    d[dataset_name][black_box]['d_count_ILS_std'] = np.std(np.array(d_count_ILS))
    d[dataset_name][black_box]['d_impl_ILS_mean'] = np.mean(np.array(d_impl_ILS)) 
    d[dataset_name][black_box]['d_impl_ILS_std'] = np.std(np.array(d_impl_ILS))
    d[dataset_name][black_box]['d_adv_ILS_mean'] = np.mean(np.array(d_adv_ILS)) 
    d[dataset_name][black_box]['d_adv_ILS_std'] = np.std(np.array(d_adv_ILS))
    d[dataset_name][black_box]['d_dist_pairs_ILS_mean'] = np.mean(np.array(d_dist_pairs_ILS)) 
    d[dataset_name][black_box]['d_dist_pairs_ILS_std'] = np.std(np.array(d_dist_pairs_ILS))
    d[dataset_name][black_box]['d_count_pairs_ILS_mean'] = np.mean(np.array(d_count_pairs_ILS)) 
    d[dataset_name][black_box]['d_count_pairs_ILS_std'] = np.std(np.array(d_count_pairs_ILS))
    d[dataset_name][black_box]['d_pred_ex_ILS_mean'] = np.mean(np.array(d_pred_ex_ILS)) 
    d[dataset_name][black_box]['d_pred_ex_ILS_std'] = np.std(np.array(d_pred_ex_ILS))
    d[dataset_name][black_box]['d_pred_pairs_ILS_mean'] = np.mean(np.array(d_pred_pairs_ILS)) 
    d[dataset_name][black_box]['d_pred_pairs_ILS_std'] = np.std(np.array(d_pred_pairs_ILS))
    pickle.dump(d, open(f'./results/{dataset_name}_{black_box}_results.p','wb'))

    # GSG
    from growingspheres import counterfactuals as cf
    d_dist_GS = []
    d_count_GS = [] 
    d_impl_GS = []
    d_adv_GS = []
    d_dist_pairs_GS = []
    d_count_pairs_GS = []
    d_pred_ex_GS = []
    d_pred_pairs_GS = []

    print(f'GS {dataset_name} {black_box} \n')

    for idx in tqdm(range(n)):
        q = X_test.iloc[idx:idx+1,:].copy()
        pred = int(predict(q))
        CF = cf.CounterfactualExplanation(q.values, predict, method='GS')
        CF.fit(n_in_layer=2000, first_radius=0.1, dicrease_radius=10, sparse=True, verbose=False)
        q_cex_GS = pd.DataFrame(CF.enemy.reshape(1,-1),columns=X_train.columns)
        pred = 1-pred
        CF = cf.CounterfactualExplanation(q_cex_GS.values, predict, method='GS')
        CF.fit(n_in_layer=2000, first_radius=0.1, dicrease_radius=10, sparse=True, verbose=False)
        q_ex_GS = pd.DataFrame(CF.enemy.reshape(1,-1),columns=X_train.columns)

        cont_idx = np.delete(range(len(X_train.columns)),idx_cat)
        d_dist_GS.append(float(cdist(q_cex_GS.iloc[:,cont_idx].values,q.iloc[:,cont_idx].values,metric='euclidean')+cdist(q_cex_GS.iloc[:,idx_cat].values,q.iloc[:,idx_cat].values,metric='hamming')))
        d_count_GS.append(float(np.sum(q_cex_GS.values!=q.values,axis=1)/len(X_train.columns)))
        d_impl_GS.append(float(np.min(cdist(q_cex_GS,X_train),axis=1)))
        r = np.argsort(cdist(q_cex_GS.iloc[:,cont_idx].values,X_train.iloc[:,cont_idx].values,metric='euclidean')+cdist(q_cex_GS.iloc[:,idx_cat].values,X_train.iloc[:,idx_cat].values,metric='hamming'))[:10].ravel()
        d_adv_GS.append(np.mean(predict(X_train.iloc[r])==predict(q)))
        d_dist_pairs_GS.append(float(cdist(q_ex_GS.iloc[:,cont_idx].values,q_cex_GS.iloc[:,cont_idx].values,metric='euclidean')+cdist(q_ex_GS.iloc[:,idx_cat].values,q_cex_GS.iloc[:,idx_cat].values,metric='hamming')))
        d_count_pairs_GS.append(float(np.sum(q_cex_GS.values!=q_ex_GS.values,axis=1)/len(X_train.columns)))
        d_pred_ex_GS.append(float(np.abs(0.5-predict(q_ex_GS,return_proba=True))))
        d_pred_pairs_GS.append(float(np.abs(predict(q_ex_GS,return_proba=True)-predict(q_cex_GS,return_proba=True))))

    d[dataset_name][black_box]['d_dist_GS_mean'] = np.mean(np.array(d_dist_GS))
    d[dataset_name][black_box]['d_dist_GS_std'] = np.std(np.array(d_dist_GS))
    d[dataset_name][black_box]['d_count_GS_mean'] = np.mean(np.array(d_count_GS))
    d[dataset_name][black_box]['d_count_GS_std'] = np.std(np.array(d_count_GS))
    d[dataset_name][black_box]['d_impl_GS_mean'] = np.mean(np.array(d_impl_GS)) 
    d[dataset_name][black_box]['d_impl_GS_std'] = np.std(np.array(d_impl_GS))
    d[dataset_name][black_box]['d_adv_GS_mean'] = np.mean(np.array(d_adv_GS)) 
    d[dataset_name][black_box]['d_adv_GS_std'] = np.std(np.array(d_adv_GS))
    d[dataset_name][black_box]['d_dist_pairs_GS_mean'] = np.mean(np.array(d_dist_pairs_GS)) 
    d[dataset_name][black_box]['d_dist_pairs_GS_std'] = np.std(np.array(d_dist_pairs_GS))
    d[dataset_name][black_box]['d_count_pairs_GS_mean'] = np.mean(np.array(d_count_pairs_GS)) 
    d[dataset_name][black_box]['d_count_pairs_GS_std'] = np.std(np.array(d_count_pairs_GS))
    d[dataset_name][black_box]['d_pred_ex_GS_mean'] = np.mean(np.array(d_pred_ex_GS)) 
    d[dataset_name][black_box]['d_pred_ex_GS_std'] = np.std(np.array(d_pred_ex_GS))
    d[dataset_name][black_box]['d_pred_pairs_GS_mean'] = np.mean(np.array(d_pred_pairs_GS)) 
    d[dataset_name][black_box]['d_pred_pairs_GS_std'] = np.std(np.array(d_pred_pairs_GS))
    pickle.dump(d, open(f'./results/{dataset_name}_{black_box}_results.p','wb'))

    # FAT-F
    import fatf.transparency.predictions.counterfactuals as fatf_cf
    d_dist_FAT = []
    d_count_FAT = [] 
    d_impl_FAT = []
    d_adv_FAT = []
    d_dist_pairs_FAT = []
    d_count_pairs_FAT = []
    d_pred_ex_FAT = []
    d_pred_pairs_FAT = []

    if black_box == 'xgb':
        cf_explainer = fatf_cf.CounterfactualExplainer(model=clf_xgb, dataset=X_train.values, categorical_indices=idx_cat, default_numerical_step_size=0.1)
    elif black_box == 'svc':
        cf_explainer = fatf_cf.CounterfactualExplainer(model=clf_svc, dataset=X_train.values, categorical_indices=idx_cat, default_numerical_step_size=0.1)
    elif black_box == 'nn':
        cf_explainer = fatf_cf.CounterfactualExplainer(predictive_function=predict, dataset=X_train.values, categorical_indices=idx_cat, default_numerical_step_size=0.1)

    print(f'FAT {dataset_name} {black_box} \n')

    for idx in tqdm(range(n)):
        q = X_test.iloc[idx:idx+1,:].values.copy()
        q = pd.DataFrame(q,columns=X_train.columns)
        q_cex_FAT, _, _ = cf_explainer.explain_instance(q.values.ravel())
        if len(q_cex_FAT)>0:
            q_ex_FAT, _, _ = cf_explainer.explain_instance(q_cex_FAT[0:1,:].ravel())
            if len(q_ex_FAT)>0:
                q_cex_FAT = pd.DataFrame(q_cex_FAT[0:1,:],columns=X_train.columns)
                q_ex_FAT = pd.DataFrame(q_ex_FAT[0:1,:],columns=X_train.columns)

                cont_idx = np.delete(range(len(X_train.columns)),idx_cat)
                d_dist_FAT.append(float(cdist(q_cex_FAT.iloc[:,cont_idx].values,q.iloc[:,cont_idx].values,metric='euclidean')+cdist(q_cex_FAT.iloc[:,idx_cat].values,q.iloc[:,idx_cat].values,metric='hamming')))
                d_count_FAT.append(float(np.sum(q_cex_FAT.values!=q.values,axis=1)/len(X_train.columns)))
                d_impl_FAT.append(float(np.min(cdist(q_cex_FAT,X_train),axis=1)))
                r = np.argsort(cdist(q_cex_FAT.iloc[:,cont_idx].values,X_train.iloc[:,cont_idx].values,metric='euclidean')+cdist(q_cex_FAT.iloc[:,idx_cat].values,X_train.iloc[:,idx_cat].values,metric='hamming'))[:10].ravel()
                d_adv_FAT.append(np.mean(predict(X_train.iloc[r])==predict(q)))
                d_dist_pairs_FAT.append(float(cdist(q_ex_FAT.iloc[:,cont_idx].values,q_cex_FAT.iloc[:,cont_idx].values,metric='euclidean')+cdist(q_ex_FAT.iloc[:,idx_cat].values,q_cex_FAT.iloc[:,idx_cat].values,metric='hamming')))
                d_count_pairs_FAT.append(float(np.sum(q_cex_FAT.values!=q_ex_FAT.values,axis=1)/len(X_train.columns)))
                d_pred_ex_FAT.append(float(np.abs(0.5-predict(q_ex_FAT,return_proba=True))))
                d_pred_pairs_FAT.append(float(np.abs(predict(q_ex_FAT,return_proba=True)-predict(q_cex_FAT,return_proba=True))))

    d[dataset_name][black_box]['d_dist_FAT_mean'] = np.mean(np.array(d_dist_FAT))
    d[dataset_name][black_box]['d_dist_FAT_std'] = np.std(np.array(d_dist_FAT))
    d[dataset_name][black_box]['d_count_FAT_mean'] = np.mean(np.array(d_count_FAT))
    d[dataset_name][black_box]['d_count_FAT_std'] = np.std(np.array(d_count_FAT))
    d[dataset_name][black_box]['d_impl_FAT_mean'] = np.mean(np.array(d_impl_FAT)) 
    d[dataset_name][black_box]['d_impl_FAT_std'] = np.std(np.array(d_impl_FAT))
    d[dataset_name][black_box]['d_adv_FAT_mean'] = np.mean(np.array(d_adv_FAT)) 
    d[dataset_name][black_box]['d_adv_FAT_std'] = np.std(np.array(d_adv_FAT))
    d[dataset_name][black_box]['d_dist_pairs_FAT_mean'] = np.mean(np.array(d_dist_pairs_FAT)) 
    d[dataset_name][black_box]['d_dist_pairs_FAT_std'] = np.std(np.array(d_dist_pairs_FAT))
    d[dataset_name][black_box]['d_count_pairs_FAT_mean'] = np.mean(np.array(d_count_pairs_FAT)) 
    d[dataset_name][black_box]['d_count_pairs_FAT_std'] = np.std(np.array(d_count_pairs_FAT))
    d[dataset_name][black_box]['d_pred_ex_FAT_mean'] = np.mean(np.array(d_pred_ex_FAT)) 
    d[dataset_name][black_box]['d_pred_ex_FAT_std'] = np.std(np.array(d_pred_ex_FAT))
    d[dataset_name][black_box]['d_pred_pairs_FAT_mean'] = np.mean(np.array(d_pred_pairs_FAT)) 
    d[dataset_name][black_box]['d_pred_pairs_FAT_std'] = np.std(np.array(d_pred_pairs_FAT))
    pickle.dump(d, open(f'./results/{dataset_name}_{black_box}_results.p','wb'))

    # WACH
    from scipy.spatial.distance import cdist, euclidean
    from scipy.optimize import minimize
    from scipy import stats

    d_dist_WACH = []
    d_count_WACH = [] 
    d_impl_WACH = []
    d_adv_WACH = []
    d_dist_pairs_WACH = []
    d_count_pairs_WACH = []
    d_pred_ex_WACH = []
    d_pred_pairs_WACH = []

    print(f'WACH {dataset_name} {black_box} \n')

    for idx in tqdm(range(n)):
        # initial conditions
        lamda = 0.1 
        x0 = np.zeros([1,X_train.values.shape[1]]) # initial guess for cf
        q = X_test.iloc[idx:idx+1,:].copy()
        pred = int(predict(q,return_proba=False))
        def dist_mad(cf, eg):
            manhat = [cdist(eg.T, cf.reshape(1,-1).T, metric='cityblock')[i][i] for i in range(len(eg.T))]
            return sum(manhat)
        def loss_function_mad(x_dash):
            target = 1-pred
            x_dash = pd.DataFrame(x_dash.reshape(1,-1),columns=X_train.columns)
            L = lamda*(predict(x_dash,return_proba=True)-target)**2 + dist_mad(x_dash.values, q.values)
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
        q_cex_W = pd.DataFrame(cf,columns=X_train.columns)
        if r == 1:
            # initial conditions
            lamda = 0.1 
            x0 = np.zeros([1,X_train.values.shape[1]]) # initial guess for cf
            q = q_cex_W.copy()
            pred = int(predict(q,return_proba=False))
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

        if r==1:
            q_ex_W = pd.DataFrame(cf,columns=X_train.columns)
            q = X_test.iloc[idx:idx+1,:].copy()

            cont_idx = np.delete(range(len(X_train.columns)),idx_cat)
            d_dist_WACH.append(float(cdist(q_cex_W.iloc[:,cont_idx].values,q.iloc[:,cont_idx].values,metric='euclidean')+cdist(q_cex_W.iloc[:,idx_cat].values,q.iloc[:,idx_cat].values,metric='hamming')))
            d_count_WACH.append(float(np.sum(q_cex_W.values!=q.values,axis=1)/len(X_train.columns)))
            d_impl_WACH.append(float(np.min(cdist(q_cex_W,X_train),axis=1)))
            r = np.argsort(cdist(q_cex_W.iloc[:,cont_idx].values,X_train.iloc[:,cont_idx].values,metric='euclidean')+cdist(q_cex_W.iloc[:,idx_cat].values,X_train.iloc[:,idx_cat].values,metric='hamming'))[:10].ravel()
            d_adv_WACH.append(np.mean(predict(X_train.iloc[r])==predict(q)))
            d_dist_pairs_WACH.append(float(cdist(q_ex_W.iloc[:,cont_idx].values,q_cex_W.iloc[:,cont_idx].values,metric='euclidean')+cdist(q_ex_W.iloc[:,idx_cat].values,q_cex_W.iloc[:,idx_cat].values,metric='hamming')))
            d_count_pairs_WACH.append(float(np.sum(q_cex_W.values!=q_ex_W.values,axis=1)/len(X_train.columns)))
            d_pred_ex_WACH.append(float(np.abs(0.5-predict(q_ex_W,return_proba=True))))
            d_pred_pairs_WACH.append(float(np.abs(predict(q_ex_W,return_proba=True)-predict(q_cex_W,return_proba=True))))
    
    d[dataset_name][black_box]['d_dist_WACH_mean'] = np.mean(np.array(d_dist_WACH))
    d[dataset_name][black_box]['d_dist_WACH_std'] = np.std(np.array(d_dist_WACH))
    d[dataset_name][black_box]['d_count_WACH_mean'] = np.mean(np.array(d_count_WACH))
    d[dataset_name][black_box]['d_count_WACH_std'] = np.std(np.array(d_count_WACH))
    d[dataset_name][black_box]['d_impl_WACH_mean'] = np.mean(np.array(d_impl_WACH)) 
    d[dataset_name][black_box]['d_impl_WACH_std'] = np.std(np.array(d_impl_WACH))
    d[dataset_name][black_box]['d_adv_WACH_mean'] = np.mean(np.array(d_adv_WACH)) 
    d[dataset_name][black_box]['d_adv_WACH_std'] = np.std(np.array(d_adv_WACH))
    d[dataset_name][black_box]['d_dist_pairs_WACH_mean'] = np.mean(np.array(d_dist_pairs_WACH)) 
    d[dataset_name][black_box]['d_dist_pairs_WACH_std'] = np.std(np.array(d_dist_pairs_WACH))
    d[dataset_name][black_box]['d_count_pairs_WACH_mean'] = np.mean(np.array(d_count_pairs_WACH)) 
    d[dataset_name][black_box]['d_count_pairs_WACH_std'] = np.std(np.array(d_count_pairs_WACH))
    d[dataset_name][black_box]['d_pred_ex_WACH_mean'] = np.mean(np.array(d_pred_ex_WACH)) 
    d[dataset_name][black_box]['d_pred_ex_WACH_std'] = np.std(np.array(d_pred_ex_WACH))
    d[dataset_name][black_box]['d_pred_pairs_WACH_mean'] = np.mean(np.array(d_pred_pairs_WACH)) 
    d[dataset_name][black_box]['d_pred_pairs_WACH_std'] = np.std(np.array(d_pred_pairs_WACH))
    pickle.dump(d, open(f'./results/{dataset_name}_{black_box}_results.p','wb'))


