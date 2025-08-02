from enum import EnumMeta
import sys
import os
from typing import List

sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
from sklearn import metrics
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score, f1_score, \
                            balanced_accuracy_score, matthews_corrcoef, precision_score, recall_score, matthews_corrcoef, precision_score, recall_score
import numpy as np
from torchmetrics.functional import calibration_error, mean_squared_error
import torch.nn.functional as F
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader as SmilesDataLoader
from dataset import ECFPDataset, OODDataset
from rdkit import Chem

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Mtask_model(nn.Module):
    def __init__(self,args, s_train, model_name='CNN'):
        super(Mtask_model, self).__init__()
    
        self.dataset = args.dataset
        self.emb_dim = args.emb_dim
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.decay = args.decay
        self.layer_size = args.layer_size
        self.split_idx = args.split_idx


        # self.criterion = nn.CrossEntropyLoss()
        self.task_no = args.task_no
        
        self.penalty_coefficients = args.penalty_coefficients

        
        # self.mtask_model = MT_DNN(args.emb_dim, args.task_no, args.layer_size, args.num_tasks)
        # self.mtask_model = MT_DNN(args.emb_dim, args.task_no, args.layer_size, args.num_tasks,
        #                           regularization_weight=REG_WEIGHT,
        #                           parameterization=PARAM,
        #                           prior_scale=PRIOR_SCALE,
        #                           return_ood=RETURN_OOD,
        #                           NN_heads=NN_heads)
        
        # Dynamically import MT_DNN from the appropriate module based on model_name
        import importlib
        mtask_module = importlib.import_module(f"{model_name}.model")
        MT_DNN = getattr(mtask_module, "MT_model")
        self.mtask_model = MT_DNN(args)  # You should instantiate MT_DNN here with the required arguments
        
        self.model_name = model_name
        
        # model_param_group = []
        # model_param_group.append({"params": self.mtask_model.parameters()})

        self.opt = optim.Adam(self.mtask_model.parameters(), lr = 0.01, weight_decay=self.decay)
        # self.criterion = nn.CrossEntropyLoss()
        self.device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
        self.softmax = nn.Softmax(dim=0)
    
    def forward(self, input, return_params=False):
        '''
        return task_num predictions
        '''
        return self.mtask_model(input, return_params=return_params)
    

    def predict(self, init_feat):
        self.softmax = nn.Softmax(dim=1)
        tasks_outs = self.forward(init_feat)
        
        tasks_preds = [None] * len(tasks_outs)
        for i, batch_out in enumerate(tasks_outs):
            pred = self.softmax(batch_out)
            
            tasks_preds[i] = pred[:, 1]

        tasks_preds = torch.stack(tasks_preds)  # shape: (batch_size, num_tasks)
        return tasks_preds




    def train(self, multi_task_train_data):
        
        self.mtask_model.train()
        
        if self.model_name == 'PreGNN':
            for i, batch in enumerate(tqdm.tqdm(multi_task_train_data, desc="{}-{}".format(self.layer_size, self.split_idx))):
                y = batch['y'].to(self.device).to(torch.long).reshape(max(batch.batch).item() + 1, self.mtask_model.num_tasks)
                x = batch['x'].to(self.device)
                edge_index = batch['edge_index'].to(self.device)
                edge_attr = batch['edge_attr'].to(self.device)
                batch_idx = batch['batch'].to(self.device)
                weights = batch['weights'].to(self.device).reshape(max(batch.batch).item() + 1, self.mtask_model.num_tasks)
            
                # Get the prediction of the each task
                # Since nn.CrossEntropyLoss include softmax function, there is not softmax in preds computation here.
                outs = self.forward([x, edge_attr, edge_index, batch_idx])
                
                criterion = nn.CrossEntropyLoss(reduction='none')
           
                # now there are n calss losses
                # loss -> [b, n]
                losses = []
                for idx, pred in enumerate(outs):
                    loss = criterion(pred.double(), y[idx])
                    # set the loss to 0 where the task has no datapoint
                    # just loss x weights will achieve this point
                    loss = loss.mul(weights[idx])
                    losses.append(loss)

           
                losses = torch.stack(losses).to(self.device)
                penalty = torch.DoubleTensor(self.penalty_coefficients).to(self.device).repeat(y.size()[0], 1).mul(y.to(torch.float))
                one = torch.ones_like(penalty)
                penalty = torch.where(penalty == 0, one, penalty)            

                # add(mul) penalty to losses
                losses = losses.mul(penalty)

                self.opt.zero_grad()
                losses.backward(losses.clone().detach())

           
                self.opt.step()
                
        else:
            for i, (batch, weights) in enumerate(tqdm.tqdm(multi_task_train_data, desc="{}-{}".format(self.layer_size, self.split_idx))):
            
                weights = weights.to(self.device)
                init_features = batch["vec"].to(self.device)
            
                
                if self.model_name == 'VMTL':
                    output_mean, z_mu, z_var, w_mu, w_var = self.forward(init_features, return_params=True)
                    label_list = batch['labels'].to(self.device).to(torch.long)
                    losses = self.mtask_model.compute_loss(output_mean, label_list, z_mu, z_var, w_mu, w_var)
                else:
                    # Get the prediction of the each task
                    # Since nn.CrossEntropyLoss include softmax function, there is not softmax in preds computation here.
                    outs = self.forward(init_features)
            

                    y = batch['labels'].to(self.device).to(torch.long)
                    criterion = nn.CrossEntropyLoss(reduction='none')
           
                    # now there are n calss losses
                    # loss -> [b, n]
                    losses = []
                    for idx, pred in enumerate(outs):
                        loss = criterion(pred.double(), y[idx])
                        # set the loss to 0 where the task has no datapoint
                        # just loss x weights will achieve this point
                        loss = loss.mul(weights[idx])
                        losses.append(loss)
                    losses = torch.stack(losses).to(self.device)
                    penalty = torch.DoubleTensor(self.penalty_coefficients).to(self.device).repeat(y.size()[0], 1).mul(y.to(torch.float))
                    one = torch.ones_like(penalty)
                    penalty = torch.where(penalty == 0, one, penalty)            

                    # add(mul) penalty to losses
                    losses = losses.mul(penalty)

                self.opt.zero_grad()
                losses.backward(losses.clone().detach())

           
                self.opt.step()
    
    def test(self, test_data_list, test_data_taskID):
        '''
        return auc, acc, aupr, f1 list
        '''
        aucs = []
        accs = []
        auprs = []
        f1s = []
        bas = []
        mccs = []
        precisions = []
        recalls = []
        self.mtask_model.eval()
        
        for taskID_idx, test_data in enumerate(test_data_list):
            y_true = []
            y_scores = []
            task_id = test_data_taskID[taskID_idx]
            print('task_id: ', task_id)
            for i, batch in enumerate(tqdm(test_data, desc="Iteration")):
                init_features = batch["vec"].to(self.device)
                # Get the prediction of the corresponding task
                outs = self.predict(init_features)[taskID_idx]
                # preds = []
                # for out in outs:
                #     preds.append(out.predictive.probs)
                preds = torch.stack(preds).transpose(0,1).to(self.device)
                
                y_scores.append(preds)
                y_true.append(batch['label'].to(self.device).view(preds.shape))

            y_true = torch.cat(y_true, dim = 0).cpu().detach().numpy()
            y_scores = torch.cat(y_scores, dim = 0).cpu().detach().numpy()
            y_preds = y_scores.copy()
            y_preds[y_preds < 0.5] = 0
            y_preds[y_preds != 0] = 1 

            
            
            auc = roc_auc_score(y_true, y_scores)
            acc = accuracy_score(y_true, y_preds)
            aupr = average_precision_score(y_true, y_scores)
            f1 = f1_score(y_true, y_preds)
            ba = balanced_accuracy_score(y_true, y_preds)
            mcc = matthews_corrcoef(y_true, y_preds)
            precisions.append(precision_score(y_true, y_preds))
            recalls.append(recall_score(y_true, y_preds))
            
            aucs.append(auc)
            accs.append(acc)
            auprs.append(aupr)
            f1s.append(f1)
            bas.append(ba)
            mccs.append(mcc)
            
        return aucs, accs, auprs, f1s, bas, mccs, precisions, recalls

    # def test_for_comp(self, test_data_list, test_data_taskID):
    #     '''
    #     return a average auc
    #     '''
    #     self.mtask_model.eval()
    #     y_true = []
    #     y_scores = []
    #     for taskID_idx, test_data in enumerate(test_data_list):
            
    #         task_id = test_data_taskID[taskID_idx]
    #         print('task_id: ', task_id)
    #         for i, batch in enumerate(tqdm(test_data, desc="Iteration")):
    #             init_features = batch["vec"].to(self.device)
    #             # Get the prediction of the corresponding task
    #             preds = self.predict(init_features)[taskID_idx]
                
    #             y_scores.append(preds)
    #             y_true.append(batch['label'].to(self.device).view(preds.shape))

    #     y_true = torch.cat(y_true, dim = 0).cpu().detach().numpy()
    #     y_scores = torch.cat(y_scores, dim = 0).cpu().detach().numpy()
    #     y_predict = np.squeeze(y_scores)
    #     for idx, val in enumerate(y_predict):
    #         if val < 0.5 :
    #             y_predict[idx] = 0
    #         else:
    #             y_predict[idx] = 1
                

        
    #     auc = roc_auc_score(y_true, y_scores)
    #     acc = accuracy_score(np.squeeze(y_true), y_predict)
    #     aupr = average_precision_score(np.squeeze(y_true), y_scores)
    #     f1 = f1_score(np.squeeze(y_true), y_predict)
    #     ba = balanced_accuracy_score(y_true, y_predict)
        
    #     # Calculate ECE and NLL
    #     ece = calibration_error(torch.tensor(y_scores), torch.tensor(y_true), n_bins=15, norm='l1', task='binary').item()
    #     # nll = mean_squared_error(torch.tensor(y_scores), torch.tensor(y_true)).item()
    #     nll = F.binary_cross_entropy_with_logits(torch.tensor(y_scores), torch.tensor(y_true, dtype=torch.float32)).item()

        
    #     return acc, auc, aupr, f1, ba, ece, nll

    def test_for_comp(self, test_data_list, test_data_taskID):
        '''
        return a average auc
        '''
        
        
        self.mtask_model.eval()
        y_true = []
        y_scores = []
        
        for taskID_idx, test_data in enumerate(test_data_list):
            
            task_id = test_data_taskID[taskID_idx]
            print('task_id: ', task_id)
            
            for i, batch in enumerate(tqdm.tqdm(test_data, desc="Iteration")):
                # node_features = batch['x'].to(self.device)
                # edge_index = batch['edge_index'].to(self.device)
                # edge_attr = batch['edge_attr'].to(self.device)
                # batch_idx = batch['batch'].to(self.device)
                # Get the prediction of the corresponding task
                
                if self.model_name == 'PreGNN':
                    x, edge_index, edge_attr, batch_idx = batch['x'].to(self.device), batch['edge_index'].to(self.device), batch['edge_attr'].to(self.device), batch['batch'].to(self.device)
                    y = batch['y'].to(self.device)
                    
                    preds = self.predict([x, edge_attr, edge_index, batch_idx])[:, taskID_idx]
                else:
                    smiles, y = batch['vec'], batch['label']
                    smiles = smiles.to(self.device)
                    y = y.to(self.device)
                
                    preds = self.predict(smiles)[:, taskID_idx]
                
                # y_vars.append(vars)
                y_scores.append(preds)
                # y_true.append(batch['y'].to(self.device).view(preds.shape))
                y_true.append(y.view(preds.shape))
            
        y_true = torch.cat(y_true, dim = 0).cpu().detach().numpy()
        y_scores = torch.cat(y_scores, dim = 0).cpu().detach().numpy()
        y_predict = np.squeeze(y_scores)
        # y_vars = torch.cat(y_vars, dim = 0).cpu().detach().numpy()
        for idx, val in enumerate(y_predict):
            if val < 0.5 :
                y_predict[idx] = 0
            else:
                y_predict[idx] = 1

        
        auc = roc_auc_score(y_true, y_scores)
        acc = accuracy_score(np.squeeze(y_true), y_predict)
        aupr = average_precision_score(np.squeeze(y_true), y_scores)
        f1 = f1_score(np.squeeze(y_true), y_predict)
        ba = balanced_accuracy_score(y_true, y_predict)
        
        # Calculate ECE and NLL
        ece = calibration_error(torch.tensor(y_scores), torch.tensor(y_true), n_bins=15, norm='l1', task='binary').item()
        # ece = ece_with_uncertainty(torch.tensor(y_vars), torch.tensor(y_scores), torch.tensor(y_true), n_bins=15, norm='l1')
        # nll = mean_squared_error(torch.tensor(y_scores), torch.tensor(y_true)).item()
        nll = F.binary_cross_entropy_with_logits(torch.tensor(y_scores), torch.tensor(y_true, dtype=torch.float32)).item()
        
        mcc = matthews_corrcoef(y_true, y_predict)
        precision = precision_score(y_true, y_predict)
        recall = recall_score(y_true, y_predict)
        
        return auc, acc, aupr, f1, ba, ece, nll, mcc, precision, recall
    
    def test_for_comp_by_families(self, test_data_list, test_data_taskID):
        
        self.mtask_model.eval()
        
        results = {
            'task_id': [],
            'auc': [],
            'acc': [],
            'aupr': [],
            'f1': [],
            'ba': [],
            'ece': [],
            'nll': [],
            'mcc': [],
            'precision': [],
            'recall': []
        }
        
        for taskID_idx, test_data in enumerate(test_data_list):
            
            task_id = test_data_taskID[taskID_idx]
            print('task_id: ', task_id)
            
            y_true = []
            y_scores = []
            
            # for i, batch in enumerate(tqdm.tqdm(test_data, desc="Iteration")):
            #     # node_features = batch['x'].to(self.device)
            #     # edge_index = batch['edge_index'].to(self.device)
            #     # edge_attr = batch['edge_attr'].to(self.device)
            #     # batch_idx = batch['batch'].to(self.device)
            #     # Get the prediction of the corresponding task
            #     smiles, y = batch['vec'], batch['label']
            #     smiles = smiles.to(self.device)
            #     y = y.to(self.device)
                
            #     # preds = self.predict(node_features, edge_attr, edge_index, batch_idx)[taskID_idx]
            #     preds = self.predict(smiles)[:, taskID_idx]
                
            #     y_scores.append(preds)
            #     # y_true.append(batch['y'].to(self.device).view(preds.shape))
            #     y_true.append(y.view(preds.shape))
            
            for i, batch in enumerate(tqdm.tqdm(test_data, desc="Iteration")):
                # node_features = batch['x'].to(self.device)
                # edge_index = batch['edge_index'].to(self.device)
                # edge_attr = batch['edge_attr'].to(self.device)
                # batch_idx = batch['batch'].to(self.device)
                # Get the prediction of the corresponding task
                
                if self.model_name == 'PreGNN':
                    x, edge_index, edge_attr, batch_idx = batch['x'].to(self.device), batch['edge_index'].to(self.device), batch['edge_attr'].to(self.device), batch['batch'].to(self.device)
                    y = batch['y'].to(self.device)
                    
                    preds = self.predict([x, edge_attr, edge_index, batch_idx])[:, taskID_idx]
                else:
                    smiles, y = batch['vec'], batch['label']
                    smiles = smiles.to(self.device)
                    y = y.to(self.device)
                
                    preds = self.predict(smiles)[:, taskID_idx]
                
                # y_vars.append(vars)
                y_scores.append(preds)
                # y_true.append(batch['y'].to(self.device).view(preds.shape))
                y_true.append(y.view(preds.shape))
            
            
            y_true = torch.cat(y_true, dim = 0).cpu().detach().numpy()
            y_scores = torch.cat(y_scores, dim = 0).cpu().detach().numpy()
            y_predict = np.squeeze(y_scores)
            for idx, val in enumerate(y_predict):
                if val < 0.5 :
                    y_predict[idx] = 0
                else:
                    y_predict[idx] = 1

            results['task_id'].append(task_id)
            results['auc'].append(roc_auc_score(y_true, y_scores))
            results['acc'].append(accuracy_score(y_true, y_scores))
            results['aupr'].append(average_precision_score(y_true, y_scores))
            results['f1'].append(f1_score(y_true, y_scores))
            results['ba'].append(balanced_accuracy_score(y_true, y_scores))
            results['ece'].append(calibration_error(torch.tensor(y_scores), torch.tensor(y_true), n_bins=15, norm='l1', task='binary').item())
            results['nll'].append(F.binary_cross_entropy_with_logits(torch.tensor(y_scores), torch.tensor(y_true, dtype=torch.float32)).item())
            results['mcc'].append(matthews_corrcoef(y_true, y_scores))
            results['precision'].append(precision_score(y_true, y_scores))
            results['recall'].append(recall_score(y_true, y_scores))
        
        return results

    def test_(self, test_data_list, test_data_taskID):
        '''
        return a average auc
        '''
        
        # ood_dataset = OODDataset(ood_dataset)
        # ood_dataloader = DataLoader(ood_dataset, batch_size=128, shuffle=True)
        
        
        self.mtask_model.eval()
        y_true = []
        y_scores = []
        ood_auroc = []
        
        tested_data_num = 0
        for taskID_idx, test_data in enumerate(test_data_list):
            
            task_id = test_data_taskID[taskID_idx]
            print('task_id: ', task_id)
            
            task_data_num = 0
            
            for i, batch in enumerate(tqdm.tqdm(test_data, desc="Iteration")):
                # node_features = batch['x'].to(self.device)
                # edge_index = batch['edge_index'].to(self.device)
                # edge_attr = batch['edge_attr'].to(self.device)
                # batch_idx = batch['batch'].to(self.device)
                # Get the prediction of the corresponding task
                smiles, y = batch['vec'], batch['label']
                smiles = smiles.to(self.device)
                y = y.to(self.device)
                
                # preds = self.predict(node_features, edge_attr, edge_index, batch_idx)[taskID_idx]
                # preds, vars = self.predict(smiles, return_uncertainty=True)
                # preds = preds[taskID_idx]
                # vars = vars[taskID_idx]
                preds = self.predict(smiles)[taskID_idx]
                
                # y_vars.append(vars)
                y_scores.append(preds)
                # y_true.append(batch['y'].to(self.device).view(preds.shape))
                y_true.append(y.view(preds.shape))
                
                task_data_num += len(smiles)

            
        y_true = torch.cat(y_true, dim = 0).cpu().detach().numpy()
        y_scores = torch.cat(y_scores, dim = 0).cpu().detach().numpy()
        y_predict = np.squeeze(y_scores)
        # y_vars = torch.cat(y_vars, dim = 0).cpu().detach().numpy()
        for idx, val in enumerate(y_predict):
            if val < 0.5 :
                y_predict[idx] = 0
            else:
                y_predict[idx] = 1

        
        auc = roc_auc_score(y_true, y_scores)
        acc = accuracy_score(np.squeeze(y_true), y_predict)
        aupr = average_precision_score(np.squeeze(y_true), y_scores)
        f1 = f1_score(np.squeeze(y_true), y_predict)
        ba = balanced_accuracy_score(y_true, y_predict)
        
        # Calculate ECE and NLL
        ece = calibration_error(torch.tensor(y_scores), torch.tensor(y_true), n_bins=15, norm='l1', task='binary').item()
        # ece = ece_with_uncertainty(torch.tensor(y_vars), torch.tensor(y_scores), torch.tensor(y_true), n_bins=15, norm='l1')
        # nll = mean_squared_error(torch.tensor(y_scores), torch.tensor(y_true)).item()
        nll = F.binary_cross_entropy_with_logits(torch.tensor(y_scores), torch.tensor(y_true, dtype=torch.float32)).item()
        
        mcc = matthews_corrcoef(y_true, y_predict)
        precision = precision_score(y_true, y_predict)
        recall = recall_score(y_true, y_predict)
        
        ood_auroc = np.mean(ood_auroc)
        return auc, acc, aupr, f1, ba, ece, nll, ood_auroc, mcc, precision, recall





        

        


        