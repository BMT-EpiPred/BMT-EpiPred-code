import argparse
import os
import warnings
import torch
import numpy as np


import pandas as pd
# from meta_model import Meta_model
from mtask_model import Mtask_model

from dataset import ECFPDataset, SmilesDataset, GraphDataset
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GeometricDataLoader
import pandas as pd
from config import args




def main(split_idx, model_name, num_workers):
    warnings.filterwarnings("ignore", message="DEPRECATION WARNING: please use MorganGenerator")
    args.split_idx = split_idx
    num_workers = num_workers

    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)

    # there are 78 protein in HME dataset
    train_percent = 80
    path_head = args.path_head

    args.dataset = "HME_dataset_{}".format(args.layer_size)
    print(args.dataset)

    
    # penalty coefficent for data balancing, num of inactive data / num of active data
    with open(path_head.split("split")[0] + "/train_{}_{}_penalty.txt".format(train_percent, split_idx), "r") as fw:
        penalty_coefficients = fw.readline().split()
        args.penalty_coefficients = list(map(float, penalty_coefficients))
    
    

    valid_data_list = []
    valid_data_taskID = []
   
    # get data in each task
    s_train, l_train = [], [],
    weights = []
    for task_idx in range(args.num_tasks):
        task_id = args.task_no[task_idx]
        # trian set
        with open(path_head + str(task_id) + '/train_{}_{}.txt'.format(train_percent, split_idx), 'r') as fw:
            for line in fw.readlines():
                smiles, label = line.split()
                label = int(label)
                if smiles not in s_train:
                    s_train.append(smiles)

                    labels = [0] * args.num_tasks
                    labels = np.array(labels)
                    labels[task_idx] = label
                    l_train.append(labels)

                    weight = labels.copy()
                    weight[task_idx] = 1
                    weights.append(weight)
                else:
                    idx = s_train.index(smiles)
                    l_train[idx][task_idx] = label
                    weights[idx][task_idx] = 1

        # valid set
        with open(path_head + str(task_id) + '/valid_{}_{}.txt'.format(train_percent, split_idx), 'r') as fw:
            s_test, l_test = [], []
            for line in fw.readlines():
                smiles, label = line.split()
                label = int(label)
                s_test.append(smiles)
                l_test.append(label)

        if model_name == 'PreGNN':
            task_valid_set = GraphDataset(s_test, l_test, task_id=task_id)
            # valid data
            valid_data = GeometricDataLoader(task_valid_set, batch_size = args.batch_size_eval)
        else:
            task_valid_set =  ECFPDataset(s_test, l_test, task_id=task_id)
            # valid data
            valid_data = DataLoader(task_valid_set, batch_size = args.batch_size_eval)
        valid_data_list.append(valid_data)
        # valid ids are mapped with test_data_list
        valid_data_taskID.append(task_valid_set.get_task_id())

    if model_name == 'PreGNN':
        train_set = GeometricDataLoader(GraphDataset(s_train, l_train, weights=weights), batch_size = args.batch_size, num_workers=num_workers, shuffle=True)
    else:
        train_set = DataLoader(SmilesDataset(s_train, l_train, weights=weights), batch_size = args.batch_size, num_workers=num_workers, shuffle=True)


    model = Mtask_model(args, s_train, model_name=model_name).to(device)


    new_dataset_param_prefix = "params/"+ args.dataset + "_param/{}/".format(model_name)
    if not os.path.exists(new_dataset_param_prefix ):
        os.makedirs(new_dataset_param_prefix)
    pretrain_model_path = new_dataset_param_prefix + 'parameter_train_{}_{}'.format(train_percent, split_idx)

    new_dataset_res_prefix = "result/" + args.dataset + "_res/{}/".format(model_name)
    if not os.path.exists(new_dataset_res_prefix ):
        os.makedirs(new_dataset_res_prefix)

    result_path = new_dataset_res_prefix + args.dataset 
    best_acc = 0
    best_auc = 0
    best_aupr = 0
    best_f1 = 0
    best_ba = 0
    best_ece = 0
    best_nll = 0
    best_mcc = 0
    best_precision = 0
    best_recall = 0
    
   
    epoch = 0
    max_epoch = args.epoch
    
    while epoch < max_epoch + 1:
    # while False:
        torch.cuda.empty_cache()
        model.train(train_set)

        if epoch % 10 == 0:
            with torch.no_grad():
                auc, acc, aupr, f1, ba, ece, nll, mcc, precision, recall = model.test_for_comp(valid_data_list, valid_data_taskID)
                
                if auc > best_auc:
                    best_auc = auc
                    best_aupr = aupr
                    best_acc = acc
                    best_f1 = f1
                    best_ba = ba
                    best_ece = ece
                    best_nll = nll
                    best_mcc = mcc
                    best_precision = precision
                    best_recall = recall
                    
                    torch.save(model.state_dict(), pretrain_model_path+'_best_valid_auc.pkl')

                fw = open(result_path + "_train_{}_{}_valid.txt".format(train_percent, split_idx), "a+")
                
                fw.write("*"*20+"epoch: " + "\t")
                fw.write(str(epoch) + "\t")
                fw.write("*"*20 + "\n")

                fw.write("valid set: ACC = ")
                fw.write(str(acc))
                fw.write("\t AURoc = {} \t".format(auc))
                fw.write("AUPR = {} \t".format(aupr))
                fw.write("f1 = {} \t".format(f1))
                fw.write("ba = {} \t".format(ba))
                fw.write("mcc = {} \t".format(mcc))
                fw.write("precision = {} \t".format(precision))
                fw.write("recall = {} \t".format(recall))
                fw.write("ece = {} \t".format(ece))
                fw.write("nll = {} \t".format(nll))
                fw.write('\n')

                fw.write("best:\t ACC = ")
                fw.write(str(best_acc))
                fw.write("\t AURoc = {} \t".format(best_auc))
                fw.write("AUPR = {} \t".format(best_aupr))
                fw.write("f1 = {} \t".format(best_f1))
                fw.write("ba = {} \t".format(best_ba))
                fw.write("mcc = {} \t".format(best_mcc))
                fw.write("precision = {} \t".format(best_precision))
                fw.write("recall = {} \t".format(best_recall))
                fw.write("ece = {} \t".format(best_ece))
                fw.write("nll = {} \t".format(best_nll))
                fw.write("\n\n")
                fw.close()
                
                f_draw = open(result_path + "_train_{}_{}_fulldata.txt".format(train_percent, split_idx), "a+")
                # f_ood.write('epoch: {}, ood_auroc: {}, acc: {}\n'.format(epoch, ood_auroc, acc))
                f_draw.write('epoch: {}, acc: {}, aupr: {}, f1: {}, ba: {}, mcc: {}, precision: {}, recall: {}, ece: {}, nll: {}\n'.format(epoch, acc, aupr, f1, ba, mcc, precision, recall, ece, nll))
                f_draw.close()

        epoch += 1



if __name__ == "__main__":
    # model_list = ['CNN', 'PreGNN', 'VMTL', 'ML_DTL']
    model_list = ['PreGNN']
    num_workers_list = [0]
    
    # model_list = ['ML_DTL']
    for model_name, num_workers in zip(model_list, num_workers_list):
        for split_idx in [0]:
            main(split_idx, model_name, num_workers)
