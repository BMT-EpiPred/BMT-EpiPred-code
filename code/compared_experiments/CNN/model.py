# Import Python libraries
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.optim as optim
import numpy as np

##########################################################################################
class CNN_MFembedded_BCM(nn.Module):
    def __init__(self, bit_size = 1024, embedding_size = 164, max_length = 164, window_height = 69, num_filter = 4096, dropout_rate=0.5, out_fdim = 1024):
        '''
        we use max_length = 164 since the ecpf4 of compounds in HME dataset have max size of 164
        we define 1024 as out fdim so that the muti-task model shall be easy to accept
        '''
        super(CNN_MFembedded_BCM, self).__init__()
        #Network Architecture
        self.embeddings     = nn.Embedding(bit_size + 1, embedding_size, padding_idx = 0) #(100, 100)
        self.bnorm_emb      = nn.BatchNorm2d(num_features = 1)   
        #####################
        self.conv1          = nn.Conv2d(in_channels = 1, out_channels = num_filter, kernel_size = (69, 1024), stride=(1, 1)) # (4096, 96, 1)
        self.bnorm_conv1    = nn.BatchNorm2d(num_features = num_filter)
        self.activate_conv1 = nn.LeakyReLU()
        self.pool_conv1     = nn.MaxPool2d(kernel_size = (max_length - window_height + 1, 1))
        # self.drop_conv1     = nn.Dropout(p=dropout_rate)
        #####################
        self.linear1        = nn.Linear(num_filter, out_fdim)
        self.bnorm_l1       = nn.BatchNorm1d(num_features = 1)
        self.activate_l1    = nn.LeakyReLU()
        self.drop1          = nn.Dropout(p=dropout_rate)
        # #####################
        # self.linear2        = nn.Linear(512, 256)
        # self.bnorm_l2       = nn.BatchNorm1d(num_features = 1)
        # self.activate_l2    = nn.LeakyReLU()
        # self.drop2          = nn.Dropout(p=dropout_rate)
        # #####################
        # self.linear3        = nn.Linear(256, 1)
        # self.activate_l3    = nn.Sigmoid()
        #####################
        #Variables
        self.embedded_size  = embedding_size
        self.max_length     = max_length
        self.pooled_size    = num_filter
        self.bit_size        = bit_size
        
    def forward(self, x: torch.Tensor):
        '''
        get init feature using as the input of muti task model
        @return
            output1 -> torch.size(batch_size, 1024(num_filter)) 
        '''
        
        embeds              = self.embeddings(x.long()).view(x.shape[0], 1, self.bit_size, self.embedded_size)
        embeds              = embeds.permute(0, 1, 3, 2)  # (batch_size, embedding_size, 1, bit_size)
        embeds              = self.bnorm_emb(embeds)
        #####################
        output1             = self.conv1(embeds)
        output1             = self.bnorm_conv1(output1)
        output1             = self.activate_conv1(output1)
        output1             = self.pool_conv1(output1).view(-1, 1, self.pooled_size) 
        # output1             = self.drop_conv1(output1)
        #####################
        output2             = self.linear1(output1) 
        output2             = self.bnorm_l1(output2) 
        output2             = self.activate_l1(output2) 
        output2             = self.drop1(output2) 
        # #####################
        # output3             = self.linear2(output2)
        # output3             = self.bnorm_l2(output3) 
        # output3             = self.activate_l2(output3)
        # output3             = self.drop2(output3)            
        # #####################
        # output4             = self.linear3(output3)
        # output4             = self.activate_l3(output4)
        return output2
##########################################################################################


class MT_DNN(torch.nn.Module):
    def __init__(self, in_fdim, layer_size, num_tasks):
        super(MT_DNN, self).__init__()
        self.in_fdim = in_fdim
        # self.out_fdim = out_fdim
        self.layer_size = layer_size
        self.num_tasks = num_tasks

        last_fdim = self.create_bond()
        self.create_sphead(last_fdim)
        
        

    def create_bond(self):
        '''
        Creates the feed-forward layers for the model.

        concrete params depends on kniome paper
    
        '''
        activation = nn.ReLU()
        dropout = nn.Dropout(p=0.5)     
        last_fdim = 1000

        ffn = []

        # Create FFN layers
        if self.layer_size == 'shallow':
            # [1000]
            ffn.extend([
                torch.nn.Linear(self.in_fdim, 1000),
                activation,
            ])
        
        elif self.layer_size == 'moderate':
            # [1500, 1000]
            ffn.extend([
                torch.nn.Linear(self.in_fdim, 1500),
                activation
            ])
            ffn.extend([
                dropout,
                torch.nn.Linear(1500, 1000),
                activation
            ])
            #last_fdim = 1000
        
        elif self.layer_size == 'deep':
            # [2000, 1000, 500]
            ffn = [
                torch.nn.Linear(self.in_fdim, 2000),
                activation
            ]
            for i in range(1,3):
                ffn.extend([
                    dropout,
                    torch.nn.Linear(2000//i, 1000//i),
                    activation
                ])
            last_fdim = 500
        elif self.layer_size == 'p_best':
            # [1024, num_tasks]
            ffn.extend([
                torch.nn.Linear(self.in_fdim, 1024),
                activation
            ])
            ffn.extend([
                dropout,
                torch.nn.Linear(1024, self.num_tasks),
                activation
            ])
            last_fdim = self.num_tasks
        else:
             raise ValueError("unmatched layer_size(shallow, moderate, deep, p_best).")
            
           
        self.bond = torch.nn.Sequential(*ffn)
        
        return last_fdim
    
    def create_sphead(self,last_fdim):
        '''
        create task specific output layers
        '''
        heads = []
        if self.num_tasks < 1:
            raise ValueError("unmatched task_num, which must greater than 1.")
        
        # each task has it's own specific output layer(head)
        for _ in range(self.num_tasks):
            #define the task as a bio-classifing problem
            ffn = nn.Linear(last_fdim, 2)
            heads.append(ffn)
        
        self.heads = nn.ModuleList(heads)
    
    def forward(self, in_feat):
        # TODO:is there a dropout layer between bond and head? now the dropout layer is missing
        out_feat = self.bond(in_feat)
        
        # compute output for each task through corresponding head
        output = []
        
        for head in self.heads:
            output.append(head(out_feat))
        
        return output

class MT_model(nn.Module):
    def __init__(self, args):
        super(MT_model, self).__init__()
        self.args = args
        self.num_tasks = args.num_tasks
        self.layer_size = args.layer_size
        self.in_fdim = 1024
        
        self.cnn_model = CNN_MFembedded_BCM()
        self.mtask_model = MT_DNN(
            in_fdim=self.in_fdim,
            layer_size=self.layer_size,
            num_tasks=self.num_tasks
        )
        
    #     self.create_sphead(1500)
        
    # def create_sphead(self,last_fdim):
    #     '''
    #     create task specific output layers
    #     '''
    #     heads = []
    #     if self.num_tasks < 1:
    #         raise ValueError("unmatched task_num, which must greater than 1.")
        
    #     # each task has it's own specific output layer(head)
    #     for _ in range(self.num_tasks):
    #         #define the task as a bio-classifing problem
    #         ffn = nn.Linear(last_fdim, 2)
    #         heads.append(ffn)
        
    #     self.heads = nn.ModuleList(heads)

    def forward(self, x, return_params=False):
        # in_feat = self.cnn_model(x)
        in_feat = self.cnn_model(x.long()).reshape(x.size()[0], 1024)
        output = self.mtask_model(in_feat)
        output = torch.stack(output).squeeze().permute(1, 0, 2)  # (batch_size, num_tasks, 2)
        return output
        # compute output for each task through corresponding head
        # output = []
        # for head in self.heads:
        #     output.append(head(out_feat))
        # output = torch.stack(output).squeeze().permute(1, 0, 2)  # (batch_size, num_tasks, 2)
        # return output