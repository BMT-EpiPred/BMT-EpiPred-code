import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import numpy as np
from torch.distributions import Normal


initial_func = nn.init.trunc_normal_

# local_reparameterization trick for sampling
def local_reparameterize_softplus(mu, var, repeat):
    '''
       the size of input(mu) is d_out * d_in, we sample one eps for each column.
       the size of output is bs * d_out * d_in.
    '''                 
    eps = torch.cuda.FloatTensor(repeat, mu.shape[0], mu.shape[1]).normal_(0,1)
    sigma = var.sqrt()
    sigma = sigma.expand(repeat, sigma.shape[0], sigma.shape[1])
    mu = mu.expand(repeat, mu.shape[0], mu.shape[1])  
    return mu + sigma*eps

# kl term for VMTL
# 1/2log(var_2/var_1) + (var_1 + (mu_1-mu_2)^2)/2var - 1/2
def kl_criterion_softplus(mu_e, var_e, mu_p, var_p):
    # var_e = var_e + 1e-6
    # var_p = var_p + 1e-6
    component1 = torch.log(var_p) - torch.log(var_e)
    component2 = var_e / var_p
    # component3 = (mu_p - mu_e).pow(2)/ var_p
    component3 = (mu_e - mu_p).pow(2)/ var_p
    KLD = 0.5 * torch.mean((component1 + component2 + component3 - 1), 1)
    return KLD

# gumbel
def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape).cuda()
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature):
    y = torch.log(f.softmax(logits, 1) + 1e-20) + sample_gumbel(logits.size())
    return f.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature, hard=False):
    y = gumbel_softmax_sample(logits, temperature)
    if not hard:
        return y
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    y_hard = (y_hard - y).detach() + y
    return y_hard


class task_shared_network(nn.Module):
    def __init__(self, d_feature, d_latent, device, dropout_index):
        super(task_shared_network, self).__init__()
        self.d_feature = d_feature
        self.d_latent = d_latent
        self.device = device
        self.dropout_index = dropout_index
        # self.rho = -3
        self.rho = 0

        self.phi_mu = nn.Parameter(
            # torch.empty((d_feature, d_latent), device=self.device, dtype=torch.float32).normal_(0., 0.1),
            # torch.empty((d_feature, d_latent), device=self.device, dtype=torch.float32).normal_(0., 1),
            initial_func(
                torch.empty((d_feature, d_latent), device=self.device, dtype=torch.float32)),
                requires_grad=True)
            
        self.phi_logvar = nn.Parameter(
            # self.rho + torch.empty((d_feature, d_latent), device=self.device, dtype=torch.float32).normal_(0., 0.1),
            # self.rho + torch.empty((d_feature, d_latent), device=self.device, dtype=torch.float32).normal_(0., 1),
            self.rho + initial_func(
                torch.empty((d_feature, d_latent), device=self.device, dtype=torch.float32)),
            requires_grad=True)
            

        self.dropout = nn.Dropout(p=self.dropout_index)

        self.phi_bias_mu = nn.Parameter(
            initial_func(
            torch.empty((1, d_latent), device=self.device, dtype=torch.float32)),
            # torch.empty((1, d_latent), device=self.device, dtype=torch.float32).normal_(0., 1),
            requires_grad=True)
        self.phi_bias_logvar = nn.Parameter(
            self.rho + initial_func(torch.empty((1, d_latent), device=self.device, dtype=torch.float32)),
            # self.rho + torch.empty((1, d_latent), device=self.device, dtype=torch.float32).normal_(0., 1),
            requires_grad=True)

    def forward(self, x, z_repeat, usefor):
        x = self.dropout(x)

        z_mu = torch.mm(x, self.phi_mu) + self.phi_bias_mu
        phi_sigma = f.softplus(self.phi_logvar, beta=1, threshold=20)
        phi_bias_sigma = f.softplus(self.phi_bias_logvar, beta=1, threshold=20)
        z_var = torch.mm(x.pow(2), phi_sigma.pow(2)) + phi_bias_sigma.pow(2)
        # m = Normal(z_mu, z_var.sqrt())

        if usefor == "c":
            return z_mu, z_var
        elif usefor == "z":
            if self.training:
                z = local_reparameterize_softplus(z_mu, z_var, z_repeat)  # z_repeat * bs * d_latent
                # z = m.rsample(torch.Size([z_repeat]))

            else:
                # fake sample
                # z = z_mu.expand(z_repeat, z_mu.shape[0], z_mu.shape[1])

                # true sample
                # m = Normal(z_mu, z_var.sqrt())
                # z = m.sample(torch.Size([z_repeat]))
                z = local_reparameterize_softplus(z_mu, z_var, z_repeat)  # z_repeat * bs * d_latent

            z = z.contiguous().view(-1, self.d_latent)
            return z, z_mu, z_var

class task_specific_network(nn.Module):
    def __init__(self, d_class, d_latent, device, classifier_bias):
        super(task_specific_network, self).__init__()
        self.device = device
        self.classifier_bias = classifier_bias
        self.rho = 0
        # self.sigmoid = nn.Sigmoid()

        self.weight_mu = nn.Parameter(
            # torch.empty((d_class, d_latent), device=self.device, dtype=torch.float32).normal_(0., 0.1),
            # torch.empty((d_class, d_latent), device=self.device, dtype=torch.float32).normal_(0., 1),
            initial_func(
                torch.empty((d_class, d_latent), device=self.device, dtype=torch.float32)),
                requires_grad=True)
        self.weight_logvar = nn.Parameter(
            # self.rho + torch.empty((d_class, d_latent), device=self.device, dtype=torch.float32).normal_(0., 0.1),
            # self.rho + torch.empty((d_class, d_latent), device=self.device, dtype=torch.float32).normal_(0., 1),
            self.rho + initial_func(
                torch.empty((d_class, d_latent), device=self.device, dtype=torch.float32)),
            requires_grad=True)

        if self.classifier_bias:
            self.bias_mu = nn.Parameter(
                # torch.empty((d_class, 1), device=self.device, dtype=torch.float32).normal_(0., 0.1), requires_grad=True)
                torch.empty((d_class, 1), device=self.device, dtype=torch.float32).normal_(0., 1), requires_grad=True)
            self.bias_logvar = nn.Parameter(
                # self.rho + torch.empty((d_class, 1), device=self.device, dtype=torch.float32).normal_(0., 0.1),
                self.rho + torch.empty((d_class, 1), device=self.device, dtype=torch.float32).normal_(0., 1),
                requires_grad=True)

    def forward(self, x):
        if self.classifier_bias:
            weight_mu = self.weight_mu + self.bias_mu
            weight_sigma = f.softplus(self.weight_logvar, beta=1, threshold=20)
            bias_sigma = f.softplus(self.bias_logvar, beta=1, threshold=20)
            weight_var = weight_sigma.pow(2) + bias_sigma.pow(2)

            # bias_mu = self.bias_mu
            # bias_sigma = f.softplus(self.bias_logvar, beta=1, threshold=20)
            # bias_var = bias_sigma.pow(2)

            if self.training:
                weight = local_reparameterize_softplus(weight_mu, weight_var, x.shape[0])
                # bias = local_reparameterize_softplus(bias_mu, bias_var, x.shape[0])
            else:
                # weight = weight_mu
                # weight = weight.expand(x.shape[0], weight.shape[0], weight.shape[1])
                m = Normal(weight_mu, weight_sigma)
                weight = m.sample(torch.Size([x.shape[0]]))
                # bias = bias_mu
                # bias = bias.expand(x.shape[0], bias.shape[0], bias.shape[1])
                # m2 = Normal(bias_mu, bias_sigma)
                # bias = m2.sample(torch.Size([x.shape[0]]))

            output = torch.bmm(weight, x.unsqueeze(2)).squeeze(2)
        else:
            weight_mu = self.weight_mu
            weight_sigma = f.softplus(self.weight_logvar, beta=1, threshold=20)
            weight_var = weight_sigma.pow(2)
            # m = Normal(weight_mu, weight_sigma)

            if self.training:
                weight = local_reparameterize_softplus(weight_mu, weight_var, x.shape[0])
                # weight = m.rsample(torch.Size([x.shape[0]]))
            else:
                # true sample
                # m = Normal(weight_mu, weight_sigma)
                # weight = m.sample(torch.Size([x.shape[0]]))
                weight = local_reparameterize_softplus(weight_mu, weight_var, x.shape[0])


                # fake sample
                # weight = self.weight_mu
                # weight = weight.expand(x.shape[0], weight.shape[0], weight.shape[1])

            output = torch.bmm(weight, x.unsqueeze(2)).squeeze(2)
        # output = self.sigmoid(output)
        return output, weight_mu, weight_var

class task_specific_gumbel(nn.Module):
    def __init__(self, device, d_task):
        super(task_specific_gumbel, self).__init__()
        self.device = device
        self.gumbel = nn.Parameter(
            nn.init.constant_(torch.empty((1, d_task - 1), device=self.device, dtype=torch.float32), 0.0),
            requires_grad=True)
        self.gumbel_w = nn.Parameter(
            nn.init.constant_(torch.empty((1, d_task - 1), device=self.device, dtype=torch.float32), 0.0),
            requires_grad=True)

    def forward(self, temp, gumbel_type):
        if gumbel_type == "feature":
            logits = self.gumbel
        elif gumbel_type == "classifier":
            logits = self.gumbel_w
        current_prior_weights = gumbel_softmax(logits, temp, False)
        probability = torch.sigmoid(logits)
        return current_prior_weights.transpose(0, 1), probability.transpose(0, 1)

# class MainModel(object):
#     def __init__(self, dataset, split_name, task_num, network_name, class_num, file_out, optim_param, all_parameters, d_feature):

#         self.dataset = dataset
#         self.split_name = split_name
#         self.temp, self.anneal, self.d_latent, self.num, self.dropout_index= all_parameters
#         self.classifier_bias = False

#         self.temp_min = 0.5
#         self.ANNEAL_RATE = 0.00003
#         self.device = 'cuda'
#         self.train_cross_loss = 0.0
#         self.train_kl_loss = 0.0
#         self.train_kl_w_loss = 0.0
#         self.train_kl_z_loss = 0.0
#         self.train_total_loss = 0.0
#         self.beta = 1e-06
#         self.target_CE = 0.0
#         self.file_out = file_out
#         self.print_interval = 10
#         self.eta = 1e-07

#         self.optim_param = optim_param
#         for val in optim_param:
#             self.optim_param[val] = optim_param[val]

#         self.task_num = task_num
#         self.network_name = network_name
#         self.d_feature = d_feature
#         # self.d_feature = d_latent
#         self.d_class = class_num
#         # network initialization************************************************************
#         self.shared_encoder = task_shared_network(self.d_feature, self.d_latent, self.device, self.dropout_index)
#         parameter_encoder = [{"params": self.shared_encoder.parameters(), "lr": 1}]

#         self.specific_w_list = []
#         self.parameters_all = []
#         self.gumbel_list = []
#         self.optimizer_list = []

#         # self.save_z_mu = []
#         # self.save_z_var = []
#         self.save_w_mu = []
#         self.save_w_var = []
#         self.save_b_mu = []
#         self.save_b_var = []

#         for i in range(self.task_num):
#             self.specific_w_list.append(task_specific_network(self.d_class, self.d_latent, self.device, self.classifier_bias))
#             self.parameters_all.append([{"params": self.specific_w_list[i].parameters(), "lr": 1}] + parameter_encoder)
#             self.gumbel_list.append(task_specific_gumbel(self.device, self.task_num))
#             self.parameters_all[i] = self.parameters_all[i] + [{"params": self.gumbel_list[i].parameters(), "lr": 1}]
#             # self.optimizer_list.append(optim.Adam(self.parameters_all[i], lr=1, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0005))
#             self.optimizer_list.append(optim.Adam(self.parameters_all[i], lr=1, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.002))

#             # self.save_z_mu.append(torch.zeros(self.d_class, self.d_latent).cuda())
#             # self.save_z_var.append(torch.ones(self.d_class, self.d_latent).cuda())
#             self.save_w_mu.append(torch.zeros(self.d_class, self.d_latent).cuda())
#             self.save_w_var.append(torch.ones(self.d_class, self.d_latent).cuda())
#             self.save_b_mu.append(torch.zeros(self.d_class, 1).cuda())
#             self.save_b_var.append(torch.ones(self.d_class, 1).cuda())

#         self.criterion = nn.CrossEntropyLoss()
#         # self.criterion = nn.BCEWithLogitsLoss()
#         self.iter_num = 1
#         self.counter = 1
#         self.current_lr = 0.0
#         self.z_repeat = 10

#         self.z_mu_prior = nn.Parameter(
#             nn.init.zeros_(torch.empty((self.d_class, self.d_latent), device=self.device, dtype=torch.float32)),
#             requires_grad=False)
#         self.z_var_prior = nn.Parameter(
#             nn.init.ones_(torch.empty((self.d_class, self.d_latent), device=self.device, dtype=torch.float32)),
#             requires_grad=False)
#         self.w_mu_prior = nn.Parameter(
#             nn.init.zeros_(torch.empty((self.d_class, self.d_latent), device=self.device, dtype=torch.float32)),
#             requires_grad=False)
#         self.w_var_prior = nn.Parameter(
#             nn.init.ones_(torch.empty((self.d_class, self.d_latent), device=self.device, dtype=torch.float32)),
#             requires_grad=False)
#         # self.b_mu_prior = nn.Parameter(
#             # nn.init.zeros_(torch.empty((self.d_class, 1), device=self.device, dtype=torch.float32)),
#             # requires_grad=False)
#         # self.b_var_prior = nn.Parameter(
#         #     nn.init.constant_(torch.empty((self.d_class, 1), device=self.device, dtype=torch.float32), 1),
#         #     requires_grad=False)
#     def optimize_model(self, input_list, label_list, number, related_inputs):

#         # update learning rate for different networks
#         if self.optimizer_list[0].param_groups[0]["lr"] >= 0.000002:
#             self.current_lr = self.optim_param["init_lr"] * (
#                         self.optim_param["gamma"] ** (self.iter_num // self.optim_param["stepsize"]))
#         for optimizer in self.optimizer_list:
#             for component in optimizer.param_groups:
#                 component["lr"] = self.current_lr * 1.0

#         # training
#         self.shared_encoder.train()
#         self.specific_w_list[number].train()
#         self.gumbel_list[number].train()

#         # inference
#         z, z_mu, z_var = self.shared_encoder(input_list, self.z_repeat,  "z")
#         output, w_mu, w_var = self.specific_w_list[number](z)
#         # outputs = torch.split(output, input_list.size()[0])
#         output_mean = torch.mean(output.view(self.z_repeat, input_list.size()[0], output.size()[1]), dim=0).squeeze()
        
#         self.save_w_mu[number] = w_mu
#         self.save_w_var[number] = w_var
        
        
#         # log-likelihood
#         # re_label_list = label_list.expand(self.z_repeat, label_list.shape[0]).contiguous().view(-1)

#         # predict = f.softmax(output, dim=1)
#         # cls_loss = self.criterion(output, re_label_list)
#         cls_loss = self.criterion(output_mean, label_list)

#         # kl_divergence
#         q_z_mu = z_mu
#         q_z_var = z_var
#         q_w_mu = w_mu
#         q_w_var = w_var
#         if True:
#         # if self.counter < self.task_num:
#             p_z_mu = self.z_mu_prior[label_list]
#             p_z_var = self.z_var_prior[label_list]
#             p_w_mu = self.w_mu_prior
#             p_w_var = self.w_var_prior

#             kl_w = torch.mean(kl_criterion_softplus(q_w_mu, q_w_var, p_w_mu, p_w_var))
#             kl_z = torch.mean(kl_criterion_softplus(q_z_mu, q_z_var, p_z_mu, p_z_var))
#         else:
#             task_order = range(self.task_num)
#             task_list = list(task_order)
#             task_list.remove(number)

#             current_prior_weights_feat, probability_feat = self.gumbel_list[number](self.temp, "feature")
#             current_prior_weights_clas, probability_clas = self.gumbel_list[number](self.temp, "classifier")

#             p_z_mu = 0.0
#             p_z_var = 0.0
#             p_w_mu = 0.0
#             p_w_var = 0.0
#             for i in range(len(task_list)):
#                 p_number = task_list[i]
#                 current_coefficient_feat = current_prior_weights_feat[i]  # 1*1
#                 current_coefficient_clas = current_prior_weights_clas[i]  # 1*1
                

#                 p_z_mu_element, p_z_var_element = self.shared_encoder(related_inputs[i], None,"c")
#                 p_z_mu_element_ = current_coefficient_feat * p_z_mu_element.detach()
#                 p_z_var_element_ = current_coefficient_feat.pow(2) * p_z_var_element.detach()
#                 p_z_mu += p_z_mu_element_
#                 p_z_var += p_z_var_element_

#                 p_w_mu_element = self.save_w_mu[p_number]
#                 p_w_var_element = self.save_w_var[p_number]
#                 p_w_mu_element_ = current_coefficient_clas * p_w_mu_element.detach()
#                 p_w_var_element_ = current_coefficient_clas.pow(2) * p_w_var_element.detach()
#                 p_w_mu += p_w_mu_element_
#                 p_w_var += p_w_var_element_

#             kl_w = torch.sum(kl_criterion_softplus(q_w_mu, q_w_var, p_w_mu, p_w_var))
#             kl_z = torch.mean(kl_criterion_softplus(q_z_mu, q_z_var, p_z_mu, p_z_var))
#         # kl_w = self.beta * kl_w
#         # kl_z = self.eta * kl_z
#         # kl_w = 1e-3 * kl_w
#         # kl_z = 1e-3 * kl_z
#         # cls_loss = 50 * cls_loss

#         # loss function
#         kl_loss = kl_w + kl_z
#         loss = cls_loss + kl_loss
#         # loss = cls_loss

#         # updates
#         self.optimizer_list[number].zero_grad()
#         loss.backward()
#         self.optimizer_list[number].step()

#         # w_mu.detach()
#         # w_var.detach()
#         # self.save_w_mu[number].detach()
#         # self.save_w_var[number].detach()
        

#         # -----------------------------------------------------------------------------------------------
#         # annealing strategy
#         self.counter += 1
#         batchtask = self.task_num
#         if self.counter % batchtask == 0:
#             self.iter_num += 1
#         if self.iter_num % 10 == 0:
#             self.beta += 1e-06
#             # self.beta += 1e-05
#             # self.eta += 1e-07
#         if self.anneal:
#             if self.iter_num % 1000 == 0:
#                 self.temp = np.max([self.temp * np.exp(-self.ANNEAL_RATE * self.iter_num), self.temp_min])

#         # print
#         self.train_cross_loss += cls_loss.item()
#         self.train_kl_loss += kl_loss.item()
#         self.train_kl_w_loss += kl_w.item()
#         self.train_kl_z_loss += kl_z.item()
#         self.train_total_loss += loss.item()

#         if self.counter == self.task_num + 1:
#             print("Iter {:05d}, lr:{:.6f}, Average CE Loss: {:.4f}; Average KL Loss: {:.4f}; Average KL_weight Loss: {:.4f}; Average KL_z Loss: {:.4f}; Average Training Loss: {:.4f}".format(
#                     int(self.counter / batchtask), self.current_lr,
#                     self.train_cross_loss / float(self.counter),
#                     self.train_kl_loss / float(self.counter),
#                     self.train_kl_w_loss / float(self.counter),
#                     self.train_kl_z_loss / float(self.counter),
#                     self.train_total_loss / float(self.counter)))
#             self.file_out.write("Iter {:05d}, lr:{:.6f}, Average CE Loss: {:.4f}; Average KL Loss: {:.4f}; Average KL_weight Loss: {:.4f}; Average KL_z Loss: {:.4f}; Average Training Loss: {:.4f}\n".format(
#                     int(self.counter / batchtask), self.current_lr,
#                     self.train_cross_loss / float(self.counter),
#                     self.train_kl_loss / float(self.counter),
#                     self.train_kl_w_loss / float(self.counter),
#                     self.train_kl_z_loss / float(self.counter),
#                     self.train_total_loss / float(self.counter)))

#         if self.counter % (self.print_interval * batchtask) == 0:
#             print("Iter {:05d}, lr:{:.6f}, Average CE Loss: {:.4f}; Average KL Loss: {:.4f}; Average KL_weight Loss: {:.4f}; Average KL_z Loss: {:.4f}; Average Training Loss: {:.4f}".format(
#                     int(self.counter / batchtask), self.current_lr,
#                     self.train_cross_loss / float(self.print_interval * batchtask),
#                     self.train_kl_loss / float(self.print_interval * batchtask),
#                     self.train_kl_w_loss / float(self.print_interval * batchtask),
#                     self.train_kl_z_loss / float(self.print_interval * batchtask),
#                     self.train_total_loss / float(self.print_interval * batchtask)))
#             self.file_out.write("Iter {:05d}, lr:{:.6f}, Average CE Loss: {:.4f}; Average KL Loss: {:.4f}; Average KL_weight Loss: {:.4f}; Average KL_z Loss: {:.4f}; Average Training Loss: {:.4f}\n".format(
#                     int(self.counter / batchtask), self.current_lr,
#                     self.train_cross_loss / float(self.print_interval * batchtask),
#                     self.train_kl_loss / float(self.print_interval * batchtask),
#                     self.train_kl_w_loss / float(self.print_interval * batchtask),
#                     self.train_kl_z_loss / float(self.print_interval * batchtask),
#                     self.train_total_loss / float(self.print_interval * batchtask)))

#             self.file_out.flush()
#             self.target_CE = self.train_cross_loss / float(self.print_interval * batchtask)
#             self.train_cross_loss = 0
#             self.train_kl_loss = 0
#             self.train_kl_w_loss = 0
#             self.train_kl_z_loss = 0
#             self.train_total_loss = 0
        
        

#     def test_model(self, input_, label, i):
#         self.shared_encoder.eval()
#         self.specific_w_list[i].eval()

#         z, _, _ = self.shared_encoder(input_, self.z_repeat, "z")
#         output, _, _ = self.specific_w_list[i](z)

#         # outputs = torch.split(output.view(self.z_repeat, input_.size()[0], calss_num), 1)
#         # torch.mean()
#         output_mean = torch.mean(output.view(self.z_repeat, input_.size()[0], output.size()[1]), dim=0)

#         # _, output_predict = torch.max(output, 1)
#         _, output_predict = torch.max(output_mean, 1)
#         # output_predict = torch.round(output_mean)
#         # re_label = label.expand(self.z_repeat, label.shape[0]).contiguous().view(-1)
#         re_label = label

#         return output_predict, re_label

class MT_model(nn.Module):  # 修改：继承nn.Module
    def __init__(self, args):
        super(MT_model, self).__init__()  # 新增
        self.temp, self.anneal, self.d_latent, self.dropout_index = 1, True, 1500, 0.7
        self.classifier_bias = False

        self.temp_min = 0.5
        self.ANNEAL_RATE = 0.00003
        self.device = 'cuda'
        self.train_cross_loss = 0.0
        self.train_kl_loss = 0.0
        self.train_kl_w_loss = 0.0
        self.train_kl_z_loss = 0.0
        self.train_total_loss = 0.0
        self.beta = 1e-06
        self.target_CE = 0.0
        self.print_interval = 10
        self.eta = 1e-07


        self.task_num = 78
        self.d_feature = 1024
        self.d_class = 2

        self.shared_encoder = task_shared_network(self.d_feature, self.d_latent, self.device, self.dropout_index)
        self.specific_w_list = nn.ModuleList([
            task_specific_network(self.d_class, self.d_latent, self.device, self.classifier_bias)
            for _ in range(self.task_num)
        ])
        self.gumbel_list = nn.ModuleList([
            task_specific_gumbel(self.device, self.task_num)
            for _ in range(self.task_num)
        ])

        self.criterion = nn.CrossEntropyLoss()
        self.iter_num = 1
        self.counter = 1
        self.current_lr = 0.0
        self.z_repeat = 10

        self.z_mu_prior = nn.Parameter(
            nn.init.zeros_(torch.empty((self.d_class, self.d_latent), device=self.device, dtype=torch.float32)),
            requires_grad=False)
        self.z_var_prior = nn.Parameter(
            nn.init.ones_(torch.empty((self.d_class, self.d_latent), device=self.device, dtype=torch.float32)),
            requires_grad=False)
        self.w_mu_prior = nn.Parameter(
            nn.init.zeros_(torch.empty((self.d_class, self.d_latent), device=self.device, dtype=torch.float32)),
            requires_grad=False)
        self.w_var_prior = nn.Parameter(
            nn.init.ones_(torch.empty((self.d_class, self.d_latent), device=self.device, dtype=torch.float32)),
            requires_grad=False)

        # 用于保存每个任务的w_mu, w_var, b_mu, b_var
        self.save_w_mu = [torch.zeros(self.d_class, self.d_latent).cuda() for _ in range(self.task_num)]
        self.save_w_var = [torch.ones(self.d_class, self.d_latent).cuda() for _ in range(self.task_num)]
        self.save_b_mu = [torch.zeros(self.d_class, 1).cuda() for _ in range(self.task_num)]
        self.save_b_var = [torch.ones(self.d_class, 1).cuda() for _ in range(self.task_num)]

    def forward(self, input_list, return_params=False):
        """
        前向传播，返回 logits
        input_list: 输入特征 (batch, d_feature)
        task_idx: int, 任务编号
        """
        z, z_mu, z_var = self.shared_encoder(input_list, self.z_repeat, "z")
        
        outs = []
        w_mu_list = []
        w_var_list = []
        for task_idx in range(self.task_num):
            # 对每个任务调用对应的特定网络
            output, w_mu, w_var = self.specific_w_list[task_idx](z)
            output_mean = torch.mean(output.view(self.z_repeat, input_list.size()[0], output.size()[1]), dim=0).squeeze()
            # 保存w_mu, w_var
            self.save_w_mu[task_idx] = w_mu
            self.save_w_var[task_idx] = w_var
            
            outs.append(output_mean)
            w_mu_list.append(w_mu)
            w_var_list.append(w_var)
        
        # 将所有任务的输出合并
        output_mean = torch.stack(outs, dim=0).permute(1, 0, 2)  # [batch, task_num, class_num]
        w_mu = torch.stack(w_mu_list, dim=0)
        w_var = torch.stack(w_var_list, dim=0)
        
        # output, w_mu, w_var = self.specific_w_list[task_idx](z)
        # output_mean = torch.mean(output.view(self.z_repeat, input_list.size()[0], output.size()[1]), dim=0).squeeze()
        # # 保存w_mu, w_var
        # self.save_w_mu[task_idx] = w_mu
        # self.save_w_var[task_idx] = w_var
        if return_params:
            return output_mean, z_mu, z_var, w_mu, w_var
        else:
            return output_mean

    def compute_loss(self, output_mean, label_list, z_mu, z_var, w_mu, w_var):
        """
        计算多任务损失
        output_mean: [batch, task_num, class_num]
        label_list: [batch, task_num]
        w_mu, w_var: [task_num, ...]
        z_mu, z_var: [batch, d_latent]
        """
        total_cls_loss = 0.0
        total_kl_w = 0.0
        total_kl_z = 0.0
        for task_idx in range(self.task_num):
            logits = output_mean[:, task_idx]           # [batch, class_num]
            labels = label_list[:, task_idx]            # [batch]
            w_mu_t = w_mu[task_idx]
            w_var_t = w_var[task_idx]
            # z_mu, z_var 是共享的
            cls_loss = self.criterion(logits, labels)
            p_z_mu = self.z_mu_prior[labels]
            p_z_var = self.z_var_prior[labels]
            p_w_mu = self.w_mu_prior
            p_w_var = self.w_var_prior
            kl_w = torch.mean(kl_criterion_softplus(w_mu_t, w_var_t, p_w_mu, p_w_var))
            kl_z = torch.mean(kl_criterion_softplus(z_mu, z_var, p_z_mu, p_z_var))
            total_cls_loss += cls_loss
            total_kl_w += kl_w
            total_kl_z += kl_z
        kl_loss = total_kl_w + total_kl_z
        loss = total_cls_loss + kl_loss
        return loss

    def predict(self, input_list):
        """
        返回所有任务的预测类别 [task_num, batch]
        """
        self.eval()
        with torch.no_grad():
            output_mean, _, _, _, _ = self.forward(input_list)
            # output_mean: [task_num, batch, class_num]
            preds = torch.argmax(output_mean, dim=2)  # [task_num, batch]
        return preds

