"""
This file defines the model.
Date: 2022-07-30
"""
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
import math

class DiagLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(DiagLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        self.diag=torch.eye(self.in_features,requires_grad=False).cuda()
        return F.linear(input, self.diag.mul(self.weight), self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class HEP(nn.Module):
    def __init__(self,args,x_mean_g,device):

        super(HEP, self).__init__()
        self.args = args
        self.input_size= args.input_size
        self.hidden_size = args.hidden_size
        self.delta_size = args.input_size
        self.mask_size = args.input_size
        self.device = device
        self.static_size=args.static_size
        self.text_embed_size=args.text_embed_size

        self.identity = torch.eye(self.input_size).cuda()
        self.x_mean_g = torch.Tensor(x_mean_g).cuda() #经验均值 即所有病人该变量的均值


        self.zl = nn.Linear(self.input_size + self.hidden_size + self.mask_size, self.hidden_size)
        self.rl = nn.Linear(self.input_size + self.hidden_size + self.mask_size, self.hidden_size)
        self.hl = nn.Linear(self.input_size + self.hidden_size + self.mask_size, self.hidden_size)

        self.gamma_x_l = DiagLinear(self.delta_size, self.delta_size)

        self.gamma_h_l = nn.Linear(self.delta_size, self.hidden_size)

        self.output_layer_onedirection=nn.Linear(self.hidden_size,self.hidden_size)
        self.output_layer_bidirection = nn.Linear(self.hidden_size*2, self.hidden_size)

        self.staic_linear=nn.Linear(self.static_size,self.hidden_size)
        self.text_liear=nn.Linear(self.text_embed_size,self.hidden_size)
        self.output_layer_2 = nn.Linear(self.hidden_size*4,1)

        self.bn_bidirection=nn.BatchNorm1d(self.hidden_size*2)
        self.bn_onedirection = nn.BatchNorm1d(self.hidden_size)

        #self.att = nn.Sequential(nn.Linear(self.input_size,self.input_size),nn.Linear(self.input_size,self.input_size),nn.Softmax(dim=-1))#,
        self.att = nn.Sequential(nn.Tanh(),
                                     nn.Linear(self.hidden_size + self.input_size, self.input_size, bias=False),
                                     nn.Softmax(dim=-1))
        self.tanh = nn.Tanh()
    def step(self, x, x_last_obsv, x_mean, h, mask, delta):
        g_x=self.gamma_x_l(delta)
        zero_x=torch.full_like(g_x,0)
        g_x=torch.where(g_x>0,g_x,zero_x)
        delta_x = torch.exp(g_x)

        g_h = self.gamma_h_l(delta)
        zero_h = torch.full_like(g_h, 0)
        g_h = torch.where(g_h > 0, g_h, zero_h)
        delta_h = torch.exp(g_h)
        x = mask * x + (1 - mask) * (delta_x * x_last_obsv + (1 - delta_x) * x_mean)
      
        re = torch.cat([x,h],dim=-1)
        att=self.att(re)
        x=att*x
        h = delta_h * h

        combined = torch.cat((x, h, mask), 2)
        z = F.sigmoid(self.zl(combined))
        r = F.sigmoid(self.rl(combined))
        combined_r = torch.cat((x, r * h, mask), 2)
        h_tilde = F.tanh(self.hl(combined_r))
        h = (1 - z) * h + z * h_tilde

        return h,att

    def output(self,h_long, h_cs, h_text):
        h_long_pooled=torch.mean(h_long,1)
        s=self.staic_linear(h_cs)
        t=self.text_liear(h_text)
        x=torch.cat((h_long_pooled,s,t),dim=1)
        logits = self.output_layer_2(x)
        return logits

    def forward(self, x_long, x_cs, x_text):
        atten_w_all, atten_w_all_rev = [], []
        batch_size = x_long.size(0)
        step_size = x_long.size(2)

        h = self._init_hidden(batch_size)
        x_seq = x_long[:, 0, :, :]
        x_seq_last_obs = x_long[:, 1, :, :]
        x_seq_last_obs_rev = x_long[:, 2, :, :]
        var_mask = x_long[:, 3, :, :]
        delta = x_long[:, 4, :, :]
        delta_rev = x_long[:, 5, :, :]
        seq_mask = x_long[:, 6, :, :]
        
        var_cnts = torch.sum(var_mask, axis=1, keepdim=True)
        ones = torch.ones_like(var_cnts)
        x_mean_cur = torch.sum(x_seq, axis=1, keepdim=True) / torch.where(var_cnts < 1, ones, var_cnts)  


        lambda_cur = self.tanh(var_cnts)
        x_mean = self.x_mean_g*(1-lambda_cur)+lambda_cur*x_mean_cur
        
        outputs = None
        for i in range(step_size):
            h,atten_w = self.step(x_seq[:, i:i + 1, :]\
                                     , x_seq_last_obs[:, i:i + 1, :] \
                                     , x_mean \
                                     , h \
                                     , var_mask[:, i:i + 1, :]\
                                     , delta[:, i:i + 1, :])
            if outputs is None:
                outputs = h
            else:
                outputs = torch.cat((outputs, h), 1)
            atten_w_all.append(atten_w)

    
        outputs_rev = None
        h = self._init_hidden(batch_size)
        for i in range(step_size-1,-1,-1):
            h,atten_w = self.step(x_seq[:, i:i + 1, :] \
                                        , x_seq_last_obs_rev[:, i:i + 1, :] \
                                        , x_mean \
                                        , h \
                                        , var_mask[:, i:i + 1, :] \
                                        , delta_rev[:, i:i + 1, :])

            if outputs_rev is None:
                outputs_rev = h

            else:
                outputs_rev = torch.cat((outputs_rev, h), 1)
            atten_w_all_rev.append(atten_w)
        atten_w_all = atten_w_all+atten_w_all_rev[::-1]
        
        outputs=torch.cat((outputs, outputs_rev), 2)
        outputs=outputs.permute(0,2,1)
        outputs=self.bn_bidirection(outputs)
        outputs=outputs.permute(0,2,1)
       
        logits=self.output(outputs,x_cs,x_text)

        atten_w_all=torch.cat(atten_w_all,dim=1)
        mean_att = (atten_w_all*seq_mask.repeat(1,2,1)).sum(dim=1) / (2*seq_mask.sum(dim=1))

        return logits,mean_att

    def _init_hidden(self, batch_size):
        Hidden_State = torch.zeros(batch_size, 1,self.hidden_size).to(self.device)
        return Hidden_State

      

