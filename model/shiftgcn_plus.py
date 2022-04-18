import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
from scipy.stats import norm
import scipy
from collections import OrderedDict

import sys
sys.path.append("./model/Temporal_shift/")
from cuda.shift import Shift



def conv_init(conv):
    nn.init.kaiming_normal(conv.weight, mode='fan_out')
    nn.init.constant(conv.bias, 0)

def bn_init(bn, scale):
    nn.init.constant(bn.weight, scale)
    nn.init.constant(bn.bias, 0)


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class Shift_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(Shift_tcn, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.bn = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.shift_in = Shift(channel=in_channels, stride=1, init_scale=1)
        self.shift = Shift(channel=out_channels, stride=stride, init_scale=1)
        self.stride = stride

        self.downsample = nn.Conv2d(in_channels, out_channels, 1)

        nn.init.kaiming_normal(self.downsample.weight, mode='fan_out')
        bn_init(self.bn2, 1)

    def forward(self, x):
        x = self.bn(x)
        x = self.shift_in(x)
        x = self.downsample(x)
        x = self.relu(x)
        x = self.shift(x)
        x = self.bn2(x)
        return x


class Shift_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3):
        super(Shift_gcn, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        if self.in_channels == 3:
            self.group = 25
        else:
            self.group = 25

        self.Linear_weight = nn.Parameter(torch.zeros((in_channels*25)//self.group ,(out_channels*25)//self.group,requires_grad=True,device='cuda'),requires_grad=True)
        nn.init.normal_(self.Linear_weight, 0,math.sqrt(1.0 / ((out_channels*25)//self.group)))

        self.Linear_bias = nn.Parameter(torch.zeros(1,1,out_channels,requires_grad=True,device='cuda'),requires_grad=True)
        nn.init.constant(self.Linear_bias, 0)

        self.Feature_Mask = nn.Parameter(torch.ones(1,25,in_channels, requires_grad=True,device='cuda'),requires_grad=True)
        nn.init.constant(self.Feature_Mask, 0)


        self.bn = nn.BatchNorm1d(25*out_channels)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)


        index_array = np.empty(25*in_channels).astype(np.int)
        for i in range(25):
            for j in range(in_channels):
                index_array[i*in_channels + j] = (i*in_channels + j + j*in_channels)%(in_channels*25)
        self.shift_in = nn.Parameter(torch.from_numpy(index_array),requires_grad=False)

        index_array = np.empty(25*out_channels).astype(np.int)
        for i in range(25):
            for j in range(out_channels):
                index_array[i*out_channels + j] = (i*out_channels + j - j*out_channels)%(out_channels*25)
        self.shift_out = nn.Parameter(torch.from_numpy(index_array),requires_grad=False)
        

    def forward(self, x0):
        n, c, t, v = x0.size()
        x = x0.permute(0,2,3,1).contiguous() #[n,t,v,c]

        x = x.view(n*t,v*c)
        x = torch.index_select(x, 1, self.shift_in)
        x = x.view(n*t,v,c)

        x = x * (torch.tanh(self.Feature_Mask)+1)

        x = torch.einsum('nwc,cd->nwd', (x, self.Linear_weight)).contiguous() # nt,v,c

        x = x + self.Linear_bias

        x = x.view(n*t,-1) 
        x = torch.index_select(x, 1, self.shift_out)
        x = self.bn(x)
        x = x.view(n,t,v,self.out_channels).permute(0,3,1,2) # n,c,t,v

        x = x + self.down(x0)
        x = self.relu(x)
        return x


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = Shift_gcn(in_channels, out_channels, A)
        self.tcn1 = Shift_tcn(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.tcn1(self.gcn1(x)) + self.residual(x)
        return x


class TeacherModel(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, in_channels=3):
        super(TeacherModel, self).__init__()
        A = 0

        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        self.relu = nn.ReLU()

        self.l1 = TCN_GCN_unit(3, int(64), A, residual=False)
        self.l2 = TCN_GCN_unit(int(64)+3, int(64), A)
        self.l3 = TCN_GCN_unit(int(64)+3, int(64), A)
        self.l4 = TCN_GCN_unit(int(64)+3, int(64), A)
        self.l5 = TCN_GCN_unit(int(64)+3, int(128), A, stride=2)
        self.l6 = TCN_GCN_unit(int(128)+3, int(128), A)
        self.l7 = TCN_GCN_unit(int(128)+3, int(128), A)
        self.l8 = TCN_GCN_unit(int(128)+3, int(256), A, stride=2)
        self.l9 = TCN_GCN_unit(int(256)+3, int(256), A)
        self.l10 = TCN_GCN_unit(int(256)+3, int(256), A)

        self.fc = nn.Linear(int(256), num_class)
        nn.init.normal(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)


    def forward(self, x):
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x0 = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        
        x0_d = F.avg_pool2d(x0, (2,1))
        x0_dd = F.avg_pool2d(x0_d, (2,1))

        x = self.l1(x0)
        x = self.l2(torch.cat([x,x0],1))
        x = self.l3(torch.cat([x,x0],1))
        x = self.l4(torch.cat([x,x0],1))
        x = self.l5(torch.cat([x,x0],1))
        x = self.l6(torch.cat([x,x0_d],1))
        x = self.l7(torch.cat([x,x0_d],1))
        x = self.l8(torch.cat([x,x0_d],1))
        x = self.l9(torch.cat([x,x0_dd],1))
        x = self.l10(torch.cat([x,x0_dd],1))

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)

        return self.fc(x)

    def get_bn_before_relu(self):
        bn1 = self.l4.tcn1.bn2
        bn2 = self.l7.tcn1.bn2
        bn3 = self.l10.tcn1.bn2
        return [bn1, bn2, bn3]

    def get_channel_num(self):
        return [64, 128, 256]

    def extract_feature(self, x):

        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x0 = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        
        x0_d = F.avg_pool2d(x0, (2,1))
        x0_dd = F.avg_pool2d(x0_d, (2,1))

        x = self.relu(self.l1(x0))
        x = self.relu(self.l2(torch.cat([x,x0],1)))
        x = self.relu(self.l3(torch.cat([x,x0],1)))
        x_feature1 = self.l4(torch.cat([x,x0],1))
        x = self.relu(x_feature1)
        x = self.relu(self.l5(torch.cat([x,x0],1)))
        x = self.relu(self.l6(torch.cat([x,x0_d],1)))
        x_feature2 = self.l7(torch.cat([x,x0_d],1))
        x = self.relu(x_feature2)
        x = self.relu(self.l8(torch.cat([x,x0_d],1)))
        x = self.relu(self.l9(torch.cat([x,x0_dd],1)))
        x_feature3 = self.l10(torch.cat([x,x0_dd],1))
        x = self.relu(x_feature3)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)

        return [x_feature1, x_feature2, x_feature3], self.fc(x)



class DY_SEModule(nn.Module):

    def __init__(self, channels, reduction=4):
        super(DY_SEModule, self).__init__()
        if channels==3:
            reduction=1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, 4, kernel_size=1,padding=0)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, epoch):
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x) 

        if epoch < 60:
            tao = -(30-1)/60.0 * epoch + 30
        else:
            tao = 1.0

        x = self.softmax(x/tao)
        return x


class unit_tcn_student(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn_student, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class Shift_tcn_student(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(Shift_tcn_student, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.bn = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.shift_in = Shift(channel=in_channels, stride=1, init_scale=1)
        self.shift = Shift(channel=out_channels, stride=stride, init_scale=1)
        self.stride = stride

        self.DY_att = DY_SEModule(in_channels)

        self.temporal_weight = nn.Parameter(torch.zeros(4, in_channels, out_channels,requires_grad=True,device='cuda'),requires_grad=True)
        nn.init.normal_(self.temporal_weight, 0, math.sqrt(1.0/ out_channels))

        self.temporal_bias = nn.Parameter(torch.zeros(4, out_channels, requires_grad=True,device='cuda'),requires_grad=True)
        nn.init.constant(self.temporal_bias, 0)

        bn_init(self.bn2, 1)

    def forward(self, x, epoch):
        n, c, t, v = x.size()
        x = self.bn(x)
        x = self.shift_in(x)

        DY_att_gate = self.DY_att(x, epoch).view(n,4) # n,4
        temporal_weight_fuse = torch.einsum('kcd,nk->ncd', (self.temporal_weight,DY_att_gate))
        temporal_bias_fuse = torch.einsum('kc,nk->nc', (self.temporal_bias,DY_att_gate)).contiguous()
        temporal_bias_fuse = temporal_bias_fuse.view(temporal_bias_fuse.size(0), temporal_bias_fuse.size(1), 1, 1)

        x = torch.einsum('nctv,ncd->ndtv',(x, temporal_weight_fuse))
        x = x + temporal_bias_fuse
        
        x = self.relu(x)
        x = self.shift(x)
        x = self.bn2(x)
        return x


class Shift_gcn_student(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3):
        super(Shift_gcn_student, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        if self.in_channels == 3:
            self.group = 25
        else:
            self.group = 25

        self.DY_att = DY_SEModule(in_channels)

        self.Linear_weight = nn.Parameter(torch.zeros(4, in_channels, out_channels,requires_grad=True,device='cuda'),requires_grad=True)
        nn.init.normal_(self.Linear_weight, 0,math.sqrt(1.0/ ((out_channels*25)//self.group)))

        self.Linear_bias = nn.Parameter(torch.zeros(4,out_channels,requires_grad=True,device='cuda'),requires_grad=True)
        nn.init.constant(self.Linear_bias, 0)

        self.Feature_Mask = nn.Parameter(torch.ones(1,25,in_channels, requires_grad=True,device='cuda'),requires_grad=True)
        nn.init.constant(self.Feature_Mask, 0)

        self.bn = nn.BatchNorm1d(25*out_channels)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

        index_array = np.empty(25*in_channels).astype(np.int)
        for i in range(25):
            for j in range(in_channels):
                index_array[i*in_channels + j] = (i*in_channels + j + j*in_channels)%(in_channels*25)
        self.shift_in = nn.Parameter(torch.from_numpy(index_array),requires_grad=False)

        index_array = np.empty(25*out_channels).astype(np.int)
        for i in range(25):
            for j in range(out_channels):
                index_array[i*out_channels + j] = (i*out_channels + j - j*out_channels)%(out_channels*25)
        self.shift_out = nn.Parameter(torch.from_numpy(index_array),requires_grad=False)
        
    def forward(self, x0, epoch):
        n, c, t, v = x0.size()

        DY_att_gate = self.DY_att(x0, epoch).view(n,4) # nt,4
        Linear_weight_fuse = torch.einsum('kcd,nk->ncd', (self.Linear_weight,DY_att_gate))
        Linear_bias_fuse = torch.einsum('kc,nk->nc', (self.Linear_bias,DY_att_gate)).contiguous().view(n,1,1,-1)

        x = x0.permute(0,2,3,1).contiguous() #[n,t,v,c]

        x = x.view(n*t,v*c)
        x = torch.index_select(x, 1, self.shift_in)
        x = x.view(n*t,v,c)

        x = x * (torch.tanh(self.Feature_Mask)+1)

        x = x.view(n, t, v, c)
        x = torch.einsum('ntvc,ncd->ntvd', (x, Linear_weight_fuse)).contiguous() # nt,v,c
        x = x + Linear_bias_fuse

        x = x.view(n*t,-1) 
        x = torch.index_select(x, 1, self.shift_out)
        x = self.bn(x)
        x = x.view(n,t,v,self.out_channels).permute(0,3,1,2) # n,c,t,v

        x = x + self.down(x0)
        x = self.relu(x)
        return x




class TCN_GCN_unit_student(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super(TCN_GCN_unit_student, self).__init__()
        self.gcn1 = Shift_gcn_student(in_channels, out_channels, A)
        self.tcn1 = Shift_tcn_student(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn_student(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x, epoch):
        x = self.tcn1(self.gcn1(x, epoch), epoch) + self.residual(x)
        return self.relu(x)



class StudentModel(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, in_channels=3):
        super(StudentModel, self).__init__()

        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        A = 0

        self.relu = nn.ReLU()
        C_mul = 2*2

        self.l1 = TCN_GCN_unit_student(3, int(8*C_mul), A, residual=False)
        self.l2 = TCN_GCN_unit_student(int(8*C_mul)+3, int(8*C_mul), A)
        self.l3 = TCN_GCN_unit_student(int(8*C_mul)+3, int(8*C_mul), A)
        self.l5 = TCN_GCN_unit_student(int(8*C_mul)+3, int(16*C_mul), A, stride=2)
        self.l6 = TCN_GCN_unit_student(int(16*C_mul)+3, int(16*C_mul), A)
        self.l8 = TCN_GCN_unit_student(int(16*C_mul)+3, int(32*C_mul), A, stride=2)
        self.l9 = TCN_GCN_unit_student(int(32*C_mul)+3, int(32*C_mul), A)

        self.fc = nn.Linear(int(32*C_mul), num_class)
        nn.init.normal(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)

    def forward(self, x, epoch):
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x0 = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        
        x0_d = F.avg_pool2d(x0, (2,1))
        x0_dd = F.avg_pool2d(x0_d, (2,1))

        x = self.l1(x0, epoch)
        x = self.l2(torch.cat([x,x0],1), epoch)
        x = self.l3(torch.cat([x,x0],1), epoch)
        x = self.l5(torch.cat([x,x0],1), epoch)
        x = self.l6(torch.cat([x,x0_d],1), epoch)
        x = self.l8(torch.cat([x,x0_d],1), epoch)
        x = self.l9(torch.cat([x,x0_dd],1), epoch)
        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)

        return self.fc(x)

    def get_bn_before_relu(self):
        bn1 = self.l3.tcn1.bn2
        bn2 = self.l6.tcn1.bn2
        bn3 = self.l9.tcn1.bn2
        return [bn1, bn2, bn3]

    def get_channel_num(self):
        return [32,64,128]

    def extract_feature(self, x, epoch):

        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x0 = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        
        x0_d = F.avg_pool2d(x0, (2,1))
        x0_dd = F.avg_pool2d(x0_d, (2,1))

        x = self.relu(self.l1(x0, epoch))
        x = self.relu(self.l2(torch.cat([x,x0],1), epoch))
        x_feature1 = self.l3(torch.cat([x,x0],1), epoch)
        x = self.relu(x_feature1)
        x = self.relu(self.l5(torch.cat([x,x0],1), epoch))
        x_feature2 = self.l6(torch.cat([x,x0_d],1), epoch)
        x = self.relu(x_feature2)
        x = self.relu(self.l8(torch.cat([x,x0_d],1), epoch))
        x_feature3 = self.l9(torch.cat([x,x0_dd],1), epoch)
        x = self.relu(x_feature3)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)

        return [x_feature1, x_feature2, x_feature3], self.fc(x)



def distillation_loss(source, target, margin):
    loss = ((source - margin)**2 * ((source > margin) & (target <= margin)).float() +
            (source - target)**2 * ((source > target) & (target > margin) & (target <= 0)).float() +
            (source - target)**2 * (target > 0).float())
    return torch.abs(loss).sum()

def build_feature_connector(t_channel, s_channel):
    C = [nn.Conv2d(s_channel, t_channel, kernel_size=1, stride=1, padding=0, bias=False),
         nn.BatchNorm2d(t_channel)]

    for m in C:
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    return nn.Sequential(*C)

def get_margin_from_BN(bn):
    margin = []
    std = bn.weight.data
    mean = bn.bias.data
    for (s, m) in zip(std, mean):
        s = abs(s.item())
        m = m.item()
        if norm.cdf(-m / s) > 0.001:
            margin.append(- s * math.exp(- (m / s) ** 2 / 2) / math.sqrt(2 * math.pi) / norm.cdf(-m / s) + m)
        else:
            margin.append(-3 * s)

    return torch.FloatTensor(margin).to(std.device)


class Model(nn.Module):
    def __init__(self, teacher_model=None, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3):
        super(Model, self).__init__()

        t_net = TeacherModel()
        s_net = StudentModel()

        t_channels = t_net.get_channel_num()
        s_channels = s_net.get_channel_num()

        self.Connectors = nn.ModuleList([build_feature_connector(t, s) for t, s in zip(t_channels, s_channels)])

        teacher_bns = t_net.get_bn_before_relu()
        margins = [get_margin_from_BN(bn) for bn in teacher_bns]
        for i, margin in enumerate(margins):
            self.register_buffer('margin%d' % (i+1), margin.unsqueeze(1).unsqueeze(2).unsqueeze(0).detach())

        self.t_net = t_net
        self.s_net = s_net

        weights = torch.load(teacher_model)
        output_device = 0
        weights = OrderedDict(
            [[k.split('module.')[-1],
              v.cuda(output_device)] for k, v in weights.items()])

        try:
            self.t_net.load_state_dict(weights)
        except:
            state = self.model.state_dict()
            diff = list(set(state.keys()).difference(set(weights.keys())))
            print('Can not find these weights:')
            for d in diff:
                print('  ' + d)
            state.update(weights)
            self.t_net.load_state_dict(state)


    def forward(self, x, epoch):

        t_feats, t_out = self.t_net.extract_feature(x)
        s_feats, s_out = self.s_net.extract_feature(x, epoch)
        feat_num = len(t_feats)

        loss_distill = 0
        for i in range(feat_num):
            s_feats[i] = self.Connectors[i](s_feats[i])
            loss_distill += distillation_loss(s_feats[i], t_feats[i].detach(), getattr(self, 'margin%d' % (i+1))) \
                            / 2 ** (feat_num - i - 1)

        return s_out, loss_distill