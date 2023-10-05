import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pickle

_weights_dict = dict()

def load_weights(weight_file):
    if weight_file == None:
        return

    try:
        weights_dict = np.load(weight_file, allow_pickle=True).item()
    except:
        weights_dict = np.load(weight_file, allow_pickle=True, encoding='bytes').item()

    return weights_dict

class KitModel(nn.Module):


    def __init__(self, weight_file):
        super(KitModel, self).__init__()
        global _weights_dict
        _weights_dict = load_weights(weight_file)

        # get other saved weights
        names = ["pc1b", "fc4W_pose", "fc3b", "fc4b", "fc5W", "fc5b", "fc3W", "fc4W_im", "conv1_2b", "conv2_1b", "conv2_2b", "conv2_2W", "conv1_1b", "pc1W"]
        global _other_weights
        with open('variables.pkl', 'rb') as file:
            variables_to_load = pickle.load(file)
        _other_weights = {name: torch.from_numpy(var) for name, var in zip(names, variables_to_load)}

        # make sizes compatible with PyTorch
        _other_weights["conv1_1b"] = _other_weights["conv1_1b"].unsqueeze(0).unsqueeze(2).unsqueeze(3)
        _other_weights["conv1_2b"] = _other_weights["conv1_2b"].unsqueeze(0).unsqueeze(2).unsqueeze(3)
        _other_weights["conv2_1b"] = _other_weights["conv2_1b"].unsqueeze(0).unsqueeze(2).unsqueeze(3)
        _other_weights["conv2_2b"] = _other_weights["conv2_2b"].unsqueeze(0).unsqueeze(2).unsqueeze(3)

        self.Conv2D = self.__conv(2, name='Conv2D', in_channels=1, out_channels=64, kernel_size=(7, 7), stride=(1, 1), groups=1, bias=True, padding='same')
        self.Conv2D_1 = self.__conv(2, name='Conv2D_1', in_channels=64, out_channels=64, kernel_size=(5, 5), stride=(1, 1), groups=1, bias=True, padding='same')
        self.Conv2D_2 = self.__conv(2, name='Conv2D_2', in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True, padding='same')
        self.Conv2D_3 = self.__conv(2, name='Conv2D_3', in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True, padding='same')
        self.pad = nn.ConstantPad2d(1, 0)

        print("initalized!\n")

    def forward(self, x1, x2):
        MatMul_1        = torch.matmul(x1, _other_weights["pc1W"])      # x1 - step 0; input: 64 x 1
        add_5           = MatMul_1 + _other_weights["pc1b"]             # x1 - step 1; MatMul_1 + pc1b/read
        Relu_5          = F.relu(add_5)                                 # x1 - step 2
        # MirrorPad       = self.pad(x2)                                  # x2 - step 0; input: 64 x 32 x 32 x 1
        # MirrorPad_1     = self.pad(MirrorPad)
        # MirrorPad_2     = self.pad(MirrorPad_1)
        Conv2D          = self.Conv2D(x2)                                    # x2 - step 1
        MatMul_3        = torch.matmul(Relu_5, _other_weights["fc4W_pose"])  # x1 - step 3; Relu_5 * fc4W_pose/read
        Relu            = F.relu(Conv2D)                                        # x2 - step 2
        MaxPool, MaxPool_idx = F.max_pool2d(Relu, kernel_size=(1, 1), stride=(1, 1), padding=0, ceil_mode=False, return_indices=True)   # x2 - step 3
        # MirrorPad_3     = self.pad(MaxPool)
        # MirrorPad_4     = self.pad(MirrorPad_3)
        Conv2D_1        = self.Conv2D_1(MaxPool)
        Relu_1          = F.relu(Conv2D_1)
        LRN             = F.local_response_norm(Relu_1, size=5, alpha=9.999999747378752e-05, beta=0.75, k=1.0)  # x2 - step 9
        MaxPool_1, MaxPool_1_idx = F.max_pool2d(LRN, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False, return_indices=True)    # x2 - step 9
        # MirrorPad_5     = self.pad(MaxPool_1)
        Conv2D_2        = self.Conv2D_2(MaxPool_1)
        Relu_2          = F.relu(Conv2D_2)
        MaxPool_2, MaxPool_2_idx = F.max_pool2d(Relu_2, kernel_size=(1, 1), stride=(1, 1), padding=0, ceil_mode=False, return_indices=True)
        # MirrorPad_6     = self.pad(MaxPool_2)
        Conv2D_3        = self.Conv2D_3(MaxPool_2)
        Relu_3          = F.relu(Conv2D_3)
        LRN_1           = F.local_response_norm(Relu_3, size=5, alpha=9.999999747378752e-05, beta=0.75, k=1.0)
        MaxPool_3, MaxPool_3_idx = F.max_pool2d(LRN_1, kernel_size=(1, 1), stride=(1, 1), padding=0, ceil_mode=False, return_indices=True)
        MaxPool_3 = torch.permute(MaxPool_3, (0,2,3,1))
        Reshape_3       = torch.reshape(input = MaxPool_3, shape = (64,16384))
        MatMul          = torch.matmul(Reshape_3, _other_weights["fc3W"])       # Reshape_3 * fc3W/read
        add_4           = MatMul + _other_weights["fc3b"]                       # MatMul + fc3b/read
        Relu_4          = F.relu(add_4)
        temp = Relu_4.detach().numpy()
        MatMul_2        = torch.matmul(Relu_4, _other_weights["fc4W_im"])       # Relu_4 * fc4W_im/read
        add_6           = MatMul_2 + MatMul_3                                   # combines x1 and x2
        add_7           = add_6 + _other_weights["fc4b"]                        # add_6 + fc4b/read
        Relu_6          = F.relu(add_7)
        MatMul_4        = torch.matmul(Relu_6, _other_weights["fc5W"])          # Relu_6 * fc5W/read
        add_8           = MatMul_4 + _other_weights["fc5b"]                     # MatMul_4 + fc5b/read
        Softmax         = F.softmax(add_8, dim=1)
        return Softmax


    @staticmethod
    def __conv(dim, name, **kwargs):
        if   dim == 1:  layer = nn.Conv1d(**kwargs)
        elif dim == 2:  layer = nn.Conv2d(**kwargs)
        elif dim == 3:  layer = nn.Conv3d(**kwargs)
        else:           raise NotImplementedError()

        temp = torch.permute(torch.from_numpy(_weights_dict[name]['weights']), (2,3,1,0))
        # print("\nname:", name)
        # print("weights:", torch.permute(torch.from_numpy(_weights_dict[name]['weights']), (2,3,1,0)).shape)
        # print(torch.permute(torch.from_numpy(_weights_dict[name]['weights']), (2,3,1,0)))
        # print("conv1_1b:", _other_weights["conv1_1b"])

        layer.state_dict()['weight'].copy_(torch.from_numpy(_weights_dict[name]['weights']))
        if 'bias' in _weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(_weights_dict[name]['bias']))
            # print("bias:", torch.from_numpy(_weights_dict[name]['bias']))
        return layer

    @staticmethod
    def __dense(name, **kwargs):
        layer = nn.Linear(**kwargs)
        layer.state_dict()['weight'].copy_(torch.from_numpy(_weights_dict[name]['weights']))
        if 'bias' in _weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(_weights_dict[name]['bias']))
        return layer
