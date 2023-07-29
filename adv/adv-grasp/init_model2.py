import torch
import imp
import os
import numpy as np
import tensorflow as tf
from gqcnn_pytorch2 import KitModel

# initialize model
model2 = KitModel("573931b36e8e4fdab6218f6c598e100d.npy")
model2.eval()

# test prediction
x1 = torch.from_numpy(np.load('input_pose.npy')).float()  # pose - 64x1
x2 = torch.from_numpy(np.load('input_im.npy')).float().permute(0,3,1,2)    # image - 64x32x32x1

# from scipy.io import savemat
# savemat('inputs_pytorch.mat', {"depth": np.load('input_im.npy'), "angle": np.load('input_pose.npy')})

output_arr = model2(y1[0], y2[0].unsqueeze(0))
output2 = model2(y1, y2)
print("output:", output_arr.shape)
print(output_arr)
print("output2:", output2.shape)
print(output2)

