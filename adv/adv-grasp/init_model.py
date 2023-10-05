import torch
import numpy as np
from gqcnn_pytorch import KitModel

# initialize model
model2 = KitModel("weights.npy")
model2.eval()

# test prediction
x1 = torch.from_numpy(np.load('data/pose_tensor1_raw.npy')).float()  # pose - 64x1
x2 = torch.from_numpy(np.load('data/image_tensor1_raw.npy')).float().permute(0,3,1,2)    # image - 64x32x32x1

# from scipy.io import savemat
# savemat('inputs_pytorch.mat', {"depth": np.load('input_im.npy'), "angle": np.load('input_pose.npy')})

output_arr = model2(x1[0], x2[0].unsqueeze(0))
output2 = model2(x1, x2)
print("output:", output_arr.shape)
print(output_arr)
print("output2:", output2.shape)
print(output2)

