import torch
import imp
import numpy as np
import tensorflow as tf
from gqcnn_pytorch2 import KitModel

print("torch version:", torch.__version__)

str = """
# TENSORFLOW MODEL
saved_model_dir = "/gqcnn/models2/saved_model/"

with tf.Session(graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], saved_model_dir)

    # Now you have access to all the variables in the graph
    for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
        print("Variable: ", var)  # This will print the variable's name and shape.
        # print("Value: ", sess.run(var))
"""

# PYTORCH MODEL
# initialize model
model2 = KitModel("573931b36e8e4fdab6218f6c598e100d.npy")
model2.eval()

# test prediction
x1 = torch.from_numpy(np.load('input_pose.npy')).float()  # pose - 64x1
x2 = torch.from_numpy(np.load('input_im.npy')).float().permute(0,3,1,2)    # image - 64x32x32x1
# x1 = torch.from_numpy(np.load('normalized_pose.npy')).float()
# x2 = torch.from_numpy(np.load('normalized_im.npy')).float().permute(0,3,1,2)
# print("input im:", x2)
# print("input pose:", x1)

# from scipy.io import savemat
# savemat('inputs_pytorch.mat', {"depth": np.load('input_im.npy'), "angle": np.load('input_pose.npy')})

output_arr = model2(x1, x2)
# output_arr = output_arr[:, -1]
# output_arr = output_arr.detach().numpy()
print("output:", output_arr.shape)
print(output_arr)
