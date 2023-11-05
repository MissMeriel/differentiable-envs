# This file contains the code for the select_grasp.py extract_tensors method in a numpy implementation, a reimplementation
    # of code from the Berkeley AUTOLAB gqcnn library grasp_to_tensors method. It also includes debugging visualizations comparing
    # to the PyTorch implementaiton of the extract_tensors method in select_grasp.py, but this is non-functional in this file.

import numpy as np
import matplotlib.pyplot as plt
import skimage
import cv2
import PIL.Image as PImage
import torch

def transform_im(d_im, cx, cy, translation, angle):
	"""
	Helper method for extract_tensors to rotate and translate a depth image based on a translation
	  and grasp angle using cv2.warpAffine method. Based on code from autolab_core.Image.transform 
	  method.
	Parameters
	----------
	d_im: np.ndarray
		Depth image to be rotated and translated
	cx: float
		X-coordinate of the center of the depth image 
	cy: float
		Y-coordinate of the center of the depth image 
	translation: np.ndarray
		2x1 vector that I don't really understand right now. 
	angle: float
		Grasp angle in randians used to rotate image to align with middle row of pixels
	Returns
	-------
	numpy.ndarray
		Depth image with affine transformation applied.	
	""" 
		
	theta = np.rad2deg(angle)

	# define matrix for translation 
	trans_mat = np.array(
		[[1, 0, translation[1, 0]], [0, 1, translation[0, 0]]], dtype=np.float32 
	)

	# define matrix for rotation 
	rot_mat = cv2.getRotationMatrix2D(
		(cx, cy), theta, 1
	)

	# print("OpenCV rotation matrix:", rot_mat)

	# convert to 3x3 matrices and combine transformations w/ matrix multiplication then revert to 2x3 
	trans_mat_aff = np.r_[trans_mat, [[0, 0, 1]]]
	rot_mat_aff = np.r_[rot_mat, [[0, 0, 1]]]
	full_mat = rot_mat_aff.dot(trans_mat_aff)
	full_mat = full_mat[:2, :]

	# apply transformation with cv2
	image = cv2.warpAffine(
		d_im,
		full_mat,
		(d_im.shape[1], d_im.shape[0]),
		flags=cv2.INTER_NEAREST
	)

	return image 

def crop_im(d_im, height, width):
	"""
	Helper method for extract_tensors to crop a depth image to specified height and width using
	  PIL. Based on code from autolab_core.Image.crop method. 
	Parameters
	----------
	d_im: np.ndarray
		Depth image to be cropped
	height: int
		Height for cropped image
	width: int
		Width for cropped image
	Returns
	-------
	np.ndarray
		Cropped depth image
	"""

	center_i = d_im.shape[0] / 2
	center_j = d_im.shape[1] / 2

	# crop using PIL
	start_row = int(np.floor(center_i - float(height) / 2))
	end_row = int(np.floor(center_i + float(height) / 2))
	start_col = int(np.floor(center_j - float(width) / 2))
	end_col = int(np.floor(center_j + float(width) / 2))

	im = PImage.fromarray(d_im)
	cropped_pil_im = im.crop(
		(
			start_col,
			start_row,
			end_col,
			end_row,
		)
	)
	crop_im = np.array(cropped_pil_im)
	
	return crop_im

def extract_tensors(d_im, grasp):
    """
	Use grasp information and depth image to get image and pose tensors in form of GQCNN input
	Parameters
	----------
	d_im: numpy.ndarray
		Numpy array depth image of object being grasped
	grasp: List containing grasp information: [(int, int): grasp_center, float: grasp_angle, float:
	   grasp_depth]
		grasp_center: Grasp center used to center depth image
		grasp_angle: Angle used to rotate depth image to align with middle row of pixels
		grasp_depth: Depth of grasp to create pose array
	Returns
	-------
	torch.tensor: pose_tensor, torch.tensor: image_tensor
		pose_tensor: 1 x 1 tensor of grasp pose
		image_tensor: 1 x 1 x 32 x 32 tensor of depth image processed for grasp
	"""

	# check type of input_dim
    if isinstance(d_im, np.ndarray):
        torch_dim = torch.tensor(d_im, dtype=torch.float32).permute(2, 0, 1)

	# construct pose tensor from grasp depth
    pose_tensor = torch.zeros([1, 1])
    pose_tensor[0] = grasp[2]

	# process depth image wrt grasp (steps 1-3) 

	# 1 - resize image tensor
    np_shape = np.asarray(d_im.shape).astype(np.float32)
    np_shape[0:2] *= (1/3)		# using 1/3 based on gqcnn library - may need to change depending on input
    output_shape = tuple(np_shape.astype(int))
    image_tensor = skimage.transform.resize(d_im.astype(np.float64), output_shape, order=1, anti_aliasing=False, mode="constant")

	# 2 - translate wrt to grasp angle and grasp center 
    dim_center_x = d_im.shape[1] // 2
    dim_center_y = d_im.shape[0] // 2
    translation = (1/3) * np.array([
        [dim_center_y - grasp[0][1]],
        [dim_center_x - grasp[0][0]]
    ])
    
    new_cx = image_tensor.shape[1] // 2
    new_cy = image_tensor.shape[0] // 2
    dim_rotated = transform_im(image_tensor.squeeze(), new_cx, new_cy, translation, grasp[1]) 
    
    """
	# CHECK PYTORCH EQUIVALENCE - nonfunctional here
	test = torch.tensor(dim_rotated)
	diff = test - torch_translated
	print("\ncurrent translation:", test.shape, torch.max(test).item(), torch.min(test).item(), "\n", test)
	print("\npytorch translation:", torch_translated.shape, torch.max(torch_translated).item(), torch.min(torch_translated).item(), "\n", torch_translated)
	print("\ndiff:", torch.max(diff).item(), torch.min(diff).item())
	print("\nnon-zero elements:", torch.count_nonzero(diff).item())

	torch_np = torch_translated.permute(1,2,0).numpy()
	diff_np = diff.permute(1,2,0).numpy() 	
	import matplotlib.pyplot as plt
	_, arr = plt.subplots(1, 3)
	arr[0].imshow(torch_np)
	arr[1].imshow(dim_rotated)	
	arr[2].imshow(diff_np)
	plt.show()
    """
    
    # 3 - crop image to size (32, 32)
    im_cropped = crop_im(dim_rotated, 32, 32)
    image_tensor = torch.from_numpy(im_cropped).float().unsqueeze(0).unsqueeze(0)
    
    """
	# CHECK PYTORCH EQUIVALENCE - nonfunctional here
	t = torch_cropped.squeeze(0)
	test = torch.tensor(im_cropped)
	diff = test - torch_cropped
	print("original:", test.shape, torch.max(test).item(), torch.min(test).item())
	print("pytorch:", t.shape, torch.max(t).item(), torch.min(t).item())
	print("diff:", torch.max(diff).item(), torch.min(diff).item(), diff)

	t = torch_cropped.squeeze(0).numpy()
	_, arr = plt.subplots(1, 3)
	arr[0].imshow(t)
	arr[1].imshow(im_cropped.squeeze())
	arr[2].imshow(diff.squeeze(0).permute(1,2,0).numpy())
	plt.show()
    """
    
    return pose_tensor, image_tensor  
	

if __name__ == "__main__":
    
    depth0 = np.load("/home/hmitchell/pytorch3d/dex_shared_dir/depth_0.npy")
    grasp = [(416, 286), -2.896613990462929, 0.607433762324266]

    pose, image = extract_tensors(depth0, grasp)
    print("pose tensor:\n", pose)
    print("\nimage tensor:\n", image)