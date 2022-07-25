from __future__ import division, print_function

import math
import os
import pdb
import pickle
import re
from random import randrange
from shutil import which

import cv2
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchstain
from PIL import Image
from sympy import im
from torch.utils.data import DataLoader, Dataset, sampler
from torchvision import models, transforms, utils


def eval_transforms(pretrained=False):
	if pretrained:
		mean = (0.485, 0.456, 0.406)
		std = (0.229, 0.224, 0.225)

	else:
		mean = (0.5,0.5,0.5)
		std = (0.5,0.5,0.5)

	trnsfrms_val = transforms.Compose(
					[
					 transforms.ToTensor(),
					 transforms.Normalize(mean,std)
					]
				)

	return trnsfrms_val

class Whole_Slide_Bag_FP(Dataset):
	def __init__(self,
		file_path,
		wsi,
		pretrained=False,
		custom_transforms=None,
		custom_downsample=1,
		target_patch_size=-1,
		norm=False,
		target_norm_path=None
		):
		"""
		Args:
			file_path (string): Path to the .h5 file containing patched data.
			pretrained (bool): Use ImageNet transforms
			custom_transforms (callable, optional): Optional transform to be applied on a sample
			custom_downsample (int): Custom defined downscale factor (overruled by target_patch_size)
			target_patch_size (int): Custom defined image size before embedding
			norm (boolean): color normalization
			target_norm_path (string): path of target patch for color normalization
		"""
		self.pretrained=pretrained
		self.wsi = wsi
		if not custom_transforms:
			self.roi_transforms = eval_transforms(pretrained=pretrained)
		else:
			self.roi_transforms = custom_transforms

		self.file_path = file_path

		with h5py.File(self.file_path, "r") as f:
			dset = f['coords']
			self.patch_level = f['coords'].attrs['patch_level']
			self.patch_size = f['coords'].attrs['patch_size']
			self.length = len(dset)
			if target_patch_size > 0:
				self.target_patch_size = (target_patch_size, ) * 2
			elif custom_downsample > 1:
				self.target_patch_size = (self.patch_size // custom_downsample, ) * 2
			else:
				self.target_patch_size = None
		self.summary()

		self.normset = norm
		if self.normset:
			target_patch = Image.open(target_norm_path).convert('RGB')
			target_patch = target_patch.resize((256,256))
			# target_patch = cv2.resize(cv2.cvtColor(cv2.imread(target_norm_path), cv2.COLOR_BGR2RGB),(256,256))
			# target_patch = target_patch.transpose(2,0,1)
			T = transforms.Compose([
					 transforms.ToTensor(),
					 transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
					])
			normalizer = torchstain.MacenkoNormalizer(backend='torch')
			normalizer.fit(T(target_patch))
			self.normalizer = normalizer
			
	def __len__(self):
		return self.length

	def summary(self):
		hdf5_file = h5py.File(self.file_path, "r")
		dset = hdf5_file['coords']
		for name, value in dset.attrs.items():
			print(name, value)

		print('\nfeature extraction settings')
		print('target patch size: ', self.target_patch_size)
		print('pretrained: ', self.pretrained)
		print('transformations: ', self.roi_transforms)

	def __getitem__(self, idx):
		with h5py.File(self.file_path,'r') as hdf5_file:
			coord = hdf5_file['coords'][idx]
		img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')

		if self.target_patch_size is not None:
			img = img.resize(self.target_patch_size)
		img = self.roi_transforms(img)
		if self.normset:
			try:
				normimg, _, _ = self.normalizer.normalize(I=img, stains=True)
				normimg = normimg.permute(2,0,1).unsqueeze(0)
				normimg = normimg.float()
				# print('dtype of norm_img:{}'.format(normimg.dtype))
				return normimg, coord
			except:
				img = img.unsqueeze(0)
				# print('dtype of img:{}'.format(img.dtype))
				return img, coord
		else:
			img = img.unsqueeze(0)
			# print('dtype of img:{}'.format(img.dtype))
			return img, coord

class Dataset_All_Bags(Dataset):

	def __init__(self, csv_path):
		all_df = pd.read_csv(csv_path)
		processed_df = all_df.loc[all_df['status']=='processed']
		self.df = processed_df.reset_index(drop=True)
	
	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		return self.df['slide_id'][idx]




