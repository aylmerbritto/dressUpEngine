from PIL import Image
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
sys.path.append('/home/arexhari/aylmer843/PF-AFN/PF-AFN_test')
from options.test_options import TestOptions
from data.base_dataset import BaseDataset, get_params, get_transform
from models.networks import ResUnetGenerator, load_checkpoint
from models.afwm import AFWM

from clothIndex import getCloth






class dressUpInference():
	def __init__(self):
		self.opt = TestOptions().parse()
		opt = TestOptions().parse()
		self.warp_model = AFWM(opt, 3)
		self.warp_model.eval()
		self.warp_model.cuda()
		self.gen_model = ResUnetGenerator(7, 4, 5, ngf=64, norm_layer=nn.BatchNorm2d)
		self.gen_model.eval()
		self.gen_model.cuda() 
		load_checkpoint(self.gen_model, opt.gen_checkpoint)
		load_checkpoint(self.warp_model, opt.warp_checkpoint)
		self.start_epoch, self.epoch_iter = 1, 0
		self.total_steps = (self.start_epoch-1) + self.epoch_iter
		self.step = 0
		self.step_per_batch = 1 / self.opt.batchSize

		self.currentClothId = None
		self.C_tensor = None
		self.E_tensor = None
		
	def infer(self, data):
		real_image = data['image']
		clothes = data['clothes']
		##edge is extracted from the clothes image with the built-in function in python
		edge = data['edge']
		edge_tmp = edge.clone()
	
		edge = torch.FloatTensor((edge_tmp.detach().numpy() > 0.5).astype(np.int))
		
		clothes = clothes * edge    
		real_image = real_image.reshape(1,real_image.shape[0], real_image.shape[1], real_image.shape[2])
		clothes = clothes.reshape(1, clothes.shape[0], clothes.shape[1], clothes.shape[2] )
		edge = edge.reshape(1, edge.shape[0], edge.shape[1], edge.shape[2] )
		real_image = real_image.cuda()
		clothes = clothes.cuda()
		edge =edge.cuda()
		flow_out = self.warp_model(real_image, clothes)
		warped_cloth, last_flow, = flow_out
		warped_edge = F.grid_sample(edge, last_flow.permute(0, 2, 3, 1),
						  mode='bilinear', padding_mode='zeros')

		gen_inputs = torch.cat([real_image, warped_cloth, warped_edge], 1)
		gen_outputs = self.gen_model(gen_inputs)
		p_rendered, m_composite = torch.split(gen_outputs, [3, 1], 1)
		p_rendered = torch.tanh(p_rendered)
		m_composite = torch.sigmoid(m_composite)
		m_composite = m_composite * warped_edge
		p_tryon = warped_cloth * m_composite + p_rendered * (1 - m_composite)
		# path = 'results/' + self.opt.name
		# os.makedirs(path, exist_ok=True)
		# sub_path = path + '/PFAFN'
		# os.makedirs(sub_path,exist_ok=True)
		# a = real_image.float().cuda()
		# b= clothes.cuda()
		# c = p_tryon
		# step = 0
		p_tryon = p_tryon.squeeze()
		cv_img = (p_tryon.permute(1,2,0).detach().cpu().numpy()+1)/2
		# cv_img = cv2.rotate(cv_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
		# print(cv_img.shape)
		# combine = torch.cat([a[0],b[0],c[0]], 2).squeeze()
		# cv_img=(combine.permute(1,2,0).detach().cpu().numpy()+1)/2
		# print(cv_img.shape)
		# bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
		# cv2.imwrite(sub_path+'/'+str(step)+'.jpg',bgr)
		rgb=(cv_img*255).astype(np.uint8)
		return rgb
	
	def runInference(self, input_frame, clothId):
		I = Image.fromarray(input_frame.astype('uint8'),'RGB')
		params = get_params(self.opt, I.size)
		transform = get_transform(self.opt, params)	
		I_tensor = transform(I)
		if clothId !=self.currentClothId:
			C_path, E_path = getCloth(clothId)
			transform_E = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
			C = Image.open(C_path).convert('RGB')
			self.C_tensor = transform(C)

			E = Image.open(E_path).convert('L')
			self.E_tensor = transform_E(E)
			self.currentClothId = clothId

		data = { 'image': I_tensor,'clothes': self.C_tensor, 'edge': self.E_tensor}

		return self.infer(data)

	def testInference(self,I_path, C_path, E_path,O_PATH):
		# I_path = '/home/arexhari/aylmer843/PF-AFN/PF-AFN_test/dataset/test_img/model5.jpg'
		# C_path = '/home/arexhari/aylmer843/PF-AFN/PF-AFN_test/dataset/test_clothes/019119_1.jpg'
		# E_path = '/home/arexhari/aylmer843/PF-AFN/PF-AFN_test/dataset/test_edge/019119_1.jpg'
		I = Image.open(I_path).convert('RGB')
		input_frame = I
		params = get_params(self.opt, I.size)
		transform = get_transform(self.opt, params)
		transform_E = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)

		I_tensor = transform(I)

		C = Image.open(C_path).convert('RGB')
		C_tensor = transform(C)

		E = Image.open(E_path).convert('L')
		E_tensor = transform_E(E)

		self.data = { 'image': I_tensor,'clothes': C_tensor, 'edge': E_tensor}
		I = input_frame.convert('RGB')
		params = get_params(self.opt, I.size)
		transform = get_transform(self.opt, params)
		transform_E = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)

		I_tensor = transform(I)

		C = Image.open(C_path).convert('RGB')
		C_tensor = transform(C)

		E = Image.open(E_path).convert('L')
		E_tensor = transform_E(E)

		data = { 'image': I_tensor,'clothes': C_tensor, 'edge': E_tensor}

		result  = self.infer(data)
		result=cv2.cvtColor(result,cv2.COLOR_RGB2BGR)
		# result = cv2.rotate(result, cv2.ROTATE_90_CLOCKWISE)
		cv2.imwrite(O_PATH,result)
		return True
