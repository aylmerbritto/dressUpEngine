import time
from options.test_options import TestOptions
from data.base_dataset import BaseDataset, get_params, get_transform
from PIL import Image
from models.networks import ResUnetGenerator, load_checkpoint
from models.afwm import AFWM
import torch.nn as nn
import os
import numpy as np
import torch
import cv2
import torch.nn.functional as F

#python test.py --name demo --resize_or_crop None --batchSize 1 --gpu_ids 0

class dressUpInference():
    def __init__(self):
        self.opt = TestOptions().parse()

        self.start_epoch, self.epoch_iter = 1, 0

        self.warp_model = AFWM(self.opt, 3)
        print(self.warp_model)
        self.warp_model.eval()
        self.warp_model.cuda()
        load_checkpoint(self.warp_model, self.opt.warp_checkpoint)

        self.gen_model = ResUnetGenerator(7, 4, 5, ngf=64, norm_layer=nn.BatchNorm2d)
        print(self.gen_model)
        self.gen_model.eval()
        self.gen_model.cuda()
        load_checkpoint(self.gen_model, self.opt.gen_checkpoint)

        self.total_steps = (self.start_epoch-1) + self.epoch_iter
        self.step = 0
        self.step_per_batch = 1 / self.opt.batchSize
    
    def infer(self, data):
        real_image = data['image']
        clothes = data['clothes']
        ##edge is extracted from the clothes image with the built-in function in python
        edge = data['edge']
        edge = torch.FloatTensor((edge.detach().numpy() > 0.5).astype(np.int))
        
        clothes = clothes * edge    
        real_image = real_image.reshape(1,real_image.shape[0], real_image.shape[1], real_image.shape[2])
        clothes = clothes.reshape(1, clothes.shape[0], clothes.shape[1], clothes.shape[2] )
        edge = edge.reshape(1, edge.shape[0], edge.shape[1], edge.shape[2] )
        flow_out = self.warp_model(real_image.cuda(), clothes.cuda())
        warped_cloth, last_flow, = flow_out
        warped_edge = F.grid_sample(edge.cuda(), last_flow.permute(0, 2, 3, 1),
                          mode='bilinear', padding_mode='zeros')

        gen_inputs = torch.cat([real_image.cuda(), warped_cloth, warped_edge], 1)
        gen_outputs = self.gen_model(gen_inputs)
        p_rendered, m_composite = torch.split(gen_outputs, [3, 1], 1)
        p_rendered = torch.tanh(p_rendered)
        m_composite = torch.sigmoid(m_composite)
        m_composite = m_composite * warped_edge
        p_tryon = warped_cloth * m_composite + p_rendered * (1 - m_composite)
        path = 'results/' + self.opt.name
        os.makedirs(path, exist_ok=True)
        sub_path = path + '/PFAFN'
        os.makedirs(sub_path,exist_ok=True)
        cv_img = p_tryon.detach().cpu().numpy()
        rgb=(cv_img*255).astype(np.uint8)
        bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
        return bgr
    
    def runInference(self, input_frame):
        I_path = '/home/arexhari/aylmer843/PF-AFN/PF-AFN_test/dataset/test_img/000066_0.jpg'
        C_path = '/home/arexhari/aylmer843/PF-AFN/PF-AFN_test/dataset/test_clothes/003434_1.jpg'
        E_path = '/home/arexhari/aylmer843/PF-AFN/PF-AFN_test/dataset/test_edge/003434_1.jpg'
        opt = TestOptions().parse()
        I = input_frame.convert('RGB')
        input_frame = I
        params = get_params(opt, I.size)
        transform = get_transform(opt, params)
        transform_E = get_transform(opt, params, method=Image.NEAREST, normalize=False)

        I_tensor = transform(I)

        C = Image.open(C_path).convert('RGB')
        C_tensor = transform(C)

        E = Image.open(E_path).convert('L')
        E_tensor = transform_E(E)

        self.data = { 'image': I_tensor,'clothes': C_tensor, 'edge': E_tensor}

        inferenceObj = dressUpInference()
        startTime = time.time()
        I = input_frame.convert('RGB')
        params = get_params(opt, I.size)
        transform = get_transform(opt, params)
        transform_E = get_transform(opt, params, method=Image.NEAREST, normalize=False)

        I_tensor = transform(I)

        C = Image.open(C_path).convert('RGB')
        C_tensor = transform(C)

        E = Image.open(E_path).convert('L')
        E_tensor = transform_E(E)

        data = { 'image': I_tensor,'clothes': C_tensor, 'edge': E_tensor}

        self.infer(data)
