import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import random
import torch
import torch.nn as nn
import torch.optim as opt
import numpy as np
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, random_split
import torchvision.utils as vutils
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import Subset
from torch.utils.data import Dataset
from skimage import io
import skimage
import os
import cv2
import datetime
import time
import pickle
from math import exp
from torch.autograd import Variable
import sys
from torch.optim.lr_scheduler import ReduceLROnPlateau
import lpips
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from pytorch_msssim import ms_ssim,MS_SSIM,SSIM
from typing import Union, List

class UNet_IR2VI(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.enc1 = self.conv(3, 32, (2,2), 1, BN=False, Drop=False)
        self.enc2 = self.conv(32, 64, (2,2), 1, BN=True, Drop=False)
        self.enc3 = self.conv(64, 128, (2,2), 1, BN=True, Drop=False)
        self.enc4 = self.conv(128, 256, (2,2), 1, BN=True, Drop=False)
        self.enc5 = self.conv(256, 512, (2,2), 1, BN=True, Drop=False)
        
        self.enc6 = self.conv(512, 512, (2,2), 1, BN=True, Drop=False)
        
        self.enc7 = self.conv(512, 512, (1,1), 1, BN=True, Drop=False)
        self.enc8 = self.conv(512, 512, (1,1), 1, BN=True, Drop=True)
        
        self.dec5 = self.conv(1024, 512, (1,1), 1, BN=True, Drop=True)
        self.dec4 = self.conv(768, 256, (1,1), 1, BN=True, Drop=True)
        self.dec3 = self.conv(384, 128, (1,1), 1, BN=True, Drop=True)
        self.dec2 = self.conv(192, 64, (1,1), 1, BN=True, Drop=True)
        self.dec1 = self.conv(96, 32, (1,1), 1, BN=True, Drop=True)
        
        self.upsampling = self.conv(32, 32, (1,1), 1, BN=False, Drop=False)
        
        self.output = nn.Sequential(nn.Conv2d(32,3,4,(1,1),1)
                                   ,nn.Sigmoid())
        
        self.up = nn.Upsample(scale_factor=2)
        
    def conv(self, in_channels, out_channels, stride, padding, BN=True, Drop=False):
        conv = nn.Sequential(nn.LeakyReLU(0.2)
                            ,nn.Conv2d(in_channels, out_channels, 4, stride, padding))
        if BN:
            conv.add_module('BN_Layer', nn.BatchNorm2d(out_channels))
        
        if Drop:
            conv.add_module('Drop_Layer', nn.Dropout(0.2))
            
        return conv
    
    def pad2target(self, input_tensor, target_size):
        batch_size, num_channels, input_height, input_width = input_tensor.size()

        target_height, target_width = target_size

        scale_factor_height = target_height / input_height
        scale_factor_width = target_width / input_width

        output_tensor = torch.nn.functional.interpolate(input_tensor, scale_factor=(scale_factor_height, scale_factor_width), mode='bilinear', align_corners=False)

        return output_tensor
    
    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)        
        enc5 = self.enc5(enc4)
        
        enc6 = self.enc6(self.pad2target(enc5, (6,9)))       
        enc7 = self.enc7(self.pad2target(enc6, (4,5)))
        enc8 = self.enc8(self.pad2target(enc7, (4,5)))
        
        dec5 = self.dec5(self.pad2target(torch.cat((self.up(enc8),enc5),1), (7,9)))
        dec4 = self.dec4(self.pad2target(torch.cat((self.up(dec5),enc4),1), (13,17)))
        dec3 = self.dec3(self.pad2target(torch.cat((self.up(dec4),enc3),1), (25,33)))
        dec2 = self.dec2(self.pad2target(torch.cat((self.up(dec3),enc2),1), (49,65)))
        dec1 = self.dec1(self.pad2target(torch.cat((self.up(dec2),enc1),1), (97,129)))
        
        up_sampling = self.upsampling(self.pad2target(self.up(dec1), (193,257)))
        output = self.output(self.pad2target(up_sampling, (193,257)))
        
        return output

class Dataset_creat(Dataset):
    def __init__(self,IR_path,VI_path,transforms:Union[List[transforms.Compose]]):
        super().__init__()
        
        self.IR_path = IR_path
        self.VI_path = VI_path
        self.filename_IR = sorted(os.listdir(self.IR_path))
        self.filename_VI = sorted(os.listdir(self.VI_path))
        self.transform = transforms[0]     
    
    def __len__(self):
        return len(os.listdir(self.IR_path))
        
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        IR_dic = os.path.join(self.IR_path, self.filename_IR[idx])
        VI_dic = os.path.join(self.VI_path, self.filename_VI[idx])
            
        img_IR = io.imread(IR_dic)
        
        if len(img_IR.shape)<3:
            img_IR = skimage.color.gray2rgb(img_IR)
        
        img_VI = io.imread(VI_dic)

        if self.transform != None:
            img_IR = self.transform(img_IR)
            
            img_VI = self.transform(img_VI)
        
        img_IR = torch.transpose(img_IR, 0, 0)
        img_VI = torch.transpose(img_VI, 0, 0)
                
        package = (img_IR,img_VI)
        return package

batch_size = 40
num_epochs = 50
learning_rate = 0.001
save_step = int(num_epochs * 0.2)

transform_pre = transforms.Compose([transforms.ToTensor()
                                  ,transforms.Resize((300,400))
                                  ,transforms.CenterCrop((256, 320))])

IR = 'Your Path to IR Images'
VI = 'Your Path to VI Images'
dataset = Dataset_creat(IR,VI,[transform_pre])

train_ratio = 0.8
test_ratio = 1 - train_ratio

train_size = int(train_ratio * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

device = torch.device('cuda:0')
model = UNet_IR2VI().to(device)
loss_ssim = SSIM(data_range=1., size_average=True, channel=3)
loss_l1 = nn.L1Loss()
loss_l2 = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=False, cooldown=5)

lr_record = learning_rate
lowest_loss = float('inf')

now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
file_name = f'result/UNet_IR2VI/{now}.txt'
file = open(file_name, "a")
sys.stdout = file

for epoch in range(num_epochs):
    start_time = time.time()

    model.train()
    
    train_loss = 0.0
    ssim_loss = 0.0
    l2_loss = 0.0
    
    for i, data in enumerate(train_loader, 0):
        ir_inputs, vi_inputs = data
        ir_inputs, vi_inputs = ir_inputs.to(device), vi_inputs.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(ir_inputs)
        
        ssim = 0.8 * (1 - loss_ssim(outputs, vi_inputs))
        l2 = (1-0.8) * loss_l2(outputs, vi_inputs)
        
        total_loss = ssim + l2
        
        total_loss.backward()
        optimizer.step()
        
        train_loss += total_loss.item()
        ssim_loss += ssim.item()
        l2_loss += l2.item()
  
        
    average_loss = train_loss / len(train_loader)
    avg_ssim_loss = ssim_loss / len(train_loader)
    avg_l2_loss = l2_loss / len(train_loader)
    
    
    model.eval()
    
    val_loss = 0.0
    ssim_val_loss = 0.0
    l2_val_loss = 0.0 

    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):
            ir_inputs_val, vi_inputs_val = data
            ir_inputs_val, vi_inputs_val = ir_inputs_val.to(device), vi_inputs_val.to(device)

            outputs_val = model(ir_inputs_val)

            ssim_val = 0.8 * (1 - loss_ssim(outputs_val, vi_inputs_val))
            l2_val = (1-0.8) * loss_l2(outputs_val, vi_inputs_val)
            
            total_val_loss = ssim_val + l2_val
            
            val_loss += total_val_loss.item()
            ssim_val_loss += ssim_val.item()
            l2_val_loss += l2_val.item()

    scheduler.step(average_loss)

    average_val_loss = val_loss / len(val_loader)
    avg_ssim_val_loss = ssim_val_loss / len(val_loader)
    avg_l2_val_loss = l2_val_loss / len(val_loader)

    
    if average_loss < lowest_loss:
        now = datetime.datetime.now().strftime("%Y-%m-%d")
        lowest_loss = average_loss
        lowest_loss_save_path = os.path.join("weight_file/UNet_IR2VI/KAIST", f"UNet_IR2VI_KAIST_lowest_loss_{now}.pth")
        state_dict_lowest_loss = model.state_dict()
        metadata_lowest_loss = {'learning_rate': scheduler.optimizer.param_groups[0]['lr'],
                                'batch_size': batch_size,
                                'num_epochs': num_epochs,
                                'current_epoch': epoch+1,
                                'optimizer_name': type(optimizer).__name__.lower(),
                                'loss_function': 'Loss_Content(SSIM, L2)',
                                'MS-SSIM': np.round(avg_ssim_loss, 6),
                                'L2': np.round(avg_l2_loss, 6),
                                'lowest_val_loss': np.round(lowest_loss, 6)}
        
        torch.save({'state_dict': state_dict_lowest_loss, 'metadata': metadata_lowest_loss}, lowest_loss_save_path)
    
    if (epoch+1) % save_step == 0 or (epoch + 1) == num_epochs:
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_path = os.path.join("weight_file/UNet_IR2VI/KAIST", f"{now}_UNet_IR2VI_KAIST.pth")
        state_dict = model.state_dict()
        metadata = {'learning_rate': scheduler.optimizer.param_groups[0]['lr'],
                    'batch_size': batch_size,
                    'num_epochs': num_epochs,
                    'current_epoch': epoch+1,
                    'optimizer_name': type(optimizer).__name__.lower(),
                    'loss_function': 'Loss_Content(SSIM, L2)',
                    'SSIM': f"(train){np.round(avg_ssim_loss, 6)} / (val){np.round(avg_ssim_val_loss, 6)}",
                    'L2': f"(train){np.round(avg_l2_loss, 6)} / (val){np.round(avg_l2_val_loss, 6)}",
                    'Total_Loss': f"(train){np.round(average_loss, 6)} / (val){np.round(average_val_loss, 6)}"}

        torch.save({'state_dict': state_dict, 'metadata': metadata}, save_path)
        
    end_time = time.time()
    time_diff = end_time - start_time
    hours, remainder = divmod(time_diff, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    if lr_record != scheduler.optimizer.param_groups[0]['lr']:
        lr_record = scheduler.optimizer.param_groups[0]['lr']
        print(f'--------------------Learning_Rate: {lr_record}--------------------')
    
    print('Epoch [{}/{}], SSIM:(Train){:.4f} / (Val){:.4f}, L2:(Train){:.4f} / (Val){:.4f}, Total_Loss:(Train){:.4f} / (Val){:.4f}, Time: {}h-{}m-{}s'.format(epoch+1, num_epochs, avg_ssim_loss, avg_ssim_val_loss, avg_l2_loss, avg_l2_val_loss, average_loss, average_val_loss, int(hours), int(minutes), int(seconds)))
    
    
sys.stdout = file