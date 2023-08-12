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

class Bridged_UNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.enc1 = nn.Sequential(nn.Conv2d(3, 32, 3, 1, 1)
                                ,nn.BatchNorm2d(32)
                                ,nn.ReLU()
                                 
                                ,nn.Conv2d(32, 32, 3, 1, 1)
                                ,nn.BatchNorm2d(32)
                                ,nn.ReLU()
                                )
        
        self.enc2 = self.conv_block_down(32, 64, fun='Relu')
        self.enc3 = self.conv_block_down(64, 128, fun='Relu')
        self.enc4 = self.conv_block_down(128, 256, fun='Elu')
        self.enc5 = self.conv_block_down(256, 512, fun='Elu')
        
        self.dec4 = self.conv_block_up(768, 256)
        self.dec3 = self.conv_block_up(384, 128)
        self.dec2 = self.conv_block_up(192, 64)
        self.dec1 = self.conv_block_up(96, 32)
        
        #---------------------------------------------------------#
        
        self.enc_1 = self.conv_block_down(64, 32, fun='Relu')
        self.enc_2 = self.conv_block_down(96, 64, fun='Relu')
        self.enc_3 = self.conv_block_down(192, 128, fun='Elu')
        self.enc_4 = self.conv_block_down(384, 256, fun='Elu')
        
        self.dec_4 = self.conv_block_up(384, 768)
        self.dec_4_ = self.conv_block_up(768, 512)
        
        self.dec_3 = self.conv_block_up(576, 384)
        self.dec_3_ = self.conv_block_up(384, 256)
        
        self.dec_2 = self.conv_block_up(288, 192)
        self.dec_2_ = self.conv_block_up(192, 128)
        
        self.dec_1 = self.conv_block_up(224, 96)
        self.dec_1_ = self.conv_block_up(96, 64)
        
        self.input_layer = nn.Sequential(nn.Conv2d(3, 32, 3, 1, 1)
                                ,nn.BatchNorm2d(32)
                                ,nn.ReLU()
                                 
                                ,nn.Conv2d(32, 32, 3, 1, 1)
                                ,nn.BatchNorm2d(32)
                                ,nn.ReLU()
                                )
        
        self.output_layer = nn.Conv2d(32,3,3,1,1)
        self.output_layer_ = nn.Conv2d(64,3,3,1,1)
        
        self.up_sample = nn.Upsample(scale_factor=2)
        
    def conv_block_down(self, in_ch, out_ch, fun='Relu'):
        if fun == 'Relu':
            conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, 1, 1)
                                ,nn.BatchNorm2d(out_ch)
                                ,nn.ReLU()
                                 
                                ,nn.Conv2d(out_ch, out_ch, 3, 2, 1)
                                ,nn.BatchNorm2d(out_ch)
                                ,nn.ReLU()
                                )
            
        if fun == 'Elu':
            conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, 1, 1)
                                ,nn.BatchNorm2d(out_ch)
                                ,nn.ELU()
                                 
                                ,nn.Conv2d(out_ch, out_ch, 3, 2, 1)
                                ,nn.BatchNorm2d(out_ch)
                                ,nn.ELU()
                                )
        return conv
    
    def conv_block_up(self, in_ch, out_ch):
        conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, 1, 1)
                            ,nn.BatchNorm2d(out_ch)
                            ,nn.ReLU()
                            )
        return conv
    
    
    def forward(self,x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)
        
        up1 = self.up_sample(enc5)
        cat1 = torch.cat((up1, enc4), 1)
        dec4 = self.dec4(cat1)
        
        up2 = self.up_sample(dec4)
        cat2 = torch.cat((up2, enc3), 1)
        dec3 = self.dec3(cat2)
        
        up3 = self.up_sample(dec3)
        cat3 = torch.cat((up3, enc2), 1)
        dec2 = self.dec2(cat3)
        
        up4 = self.up_sample(dec2)
        cat4 = torch.cat((up4, enc1), 1)
        dec1 = self.dec1(cat4)
        
        output = self.output_layer(dec1)
        
        #---------------------------------------------------#
        
        enc_ = self.input_layer(output)
        
        cat_1 = torch.cat((enc_, dec1), 1)
        enc_1 = self.enc_1(cat_1)
        
        cat_2 = torch.cat((enc_1, dec2), 1)
        enc_2 = self.enc_2(cat_2)
        
        cat_3 = torch.cat((enc_2, dec3), 1)
        enc_3 = self.enc_3(cat_3)
        
        cat_4 = torch.cat((enc_3, enc4), 1)
        enc_4 = self.enc_4(cat_4)
        
        up_1 = self.up_sample(enc_4)
        cat_5 = torch.cat((up_1, enc_3), 1) 
        dec_4 = self.dec_4(cat_5) + cat1
        dec_4_ = self.dec_4_(dec_4)
        
        up_2 = self.up_sample(dec_4_)
        cat_6 = torch.cat((up_2, enc_2), 1) 
        dec_3 = self.dec_3(cat_6) + cat2
        dec_3_ = self.dec_3_(dec_3)
        
        up_3 = self.up_sample(dec_3_)
        cat_7 = torch.cat((up_3, enc_1), 1) 
        dec_2 = self.dec_2(cat_7) + cat3
        dec_2_ = self.dec_2_(dec_2)
        
        up_4 = self.up_sample(dec_2)
        cat_8 = torch.cat((up_4, enc_), 1) 
        dec_1 = self.dec_1(cat_8) + cat4
        dec_1_ = self.dec_1_(dec_1)
        
        output_ = self.output_layer_(dec_1_)
        
        return output_

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

batch_size = 56
num_epochs = 100
learning_rate = 0.01
save_step = int(num_epochs * 0.1)

transform_pre = transforms.Compose([transforms.ToTensor()
                                  ,transforms.Resize((300,400))
                                  ,transforms.CenterCrop((192, 256))])

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
model = Bridged_UNet().to(device)
loss_msssim = MS_SSIM(data_range=1., size_average=True, channel=3, weights=[0.5, 1., 2., 4., 8.], K=(0.01, 0.04))
loss_ssim = SSIM(data_range=1., size_average=True, channel=3)
loss_l1 = nn.L1Loss()
loss_l2 = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=False, cooldown=5)

lr_record = learning_rate
lowest_loss = float('inf')

now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
file_name = f'result/{now}.txt'
file = open(file_name, "a")
sys.stdout = file

torch.autograd.set_detect_anomaly(True)

for epoch in range(num_epochs):
    start_time = time.time()

    model.train()
    
    train_loss = 0.0
    msssim_loss = 0.0
    l1_loss = 0.0
    
    for i, data in enumerate(train_loader, 0):
        ir_inputs, vi_inputs = data
        ir_inputs, vi_inputs = ir_inputs.to(device), vi_inputs.to(device)

        optimizer.zero_grad()
        
        outputs = model(ir_inputs)
        
        msssim = 0.84 * (1 - loss_msssim(outputs, vi_inputs))
        l1 = (1-0.84) * loss_l1(outputs, vi_inputs)
        
        total_loss = msssim + l1
        
        total_loss.backward()
        optimizer.step()
        
        train_loss += total_loss.item()
        msssim_loss += msssim.item()
        l1_loss += l1.item()
  
    average_loss = train_loss / len(train_loader)
    avg_msssim_loss = msssim_loss / len(train_loader)
    avg_l1_loss = l1_loss / len(train_loader)
    
    model.eval()
    
    msssim_val = 0.0
    l1_val = 0.0
    ssim_val = 0.0

    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):
            ir_inputs_val, vi_inputs_val = data
            ir_inputs_val, vi_inputs_val = ir_inputs_val.to(device), vi_inputs_val.to(device)

            outputs_val = model(ir_inputs_val)

            ssim_val_value = loss_ssim(outputs_val, vi_inputs_val)
            l1_val_value = loss_l1(outputs_val, vi_inputs_val)
            msssim_val_value = loss_msssim(outputs_val, vi_inputs_val)
            
            ssim_val += ssim_val_value.item()
            l1_val += l1_val_value.item()
            msssim_val += msssim_val_value.item()

    scheduler.step(average_loss)

    avg_ssim_val = ssim_val / len(val_loader)
    avg_msssim_val = msssim_val / len(val_loader)
    avg_l1_val = l1_val / len(val_loader)

    
    if average_loss < lowest_loss:
        now = datetime.datetime.now().strftime("%Y-%m-%d")
        lowest_loss = average_loss
        lowest_loss_save_path = os.path.join("weight/", f"Bridged_UNet_lowest_loss_{now}.pth")
        state_dict_lowest_loss = model.state_dict()
        metadata_lowest_loss = {'learning_rate': scheduler.optimizer.param_groups[0]['lr'],
                                'batch_size': batch_size,
                                'num_epochs': num_epochs,
                                'current_epoch': epoch+1,
                                'optimizer_name': type(optimizer).__name__.lower(),
                                'loss_function': 'Loss_Content(MS-SSIM, L1)',
                                'MS-SSIM': np.round(avg_msssim_val, 6),
                                'L1': np.round(avg_l1_val, 6),
                                'lowest_val_loss': np.round(lowest_loss, 6)}
        
        torch.save({'state_dict': state_dict_lowest_loss, 'metadata': metadata_lowest_loss}, lowest_loss_save_path)
    
    if (epoch+1) % save_step == 0 or (epoch + 1) == num_epochs:
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_path = os.path.join("weight/", f"{now}_Bridged_UNet.pth")
        state_dict = model.state_dict()
        metadata = {'learning_rate': scheduler.optimizer.param_groups[0]['lr'],
                    'batch_size': batch_size,
                    'num_epochs': num_epochs,
                    'current_epoch': epoch+1,
                    'optimizer_name': type(optimizer).__name__.lower(),
                    'loss_function': 'Loss_Content(MS-SSIM, L1)',
                    'MS-SSIM': f"(train){np.round(avg_msssim_loss, 6)} / (val){np.round(avg_msssim_val, 6)}",
                    'L1': f"(train){np.round(avg_l1_loss, 6)} / (val){np.round(avg_l1_val, 6)}",
                    'Total_Loss': f"(train){np.round(average_loss, 6)}"}

        torch.save({'state_dict': state_dict, 'metadata': metadata}, save_path)
        
    end_time = time.time()
    time_diff = end_time - start_time
    hours, remainder = divmod(time_diff, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    if lr_record != scheduler.optimizer.param_groups[0]['lr']:
        lr_record = scheduler.optimizer.param_groups[0]['lr']
        print(f'--------------------Learning_Rate: {lr_record}--------------------')
    
    print('Epoch [{}/{}], (Train_Loss) MS-SSIM:{:.4f}, L1:{:.4f}, Total:{:.4f}   (Val_Value) MS-SSIM:{:.4f}, SSIM:{:.4f}, L1:{:.4f}, Time: {}h-{}m-{}s'.format(epoch+1, num_epochs, avg_msssim_loss, avg_l1_loss, average_loss, avg_msssim_val, avg_ssim_val, avg_l1_val, int(hours), int(minutes), int(seconds)))
    
    
sys.stdout = file