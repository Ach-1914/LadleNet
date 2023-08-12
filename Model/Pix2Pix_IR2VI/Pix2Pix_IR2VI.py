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
from pytorch_msssim import ssim, ms_ssim,MS_SSIM,SSIM
from typing import Union, List
from scipy.ndimage import gaussian_filter

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.enc1 = nn.Sequential(nn.Conv2d(3,64,4,1,2)
                                 ,nn.MaxPool2d(2)
                                 ,nn.LeakyReLU(0.2))
        
        self.enc2 = self.Conv_down_up(64,128,model='down')
        self.enc3 = self.Conv_down_up(128,256,model='down')
        self.enc4 = self.Conv_down_up(256,512,model='down')
        self.enc5 = self.Conv_down_up(512,512,model='down')
        self.enc6 = self.Conv_down_up(512,512,model='down')
        self.enc7 = self.Conv_down_up(512,512,model='down')
        
        self.enc8 = nn.Sequential(nn.Conv2d(512,512,4,1,2)
                                 ,nn.MaxPool2d(2)
                                 ,nn.LeakyReLU(0.2))
        
        self.dec8 = self.Conv_down_up(512,512,model='up',dropout=True)
        self.dec8_skip = self.skip_conv(1024,512)
        
        self.dec7 = self.Conv_down_up(512,512,model='up',dropout=True)
        self.dec7_skip = self.skip_conv(1024,512)
        
        self.dec6 = self.Conv_down_up(512,512,model='up',dropout=True)
        self.dec6_skip = self.skip_conv(1024,512)
        
        self.dec5 = self.Conv_down_up(512,512,model='up')
        self.dec5_skip = self.skip_conv(1024,512)
        
        self.dec4 = self.Conv_down_up(512,256,model='up')
        self.dec4_skip = self.skip_conv(768,256)
        
        self.dec3 = self.Conv_down_up(256,128,model='up')
        self.dec3_skip = self.skip_conv(384,128)
        
        self.dec2 = self.Conv_down_up(128,64,model='up')
        self.dec2_skip = self.skip_conv(192,64)
        
        self.dec1 = self.Conv_down_up(64,3,model='up')
        self.dec1_skip = self.skip_conv(64,3)
        
        self.MaxPooling = nn.MaxPool2d(2)
        
    def Conv_down_up(self, in_channels, out_channels, model: str, dropout=False):
        conv = nn.Sequential()
        
        if model=='down':
            conv.add_module('Down_Sample', nn.Conv2d(in_channels, out_channels, 4, 1, 2))
            conv.add_module('Max_Pooling', nn.MaxPool2d(2))
            conv.add_module('IN_Layer', nn.InstanceNorm2d(out_channels))
            conv.add_module('Leaky_Relu', nn.LeakyReLU(0.2))
            
        if model=='up':
            conv.add_module('Up_Sample', nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1))
            conv.add_module('Leaky_Relu', nn.LeakyReLU(0.2))
            conv.add_module('IN_Layer', nn.InstanceNorm2d(out_channels))
            
            if dropout:
                conv.add_module('Dropout', nn.Dropout(0.5))
                
        return conv
        
    def skip_conv(self, in_channels, out_channels):
        skip_conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 1, 1)
                                 ,nn.LeakyReLU(0.2))
        return skip_conv
        
    def forward(self,x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)  
        enc5 = self.enc5(enc4)       
        enc6 = self.enc6(enc5)        
        enc7 = self.enc7(enc6)        
        enc8 = self.enc8(enc7)
        
        dec8 = self.dec8(enc8)
        dec8_skip = torch.cat((dec8, enc7), 1)
        dec8 = self.dec8_skip(dec8_skip)
        
        dec7 = self.dec7(dec8)
        dec7_skip = torch.cat((dec7, enc6), 1)
        dec7 = self.dec7_skip(dec7_skip)
        
        dec6 = self.dec6(dec7)
        dec6_skip = torch.cat((dec6, enc5), 1)
        dec6 = self.dec6_skip(dec6_skip)
        
        dec5 = self.dec5(dec6)
        dec5_skip = torch.cat((dec5, enc4), 1)
        dec5 = self.dec5_skip(dec5_skip)
        
        dec4 = self.dec4(dec5)
        dec4_skip = torch.cat((dec4, enc3), 1)
        dec4 = self.dec4_skip(dec4_skip)
        
        dec3 = self.dec3(dec4)
        dec3_skip = torch.cat((dec3, enc2), 1)
        dec3 = self.dec3_skip(dec3_skip)
        
        dec2 = self.dec2(dec3)
        dec2_skip = torch.cat((dec2, enc1), 1)
        dec2 = self.dec2_skip(dec2_skip)
        
        dec1 = self.dec1(dec2)
        dec1 = self.dec1_skip(dec1)
        
        return dec1

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.dis = nn.Sequential(
            # input is (nc) x 256 x 256
            nn.Conv2d(6, 32, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            
            # state size. (ndf) x 128 x 128
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2),
            
            # state size. (ndf*2) x 64 x 64
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),
            
            # state size. (ndf*4) x 32 x 32
            nn.Conv2d(128, 256, 4, stride=1, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2),
            
            # state size. (ndf*8) x 31 x 31
            nn.Conv2d(256, 1, 4, stride=1, padding=1),
            
            # output size. 1 x 30 x 30
            nn.Sigmoid(),
            
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, ir_img, vi_img):
        x = torch.cat((ir_img, vi_img), 1)
        return self.dis(x)

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
num_epochs = 100
learning_rate = 0.0002
save_step = int(num_epochs * 0.1)

transform_pre = transforms.Compose([transforms.ToTensor()
                                  ,transforms.Resize((400,400))
                                  ,transforms.CenterCrop((256, 256))])

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
gen = Generator().to(device)
dis = Discriminator().to(device)
content_criterion = nn.L1Loss()
adversarial_criterion = torch.nn.BCELoss()
loss_ssim = SSIM(data_range=1., size_average=True, channel=3)
loss_l1 = nn.L1Loss()
optimizer_G = torch.optim.Adam(gen.parameters(), lr=learning_rate, betas=(0.5,0.9))
optimizer_D = torch.optim.Adam(dis.parameters(), lr=learning_rate, betas=(0.5,0.9))

lr_record = learning_rate
lowest_loss = float('inf')

now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
file_name = f'result/{now}.txt'
file = open(file_name, "a")
sys.stdout = file

for epoch in range(num_epochs):
    start_time = time.time()

    gen.train()
    dis.train()
    
    running_gen_loss = 0.0
    running_dis_loss = 0.0
    
    for i, data in enumerate(train_loader, 0):
        ir_inputs, vi_inputs = data
        ir_imgs, vi_imgs = ir_inputs.to(device), vi_inputs.to(device)
        
        batch_size = ir_imgs.size(0)
        
        real_label = torch.full((batch_size,), 1., device=device)
        fake_label = torch.full((batch_size,), 0., device=device)
        
        optimizer_G.zero_grad()
  
        fake_imgs = gen(ir_imgs)
        output = dis(fake_imgs.detach(), vi_imgs).squeeze()
        
        content_loss = content_criterion(fake_imgs, vi_imgs)
        adversarial_loss = adversarial_criterion(output, real_label)
        total_loss = content_loss + 100 * adversarial_loss

        total_loss.backward()
        optimizer_G.step()
        
        optimizer_D.zero_grad()

        output_real = dis(ir_imgs, vi_imgs).squeeze()
        loss_D_real = adversarial_criterion(output_real, real_label)
        
        output_fake = dis(ir_imgs, fake_imgs.detach()).squeeze()
        loss_D_fake = adversarial_criterion(output_fake, fake_label)
        loss_D = (loss_D_real + loss_D_fake) / 2

        loss_D.backward()
        optimizer_D.step()

        running_gen_loss += total_loss.item()
        running_dis_loss += loss_D.item()
    
    avg_gen_loss = running_gen_loss / len(train_loader)
    avg_dis_loss = running_dis_loss / len(train_loader)

    gen.eval()
    
    l1_val = 0.0
    ssim_val = 0.0

    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):
            ir_inputs_val, vi_inputs_val = data
            ir_inputs_val, vi_inputs_val = ir_inputs_val.to(device), vi_inputs_val.to(device)

            outputs_val = gen(ir_inputs_val)

            ssim_val_value = loss_ssim(outputs_val, vi_inputs_val)
            l1_val_value = loss_l1(outputs_val, vi_inputs_val)
            
            ssim_val += ssim_val_value.item()
            l1_val += l1_val_value.item()

    avg_ssim_val = ssim_val / len(val_loader)
    avg_l1_val = l1_val / len(val_loader)
    
    if (epoch+1) % save_step == 0 or (epoch + 1) == num_epochs:
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_path_G = os.path.join("weight/", f"{now}_Pix2Pix_IR2VI_G_KAIST.pth")
        state_dict_G = gen.state_dict()
        metadata_G = {'learning_rate': optimizer_G.param_groups[0]['lr'],
                    'batch_size': batch_size,
                    'num_epochs': num_epochs,
                    'current_epoch': epoch+1,
                    'optimizer_name': type(optimizer_G).__name__.lower(),
                    'loss_function': 'Loss_Content(L1) + Loss_Adversarial',
                    'Gen_Loss': np.round(avg_gen_loss, 6),
                    'Dis_Loss': np.round(avg_dis_loss, 6),
                    'SSIM': f"(val){np.round(avg_ssim_val, 6)}",
                    'L1': f"(val){np.round(avg_l1_val, 6)}"}

        torch.save({'state_dict': state_dict_G, 'metadata': metadata_G}, save_path_G)

        save_path_D = os.path.join("weight/", f"{now}_Pix2Pix_IR2VI_D_KAIST.pth")
        state_dict_D = dis.state_dict()
        metadata_D = {'learning_rate': optimizer_D.param_groups[0]['lr'],
                    'batch_size': batch_size,
                    'num_epochs': num_epochs,
                    'current_epoch': epoch+1,
                    'optimizer_name': type(optimizer_D).__name__.lower(),
                    'loss_function': 'Loss_Content(L1) + Loss_Adversarial',
                    'Gen_Loss': np.round(avg_gen_loss, 6),
                    'Dis_Loss': np.round(avg_dis_loss, 6),
                    'SSIM': f"(val){np.round(avg_ssim_val, 6)}",
                    'L1': f"(val){np.round(avg_l1_val, 6)}"}

        torch.save({'state_dict': state_dict_D, 'metadata': metadata_D}, save_path_D)
        
    end_time = time.time()
    time_diff = end_time - start_time
    hours, remainder = divmod(time_diff, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print('Epoch [{}/{}], Gen_Loss:{:.4f}, Dis_Loss:{:.4f}, SSIM:(val){:.4f}, L1:(val){:.4f}, Time: {}h-{}m-{}s'.format(epoch+1, num_epochs, avg_gen_loss, avg_dis_loss, avg_ssim_val, avg_l1_val, int(hours), int(minutes), int(seconds)))
    
    
sys.stdout = file