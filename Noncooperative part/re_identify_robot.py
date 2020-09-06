# Python 3.7
# Acknoledgment for Harshvardhan Gupta.
# Some work(data loading and sub-network structure) are based on "Facial-Similarity-with-Siamese-Networks-in-Pytorch"
# Which is avalable at: https://github.com/harveyslash/Facial-Similarity-with-Siamese-Networks-in-Pytorch
import random
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset
import torchvision.datasets as dset
import torchvision.transforms as transforms

def normalize(data):
    m = np.mean(data)
    mx = max(data)
    mn = min(data)
    return np.array([[(i - m)/(mx - mn) for i in data]])

class Config_r():
    reid_robot_dir = "./data/reid_robot/"

class TestSiameseNetworkDataset(Dataset):
    
    def __init__(self,imageFolderDataset,transform=None):
        self.imageFolderDataset = imageFolderDataset  
        self.transform = transform
        
    def __getitem__(self,index):
        img0_tuple = self.imageFolderDataset.imgs[index]#imgs has the form like [('./person_i/xxx.jpg',label_i),...]
        img0 = Image.open(img0_tuple[0])
        
        #Global HSV feature
        img0_hsv = img0.convert('HSV')
        
        #Local Laplacian texture feature
        img0_grad = cv2.imread(img0_tuple[0])
        img0_grad = cv2.Laplacian(img0_grad,cv2.CV_16S,ksize = 3)
        img0_grad = cv2.convertScaleAbs(img0_grad)#absolute value
        img0_grad = Image.fromarray(img0_grad.astype('uint8')).convert('RGB')
        
        if self.transform is not None:
            img0_hsv = self.transform(img0_hsv)
            img0_grad = self.transform(img0_grad)
        label = img0_tuple[1]
        return img0_hsv, img0_grad, label
    
    def __len__(self):
        return len(self.imageFolderDataset.imgs)

#A simple CNN with 3 convolutional layers and 3 fully connected layers
class SiameseNetwork(nn.Module):
    
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, 12, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(12),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(12, 24, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(24),

            nn.ReflectionPad2d(1),
            nn.Conv2d(24, 24, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(24),
        )

        self.fc = nn.Sequential(
            nn.Linear(24*160*60, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 18),
        )

    def forward(self, input0, input1):
        
        output0 = self.cnn(input0)#batch_size * depth * width * height
        output0 = output0.view(output0.size()[0], -1)#(batch_size) * (depth * width * height)
        output0 = self.fc(output0)
        
        output1 = self.cnn(input1)
        output1 = output1.view(output1.size()[0], -1)
        output1 = self.fc(output1)
        
        return output0, output1

def f_Distance(img0_hsv, img0_grad, img1_hsv, img1_grad):
    a = 0.9
    output0_hsv,output1_hsv = net(Variable(img0_hsv),Variable(img1_hsv))
    output0_grad,output1_grad = net(Variable(img0_grad),Variable(img1_grad))
    output0 = torch.cat((a * output0_hsv, (1-a) * output0_grad), 1)
    output1 = torch.cat((a * output1_hsv, (1-a) * output1_grad), 1)
                
    euclidean_distance = F.pairwise_distance(output0, output1)
    cos_distance = np.linalg.norm(normalize(output0.data.numpy()[0]) - normalize(output1.data.numpy()[0]))
    distance  = cos_distance * euclidean_distance
    return distance

if __name__=='__main__':
    net = SiameseNetwork()
    
    #Load model
    save_p = "./net_test.pth"
    checkpoint = torch.load(save_p)
    net.load_state_dict(checkpoint['net'])

    #Load target person's prepared photo and the captured photo
    folder_dataset_test = dset.ImageFolder(root=Config_r.reid_robot_dir)
    reid_robot_siamese_dataset = TestSiameseNetworkDataset(imageFolderDataset=folder_dataset_test,
                                        transform=transforms.Compose([transforms.Resize((160,60)),
                                                                      transforms.ToTensor()]))
    reid_robot_dataloader = DataLoader(reid_robot_siamese_dataset,num_workers=0,batch_size=1,shuffle=False)
    
    _,data1 = list(enumerate(reid_robot_dataloader,0))[0]
    _,data2 = list(enumerate(reid_robot_dataloader,0))[1]
    img0_hsv, img0_grad,_ = data1
    img1_hsv, img1_grad,_ = data2
    distance = f_Distance(img0_hsv, img0_grad, img1_hsv, img1_grad)
    distance = distance.detach().numpy().tolist()[0]
    print("distance:{:.3f}".format(distance))
    
    seperate_line = 10 
    max_dis = 8.840#Gain from re_identify_prepare.py
    judge_dis = min(seperate_line, max_dis)
    
    if distance < judge_dis:
        print('same person')
    else:
        print('different person')

