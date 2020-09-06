# Python 3.7
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset
import torchvision.datasets as dset
import torchvision.transforms as transforms
from train import Config, SiameseNetwork, ContrastiveLoss, normalize
from test_CMC import TestSiameseNetworkDataset

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
    
    #Load pictures of the target person
    folder_dataset_test = dset.ImageFolder(root=Config.reid_prepare_dir)
    reid_prepare_siamese_dataset = TestSiameseNetworkDataset(imageFolderDataset=folder_dataset_test,
                                                       transform=transforms.Compose([transforms.Resize((160,60)),
                                                                                     transforms.ToTensor()]))
    reid_prepare_dataloader = DataLoader(reid_prepare_siamese_dataset,num_workers=0,batch_size=1,shuffle=True)
    
    #Calculate distances between every 2 and get maximum.
    dis=[]
    for i,data_i in enumerate(reid_prepare_dataloader,0):
        img0_hsv, img0_grad, label_0 = data_i
        for j,data_j in enumerate(reid_prepare_dataloader,0):
            img1_hsv, img1_grad, label_1 = data_j
            if torch.equal(img0_hsv, img1_hsv):
                continue 
            else:
                distance  = f_Distance(img0_hsv, img0_grad, img1_hsv, img1_grad)
                distance = distance.detach().numpy().tolist()
                dis.append(distance[0])
    max_dis=max(dis)
    print("max_dis:{:.3f}".format(max_dis))

