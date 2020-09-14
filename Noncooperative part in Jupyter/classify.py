#! /usr/bin/python
# -*- coding: UTF-8 -*-
# python 3.7
# input: put CUHK01 dataset under data file: ./data/CUHK01
# output: training_set (./data/training_set) and test_set (./data/test_set)
 
import os
import difflib
import shutil

file_dir = "./data/CUHK01/campus"
save_train_dir = "./data/training_set"
save_test_dir = "./data/test_set"

def order(file_list,file_dir_list):
    file_dict = {}
    for parent, _, filenames in os.walk(file_dir):
        for filename in filenames:
            file_dict[filename] = os.path.join(parent, filename)
    return sorted(file_dict.items(),key = lambda x:x[0], reverse = False)            

def move_file(dict):
    index = 0
    dict_len = len(dict)
    while index < (dict_len-3):
        if index <= 870*4:#Keep 100 persons' images in test_set
            save_dir = save_train_dir
        else:
            save_dir = save_test_dir
        
        pic1_root = dict[index][1]        
        pic2_root = dict[index+1][1]          
        pic3_root = dict[index+2][1]          
        pic4_root = dict[index+3][1]
        
        save_doc_name = str(index//4)
        save_doc_root = os.path.join(save_dir, save_doc_name)
        if not os.path.exists(save_doc_root):
            os.makedirs(save_doc_root) 
            
        shutil.copy(pic1_root,save_doc_root) 
        shutil.copy(pic2_root,save_doc_root)
        shutil.copy(pic3_root,save_doc_root)
        shutil.copy(pic4_root,save_doc_root)        
        index += 4

if __name__=='__main__':
    file_list=[]
    file_dir_list=[]
    file_dict=order(file_list,file_dir_list)
    move_file(file_dict)