"""
This file performs buiding dataset.
Date: 2022-07-30
"""

import pdb
import numpy as np
import pandas as pd

def split_data(x,train_end,valid_end):
    return x[:train_end],x[train_end:valid_end],x[valid_end:]

def load_data(args):
    loaded_data = np.load(args.path+'data_long.npz',allow_pickle=True)
    x_long = loaded_data["datas"]
    seq_mask = loaded_data["mask"]
    x_cs=np.load(args.path+'cs_data.npy',allow_pickle=True)  
    x_text = np.load(args.path+'text_emb.npy', allow_pickle=True) 
    x_long_max, x_long_min = \
        np.max(np.max(np.abs(x_long), axis=1, keepdims=True), axis=0, keepdims=True), np.min(np.min(np.abs(x_long), axis=1, keepdims=True), axis=0, keepdims=True) 
    if args.label_type == '1_month':
        label = np.load(args.path+'y_1month.npy', allow_pickle=True)
    elif args.label_type == '3_month':
        label = np.load(args.path+'y_3month.npy', allow_pickle=True)
    else:
        raise FileNotFoundError("Not valid type.")
    id_permute = [i for i in range(len(label))]
    np.random.seed(args.seed)
    #np.random.shuffle(id_permute)
    x_long, x_cs, x_text, label = x_long[id_permute], x_cs[id_permute], x_text[id_permute], label[id_permute]

    x_long = (x_long-x_long_min)/(x_long_max-x_long_min)
    x_long = np.clip(x_long,0,1)

    delta = np.zeros_like(x_long)  
    for i in range(0, x_long.shape[1]):
        delta[:, i, :] = 1

    delta_rev = np.zeros_like(x_long)  
    for i in range(x_long.shape[1]-2,-1,-1):
        delta_rev[:,i, :] = 1

    Mask=np.where(x_long==0,0,1) 
    x_mean_g = np.sum(np.sum(x_long, axis=0, keepdims=True), axis=1) / np.sum(np.sum(Mask, axis=0, keepdims=True), axis=1, keepdims=True)
  
    x_long_last_obs = np.copy(x_long)
    for ts in range(1,x_long.shape[1]):
        delta[:,ts,:] = delta[:, ts, :] + delta[:, ts-1,:]*(1-Mask[:,ts-1,:]) 
        x_long_last_obs[:,ts,:]=x_long_last_obs[:,ts-1,:]*(1-Mask[:,ts-1,:])+x_long[:,ts-1,:]*Mask[:,ts-1,:]

    x_long_last_obs_rev = np.copy(x_long)
    for rts in range(x_long.shape[1]-1,0,-1):
        delta_rev[:, rts-1, :] = delta_rev[:, rts-1, :] + delta_rev[:, rts, :] * (1 - Mask[:, rts, :])
        x_long_last_obs_rev[:, rts-1, :] = x_long_last_obs_rev[:, rts, :] * (1 - Mask[:, rts, :]) + x_long[:, rts, :] * Mask[:, rts,:]

    delta = delta / delta.max()  
    delta_rev = delta_rev/delta_rev.max()

    all_data = np.stack((x_long, x_long_last_obs,x_long_last_obs_rev, Mask, delta, delta_rev, np.repeat(seq_mask,x_long.shape[-1],axis=-1)), axis=1)
    
   
    train_end = int(len(label) * 0.5)
    valid_end = int(len(label) * 0.75)

    train_x_all, valid_x_all, test_x_all = split_data(all_data, train_end, valid_end)
    train_y, valid_y, test_y = split_data(label, train_end, valid_end)
    train_cs, valid_cs, test_cs = split_data(x_cs, train_end, valid_end)
    train_text, valid_text, test_text = split_data(x_text, train_end, valid_end)
   

    return train_x_all, train_y, valid_x_all, valid_y, test_x_all, test_y, x_mean_g,train_cs,valid_cs,test_cs,train_text,valid_text,test_text

