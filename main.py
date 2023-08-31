"""
This file is the main entry of the project. It performs training and testing.
Date: 2022-07-30
"""

import os
import argparse
import torch
import torch.utils.data
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score
from datetime import datetime
from tqdm import tqdm

from dataloader import load_data
from loss_op.BCEFocalLoss import BCEFocalLoss
from model_zoo.HEP import HEP
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=15,
                        help='random seed.')
    parser.add_argument('--epoches', type=int, default=10,
                        help='total epoch for training.')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='batch size.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay.')
    parser.add_argument('--hidden_size', type=int, default=16,
                        help='Number of hidden units.')
    parser.add_argument('--focal_weight', default=0.07, type=float,
                        help='class weight for focal loss.')
    parser.add_argument('--eval_epoch', default=10, type=int,
                        help='itervals for evaluation.')

    parser.add_argument('--path', type=str, default='data/',
                        help='Number of hidden units.')
    parser.add_argument('--label_type', default='1_month', type=str, choices=['1_month','3_month'],
                        help='choose whether to predict 1 or 3 month mortality.')
    parser.add_argument('--saved_dir', default='model_temp/', type=str,
                        help='path to save the trained model.')
   
    args = parser.parse_known_args()[0]
    return args



def convert_to_tensor(x):
    return [torch.Tensor(x[i]) for i in range(len(x))]

def make_dataloader(args):
    train_all_data, train_y, valid_all_data, valid_y, test_x, test_y, x_mean_g, \
        train_cs, valid_cs, test_cs, train_text, valid_text, test_text, \
    = convert_to_tensor(load_data(args))
    args.input_size = train_all_data.shape[-1]
    args.static_size =  train_cs.shape[-1]
    args.text_embed_size = train_text.shape[-1]
    

    datasets = {}
    datasets["train"] = torch.utils.data.TensorDataset(train_all_data, train_y, train_cs, train_text)
    datasets["valid"] = torch.utils.data.TensorDataset(valid_all_data, valid_y, valid_cs, valid_text)
    datasets["test"] = torch.utils.data.TensorDataset(test_x, test_y, test_cs, test_text)

    dataloader = {}
    dataloader["train"] = torch.utils.data.DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True)
    dataloader["valid"] = torch.utils.data.DataLoader(datasets["valid"], batch_size=args.batch_size, shuffle=False)
    dataloader["test"] = torch.utils.data.DataLoader(datasets["test"], batch_size=args.batch_size, shuffle=False)
    return dataloader, x_mean_g




class Trainer:
    def __init__(self, args, model, dataloader, device='cpu'):
        self.args = args
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.optimizer =  optim.Adam(params=self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.criterion = BCEFocalLoss(alpha=args.focal_weight)
        self.sigmoid = torch.nn.Sigmoid()
        self.epoches = self.args.epoches

    def _train_epoch(self):
        total_loss = 0.0
        self.model.train()
        for batch_data, batch_label, batch_s, batch_t in tqdm(self.dataloader["train"], ncols=75):
            batch_data = batch_data.to(self.device)
            batch_labels = batch_label.to(self.device)
            pred, atten_mean = self.model(batch_data, batch_s, batch_t)
            loss = self.criterion(batch_labels, pred)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10, norm_type=2)
            self.optimizer.step()
            total_loss = total_loss + loss.item()
       
    
        return total_loss


    def _valid_epoch(self):
        self.model.eval()
        print("-------------starting eval------------------")
        loss_batch = 0
        preds_batch = []
        labels_batch = []
        batch_num = 0
        with torch.no_grad():
            for batch_data, batch_label, batch_s, batch_t in tqdm(self.dataloader['valid'], ncols=75):
                batch_data = batch_data.to(self.device)
                batch_label = batch_label.to(self.device)
                logits, mean_att = self.model(batch_data, batch_s, batch_t) 
                preds_batch.append(logits)
                labels_batch.append(batch_label)
                loss = self.criterion(batch_label, logits)
                if torch.isnan(loss):
                    continue
                loss_batch = loss_batch+loss
                batch_num = batch_num + 1
            loss_batch = loss_batch / batch_num
            preds = np.concatenate([self.sigmoid(x).cpu().numpy() for x in preds_batch],axis=0)
            valid_y = np.concatenate([x.detach().cpu().numpy() for x in labels_batch],axis=0).astype(np.int64)
            
            auc = roc_auc_score(valid_y, preds)
            print("Loss:", loss_batch, "   AUC:", auc)
        return loss_batch, auc


    def train(self):
        eval_loss_min = float('inf')
        best_model_filename = None
        for epoch in range(self.epoches):
            print("-------------starting training------------------")
            train_loss = self._train_epoch()
            print("Epoch = {}, train loss = {:.2f}".format(epoch + 1,train_loss))
            if epoch%self.args.eval_epoch==0:
                eval_loss, eval_auc = self._valid_epoch()
                if eval_loss < eval_loss_min:
                    eval_loss_min = eval_loss
                    best_model_filename = self.args.saved_dir + '%d_%.3f_p.ckpt' % (epoch + 1, eval_loss)
                    torch.save(self.model.state_dict(), best_model_filename)
                print("Epoch = {}, eval loss = {:.2f}, eval loss =  {:.2f}".format(epoch + 1, eval_loss, eval_auc))
           
        return best_model_filename
    


    def inference(self,model):
        print("-------------starting testing------------------")
        model.eval()
        loss_batch = 0
        preds_batch = []
        labels_batch = []
        batch_num = 0
        with torch.no_grad():
            for batch_data, batch_label, batch_s, batch_t in tqdm(self.dataloader['test'], ncols=75):
                batch_data = batch_data.to(self.device)
                batch_label = batch_label.to(self.device)
                pred, atten_mean = model(batch_data, batch_s, batch_t)  
                preds_batch.append(pred)
                labels_batch.append(batch_label)
                loss = self.criterion(batch_label, pred)
                loss_batch = loss_batch + loss
                batch_num = batch_num + 1
            loss_batch = loss_batch / batch_num
            preds = np.concatenate([self.sigmoid(x).cpu().numpy() for x in preds_batch],axis=0)
            test_y = np.concatenate([x.detach().cpu().numpy() for x in labels_batch],axis=0).astype(np.int64)
            auc = roc_auc_score(test_y, preds)
            print("Loss:", loss_batch, "Test AUC:", auc)
      
def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader, x_mean_g = make_dataloader(args)

    train_model = HEP(args, x_mean_g, device)
    train_model = torch.nn.DataParallel(train_model).to(device)

    save_path = args.saved_dir
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    trainer = Trainer(args,train_model,dataloader,device=device)
    saved_model = trainer.train()
    

    test_model = HEP(args, x_mean_g, device)
    test_model = torch.nn.DataParallel(test_model)
    test_model.load_state_dict(torch.load(saved_model))
    test_model = test_model.to(device)
    trainer.inference(test_model)



if __name__ == "__main__":
    main(parse_args())