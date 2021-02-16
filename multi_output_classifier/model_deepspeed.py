import os
import albumentations
import  matplotlib.pyplot as plt
import pandas as pd

# import tez

# from tez.datasets import ImageDataset
# from tez.callbacks import EarlyStopping

import torch
import torch.nn as nn

import torchvision

from sklearn import metrics, model_selection
# from efficientnet_pytorch import EfficientNet
from pathlib import Path
import argparse
import os

import albumentations
import pandas as pd
# import tez
import torch
import torch.nn as nn
# from efficientnet_pytorch import EfficientNet
from sklearn import metrics, model_selection, preprocessing
# from tez.callbacks import EarlyStopping, Callback
# from tez.datasets import ImageDataset
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

import numpy as np
import cv2

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image

from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import pickle


import deepspeed

from tqdm import tqdm




# =====================================================================
# Dataset                                                        =
# =====================================================================
class FashionImageDataset(Dataset):

    def __init__(self, IMAGE_PATH, DF_PATH, labels_required = True, use_features=True, aug=None, inference=False):
        """
        Args:
            IMAGE_PATH (string): Directory with all the images or vectors
            DF_PATH (string): Path to csv file
            labels_required (bool): target labels required or not
            use_features (bool): set this to false if want to use images as source instead of vectors or use_features
            aug: augumentation
            inference: set it to True when use the model for inference
        """
        self.image_dir       = IMAGE_PATH
        self.df              = pd.read_csv(DF_PATH)
       
        self.labels_required = labels_required
        self.use_features    = use_features
        self.inference       = inference

        if self.use_features:
             self.images = [str(i) + '.npy' for i in self.df.id.tolist()]
        else:
             self.images = [str(i) + '.jpg' for i in self.df.id.tolist()]

        if aug is None:
          self.aug = albumentations.Compose(
              [
                  albumentations.Normalize(
                      mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225],
                      max_pixel_value=255.0,
                      p=1.0,
                  ),
              ],
              p=1.0,
          )
        else:
          self.aug = aug

        if self.labels_required:
            self.genderLabels         = self.df.gender.tolist()
            self.masterCategoryLabels = self.df.masterCategory.tolist()
            self.subCategoryLabels    = self.df.subCategory.tolist()
            self.articleTypeLabels    = self.df.articleType.tolist()
            self.baseColourLabels     = self.df.baseColour.tolist()
            self.seasonLabels         = self.df.season.tolist()
            self.usageLabels          = self.df.usage.tolist()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        filename =self.images[idx]

        if self.use_features:
            img = np.load(os.path.join(self.image_dir, filename)).astype(np.float32)
            img = torch.from_numpy(img)
        else:
            # try:
            img = cv2.imread(os.path.join(self.image_dir, filename))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img,(256,256))
            img = torch.from_numpy(self.aug(image=img)['image']).type(torch.FloatTensor)
            img = img.permute(2, 0, 1)
            # except:
            #     print(filename)
            #     img = torch.rand((3, 256, 256)).type(torch.FloatTensor)


        if self.labels_required:
            genderLabel         = torch.tensor(self.genderLabels[idx])
            masterCategoryLabel = torch.tensor(self.masterCategoryLabels[idx])
            subCategoryLabel    = torch.tensor(self.subCategoryLabels[idx])
            articleTypeLabel    = torch.tensor(self.articleTypeLabels[idx])
            baseColourLabel     = torch.tensor(self.baseColourLabels[idx])
            seasonLabel         = torch.tensor(self.seasonLabels[idx])
            usageLabel          = torch.tensor(self.usageLabels[idx])

            return {'image': img, 'genderLabel': genderLabel, 'masterCategoryLabel': masterCategoryLabel, 'subCategoryLabel': subCategoryLabel, 
                    'articleTypeLabel': articleTypeLabel, 'baseColourLabel': baseColourLabel, 'seasonLabel': seasonLabel, 'usageLabel': usageLabel
            }

        if self.inference:
            return {'image': img }

        return {'image': img, 'filename': filename.split('.')[0]}


class FashionModel(nn.Module):
    def __init__(self, num_classes, in_features=1536, intermediate_features=512):
        super().__init__()

        # self.effnet = EfficientNet.from_pretrained("efficientnet-b3")
        # in_features = 1536
        # intermediate_features = 512
        # Layer 1
        self.bn1            = nn.BatchNorm1d(num_features=in_features)
        self.dropout1       = nn.Dropout(0.25)
        self.linear1        = nn.Linear(in_features=in_features, out_features=intermediate_features, bias=False)
        self.relu           = nn.ReLU()
        # Layer 2
        self.bn2            = nn.BatchNorm1d(num_features=intermediate_features)
        self.dropout2       = nn.Dropout(0.5)

        self.gender         = nn.Linear(intermediate_features, num_classes['gender'])
        self.masterCategory = nn.Linear(intermediate_features, num_classes['masterCategory'])
        self.subCategory    = nn.Linear(intermediate_features, num_classes['subCategory'])
        self.articleType    = nn.Linear(intermediate_features, num_classes['articleType'])
        self.baseColour     = nn.Linear(intermediate_features, num_classes['baseColour'])
        self.season         = nn.Linear(intermediate_features, num_classes['season'])
        self.usage          = nn.Linear(intermediate_features, num_classes['usage'])
       
        self.step_scheduler_after = "epoch"

    def monitor_metrics(self, outputs, targets):
        if targets is None:
            return {}
        accuracy = []
        for k,v in outputs.items():
            out  = outputs[k]
            targ = targets[k]
            # print(out)
            out  = torch.argmax(out, dim=1).cpu().detach().numpy()
            targ = targ.cpu().detach().numpy()
            accuracy.append(metrics.accuracy_score(targ, out))
        return {'accuracy': sum(accuracy)/len(accuracy)}

    def forward(self, image, genderLabel=None, masterCategoryLabel=None, subCategoryLabel=None, articleTypeLabel=None, baseColourLabel=None, seasonLabel=None, usageLabel=None):
        batch_size, _ = image.shape

        x = self.linear1(self.dropout1(self.bn1(image)))
        x = self.relu(x)
        x = self.dropout2(self.bn2(x))

        targets = {}
        if genderLabel is None:
            targets = None
        else:
            targets['gender']         = genderLabel
            targets['masterCategory'] = masterCategoryLabel
            targets['subCategory']    = subCategoryLabel
            targets['articleType']    = articleTypeLabel
            targets['baseColour']     = baseColourLabel
            targets['season']         = seasonLabel
            targets['usage']          = usageLabel
        outputs                   = {}
        outputs["gender"]         = self.gender(x)
        outputs["masterCategory"] = self.masterCategory(x)
        outputs["subCategory"]    = self.subCategory(x)
        outputs["articleType"]    = self.articleType(x)
        outputs["baseColour"]     = self.baseColour(x)
        outputs["season"]         = self.season(x)
        outputs["usage"]          = self.usage(x)

        if targets is not None:
            loss = []
            for k,v in outputs.items():
                loss.append(nn.CrossEntropyLoss()(outputs[k], targets[k]))
            loss = sum(loss)
            metrics = self.monitor_metrics(outputs, targets)
            return outputs, loss, metrics
        return outputs, None, None
    
    def extract_features(self, image):
        batch_size, _ = image.shape

        features = self.linear1(self.dropout1(self.bn1(image)))

        return features


def add_argument():

    parser = argparse.ArgumentParser(description='CIFAR')

    #data
    # cuda
    parser.add_argument('--with_cuda',
                        default=True,
                        action='store_true',
                        help='use CPU in case there\'s no GPU support')
    parser.add_argument('--use_ema',
                        default=False,
                        action='store_true',
                        help='whether use exponential moving average')

    # train
    parser.add_argument('-b',
                        '--batch_size',
                        default=32,
                        type=int,
                        help='mini-batch size (default: 32)')
    parser.add_argument('-e',
                        '--epochs',
                        default=5,
                        type=int,
                        help='number of total epochs (default: 30)')
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')

    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()

    return args


if __name__ == "__main__":


    dfx = pd.read_csv('df_final.csv')
    class_dict = dict([[i, dfx[i].nunique()] for i in dfx.columns.tolist()[1:]])
    train_dataset = FashionImageDataset(IMAGE_PATH='./big_model_vectors', DF_PATH= 'train.csv')
    val_dataset   = FashionImageDataset(IMAGE_PATH='./big_model_vectors', DF_PATH= 'val.csv')


    train_dl = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)
    val_dl = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=8)

    model = FashionModel(num_classes=class_dict, in_features=512, intermediate_features=128)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    args = add_argument()

    model_engine, optimizer, trainloader, __ = deepspeed.initialize(args=args, 
                model=model, model_parameters=parameters, training_data=trainset)

    for epoch in range(20):  # loop over the dataset multiple times
        tk0 = tqdm(train_dl, total=len(train_dl))
        losses = []
        monitor = []
        
        model.train()
        
        for i, data in enumerate(tk0):
            # get the inputs; data is a list of [inputs, labels]
            for key, value in data.items():
                data[key] = value.to(model_engine.local_rank)

            # zero the parameter gradients
            # optimizer.zero_grad()

            # forward + backward + optimize
            outputs, loss, metric = model_engine(**data)
            
            losses.append(loss.item())
            monitor.append(metric['accuracy'])
            model_engine.backward(loss)
            model_engine.step()
            # loss.backward()
            # optimizer.step()
            tk0.set_postfix(loss=round(sum(losses)/len(losses), 2), stage="train", accuracy=round(sum(monitor)/len(monitor), 3))
            tk0.close()
            # scheduler.step()
            # tk0 = tqdm(val_dl, total=len(val_dl))
            # val_losses = []
            # val_monitor = []
            # model.eval()
            # for i, data in enumerate(tk0):
            #     for key, value in data.items():
            #         data[key] = value.to('cuda')
            #     optimizer.zero_grad()
            #     with torch.no_grad():
            #         outputs, loss, metric = model(**data)
            #     val_losses.append(loss.item())
            #     val_monitor.append(metric['accuracy'])
            #     tk0.set_postfix(loss=round(sum(val_losses)/len(val_losses), 2), stage="valid", accuracy=round(sum(val_monitor)/len(val_monitor), 3))
            # tk0.close()
            # curr_loss = round(sum(val_losses)/len(val_losses))
            # if curr_loss < best_loss:
            #     model_dict = {}
            #     model_dict["state_dict"] = model.state_dict()
            #     # model_dict["optimizer"] = optimizer.state_dict()
            #     # model_dict["scheduler"] = scheduler.state_dict()
            #     # model_dict["epoch"] = epoch
            #     # model_dict["fp16"] = False
            #     torch.save(model_dict, 'model_best.pt')
            #     print(f"Model Saved | Loss impoved from {best_loss} -----> {curr_loss}")
            #     best_loss = curr_loss
            #     counter = 0
            # else:
            #     counter += 1
            # if counter == 5:
            #     print("Model not improved from 5 epochs...............")
            #     print("Training Finished..................")
            #     break
    print('Finished Training')



