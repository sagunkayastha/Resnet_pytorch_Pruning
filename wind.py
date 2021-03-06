from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import sys
import pickle
# plt.ion()   # interactive mode

# Just normalization for validation
Resume = True
num_epochs = 25
lr = 0.001
batch_size = 64


data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(28),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'val': transforms.Compose([
        transforms.Resize(28),
        transforms.CenterCrop(28),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
}

data_dir = '../dataset2/'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

# print(class_names)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_model(model, criterion, optimizer, scheduler, num_epochs=25,ep=0,best_acc=0):
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        epoch=ep
        print('Epoch {}/{}'.format(epoch+1, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            samples=0
            i=0
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                
                inputs = inputs.cuda()
                labels = labels.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()

					
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                samples += inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                if i%10 ==0:
                    step_loss=float(running_loss) / float(samples), 
                    step_accuracy=float(running_corrects) / float(samples)
                    
                    sys.stdout.write("Accuracy in epoch : %d - Step : %d , loss = %f , Accuracu = %f \r" % (epoch+1, i, step_loss[0], step_accuracy) )
                    sys.stdout.flush()
                    
                    
                i+=1

            Obj_to_save = {'epoch' : epoch, 'best_acc' : best_acc, 'lr': lr }
            f = open('params.pkl', 'w')   # Pickle file is newly created where foo1.py is
            pickle.dump(Obj_to_save, f)          # dump data to f
            f.close()                 
            
            
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, "checkpoints/ckpt.pth")

        
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

model_ft = models.resnet50(pretrained=False)
num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).

model_ft.fc =    torch.nn.Linear(
        in_features=num_ftrs,
        out_features=31
    )

if Resume == True:
    model_ft.load_state_dict(torch.load("checkpoints/ckpt.pth"))
    
    f = open('params.pkl', 'rb')   # 'r' for reading; can be omitted
    config = pickle.load(f)         # load file content as mydict
    print(config)
    f.close() 
    
    
model_ft = model_ft.cuda()

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.Adam(model_ft.parameters(), lr=lr)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)

# model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
#                        num_epochs=50,ep=mydict['epoch'],best_acc=config['best_acc'])

# def run():
#     torch.multiprocessing.freeze_support()
#     print('loop')

# if __name__ == '__main__':
#     run()
#     model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
#                        num_epochs=25,ep=mydict['epoch'],best_acc=config['best_acc'])
    