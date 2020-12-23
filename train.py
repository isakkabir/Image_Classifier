# Imports here
import torch
import argparse 
from torch import nn, optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
from PIL import Image
import json
import time 

#if __name__ == '__main__':
parser = argparse.ArgumentParser() 
parser.add_argument("--data_dir", type = str)
parser.add_argument("--valid_dir", type = str)
parser.add_argument("--save_dir", type = str, help = " Directory to save your model") 
parser.add_argument("--gpu", help = " gpu command, choose 'cuda' or 'cpu'", action = 'store_true', default = False)
parser.add_argument("--structure", type = str, help = "Choose from 'vgg', densenet or 'alexnet'", default = 'vgg16') 
parser.add_argument("--hidden_layer1", type = int, help = "Integer number", default = 120) 
parser.add_argument("--lr", type = float, help = "Float number", default = 0.001) 
parser.add_argument("--epochs", type = int, help = " Integer number", default = 1) 
args = parser.parse_args()


print('parameters are as follows.')
print('data_directory: {}',args.data_dir)
print('save_dir: {}'.format(args.save_dir))
print('valid_dir: {}'.format(args.valid_dir))
print('selected_model: {}'.format(args.structure))
print('hidden units: {}'.format(args.hidden_layer1))
print('learning rate: {}'.format(args.lr))
print('number of epochs: {}'.format(args.epochs))
print('cuda or cpu: {}'.format(args.gpu))
    
def data_load(data_dir, valid_dir):
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    
    validation_transforms = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], 
                                                                 [0.229, 0.224, 0.225])])
    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
    validation_data = datasets.ImageFolder(valid_dir + '/valid', transform=validation_transforms)
    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validationloader = torch.utils.data.DataLoader(validation_data, batch_size=32)
    return trainloader, validationloader

import json
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
 ######## Pre model ############
structures = {"vgg16":25088,"densenet121" : 1024,"alexnet" : 9216 }

def pre_model(structure, hidden_layer1, lr):
   dropout=0.5

   if structure == 'vgg16':
       print("check")
       model = models.vgg16(pretrained=True)
   elif structure == 'densenet121':
       model = models.densenet121(pretrained=True)
   elif structure == 'alexnet':
       model = models.alexnet(pretrained = True)
   print(args.structure)


   for param in model.parameters():
       param.requires_grad = False

   from collections import OrderedDict
   classifier = nn.Sequential(OrderedDict([
   ('dropout',nn.Dropout(dropout)),
   ('inputs', nn.Linear(structures[structure], hidden_layer1)),
   ('relu1', nn.ReLU()),
   ('hidden_layer1', nn.Linear(hidden_layer1, 90)),
   ('relu2',nn.ReLU()),
   ('hidden_layer2',nn.Linear(90,80)),
   ('relu3',nn.ReLU()),
   ('hidden_layer3',nn.Linear(80,102)),
   ('output', nn.LogSoftmax(dim=1))]))

   model.classifier = classifier
   criterion = nn.NLLLoss()
   optimizer = optim.Adam(model.classifier.parameters(), lr)
   model.cuda()

   return model , optimizer ,criterion

##########          Training         #################
def training_deep_network(model,trainloader,validationloader,epochs,gpu):
  print_every = 5
  steps = 0
  loss_show=[]
  model.cuda()
  #start = time.time()
  print('Training started')
  # change to cuda
  #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  if(gpu):
    device = 'cuda'
  else:
      'cpu'
  for e in range(epochs):
      running_loss = 0
      for ii, (inputs, labels) in enumerate(trainloader):
          steps += 1
          inputs,labels = inputs.to(device), labels.to(device)
          optimizer.zero_grad()
          # Forward and backward passes
          outputs = model.forward(inputs)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()
          if steps % print_every == 0:
                model.eval()
                vlost = 0
                accuracy=0
                
                for ii, (inputs2,labels2) in enumerate(validationloader):
                    optimizer.zero_grad()
                
                    inputs2, labels2 = inputs2.to(device) , labels2.to(device)
                    model.to(device)
                    with torch.no_grad():    
                        outputs = model.forward(inputs2)
                        vlost = criterion(outputs,labels2)
                        ps = torch.exp(outputs).data
                        equality = (labels2.data == ps.max(1)[1])
                        accuracy += equality.type_as(torch.FloatTensor()).mean()
                    
                vlost = vlost / len(validationloader)
                accuracy = accuracy /len(validationloader)
            
                print("Epoch: {}/{}... ".format(e, epochs),
                      "Loss: {:0.4f}".format(running_loss/print_every),
                      "Validation Lost {:0.4f}".format(vlost),
                      "Accuracy: {:0.4f}".format(accuracy))
          running_loss = 0

#print("\nTotal time: {:.0f}m {:.0f}s".format(time_elapsed//60, time_elapsed % 60))
    
##########          GPU         #################
def gpu(gpu):
   '''
   This function enables user to switch gpu and cpu
   '''
   # switch to GPU or CPU
   if gpu == 'cuda' and torch.cuda.is_available():
       device = torch.device('cuda')
   else:
       device = torch.device('cpu')        
        
# TODO: Save the checkpoint 
def save_model(model, train_data, filename):
    model.to("cpu")
    model.class_to_idx = cat_to_name
    #model.class_to_idx = train_data.class_to_idx
    torch.save({'structure' :'vgg16',
                'state_dict':model.state_dict(),
                'class_to_idx':model.class_to_idx, 
                'optimizer_state_dict': optimizer.state_dict()},
                'checkpoint.pth')


       
# load training data and valid data
train_data, valid_data = data_load(args.data_dir, args.valid_dir)
# define which pre_trained model to use and number of hidden layer
premodel,optimizer,criterion = pre_model(args.structure, args.hidden_layer1, args.lr)

#save model 
save_model(premodel, train_data, 'checkpoint.pth')
# choose gpu or cpu to train the model
device = gpu(args.gpu)
# adjust learning rate
learning_rate = args.lr
# adjust number of epochs to run
epochs = args.epochs
# get the pretrained model archtecture vgg or alexnet
structure = args.structure
# get the save directory
save_dir = args.save_dir
# train the model with the above defined parameters
training_deep_network(premodel, train_data, valid_data, epochs, args.gpu)
