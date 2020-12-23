import torch
import argparse 
from torch import nn, optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
from PIL import Image
import json
import time 

parser = argparse.ArgumentParser() 
parser.add_argument("--test_dir", type= str)
parser.add_argument('--top_k', type= int, help="return top K most likely classes", default=5)
parser.add_argument('--category_names', type= str, help="choose a file for categories to real names", default="cat_to_name.json")
parser.add_argument("--structure", type = str, help = "Choose from 'vgg', densenet or 'alexnet'", default = 'vgg16') 
parser.add_argument("--gpu", help = " gpu command", action = 'store_true', default = False) 
parser.add_argument("--image_path", type = str, help = "Directory for single image path", default = "/home/workspace/ImageClassifier/flowers/test/15/image_06351.jpg")
args = parser.parse_args()

print('parameters are as follows.')
print('test_dir: {}',args.test_dir)
print('top_k: {}'.format(args.top_k))
print('category_names: {}'.format(args.category_names))
print('selected_model: {}'.format(args.structure))
print('cuda or cpu: {}'.format(args.gpu))
print('image_path: {}'.format(args.image_path))



def data_load(test_dir):
    test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])


    # TODO: Load the datasets with ImageFolder
    test_data = datasets.ImageFolder(test_dir + '/test', transform = test_transforms)
   
    # TODO: Using the image datasets and the trainforms, define the dataloaders
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
    return testloader

import json
with open('/home/workspace/ImageClassifier/cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
 ######## Pre model ############
structures = {"vgg16":25088,"densenet121" : 1024,"alexnet" : 9216 }

def pre_model(structure, hidden_layer1, lr):
    dropout=0.5
    if structure == 'vgg16':
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

    return model , optimizer, criterion

    
def load_model(path):
    checkpoint = torch.load('checkpoint.pth')
    structure = checkpoint['structure']
    #hidden_layer1 = checkpoint['hidden_layer1']
    model,_,_ = pre_model(structure, hidden_layer1 = 10, lr = 0.01)
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    return model



def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Open the image & resize 
    im_pil = Image.open(image)
    im_pil = im_pil.resize((256,256))
    
    #Cropping
    value = 0.5*(256-224)
    im_pil = im_pil.crop((value,value,256-value,256-value))
    im_pil = np.array(im_pil)/255

    #Normalizing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    im_pil = (im_pil - mean) / std
    im_pil = im_pil.transpose(2, 0, 1)
    return torch.from_numpy(im_pil)


def predict(model, image_path, topk):
    ''' calculate the topk prediction of the given image-file and 
    return the probabilities, lables and resolved flower-names
     '''
    # ------ load image data -----------
    processed_image = process_image(image_path)
    # ----------------------------------
    print("get prediction ... ", end="")
    model.eval()
    # prepare image tensor for prediction
    torch_image = torch.from_numpy(np.expand_dims(processed_image,axis = 0)).type(torch.FloatTensor).to("cuda")
    with torch.no_grad():
        outputs = model.forward(torch_image.cuda())
    
    probabilities = torch.exp(outputs).data
    
    # getting the topk (=5) probabilites and indexes
    prob = torch.topk(probabilities, topk)[0].tolist()[0] # probabilities
    index = torch.topk(probabilities, topk)[1].tolist()[0] # index
    ind = []
    for i in range(len(model.class_to_idx.items())):
        ind.append(list(model.class_to_idx.items())[i][0])
    #print(ind)
    # transfer index to label
    label = []
    for i in range(topk):
        label.append(ind[index[i]])
    return prob, label
    
    

# load test data
test_data = data_load(args.test_dir)

# load model 
model = load_model('checkpoint.pth') 

# get the pretrained model archtecture vgg or alexnet
structure = args.structure

#Select an image 
img= ("/home/workspace/ImageClassifier/flowers/test/15/image_06351.jpg")
prob, label = predict(model, img, args.top_k)
print("check 2nd")
for i in range(args.top_k):
      print(" {} with {:.3f} is {} for flower {} ".format(i+1, prob[i], label[i], cat_to_name[str(label[i])]))