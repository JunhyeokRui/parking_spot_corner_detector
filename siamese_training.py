from pytorch_metric_learning import losses, miners, distances, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
### MNIST code originally from https://github.com/pytorch/examples/blob/master/mnist/main.py ### 
from torchvision import datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from torchvision.models import resnet18
import argparse

parser = argparse.ArgumentParser(description='corner_annotation')
parser.add_argument('--epoch', type=int, default=300)
parser.add_argument('--margin', type=float, default=0.3)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--lr', default=0.01,type=float)
parser.add_argument('--name', required=True)
parser.add_argument('--crop-size', default=20, type=int)
parser.add_argument('--rot-aug', default=False, action='store_true')
parser.add_argument('--gaussian-aug', default=False, action='store_true')
parser.add_argument('--resize', default=0)


args = parser.parse_args()

def train(model, loss_func, mining_func, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        embeddings = model(data)
        indices_tuple = mining_func(embeddings, labels)
        loss = loss_func(embeddings, labels, indices_tuple)
        loss.backward()
        optimizer.step()
        if batch_idx % 20 == 0:
            print("Epoch {} Iteration {}: Loss = {}, Number of mined triplets = {}".format(epoch, batch_idx, loss, mining_func.num_triplets))

### convenient function from pytorch-metric-learning ###
def get_all_embeddings(dataset, model):
    tester = testers.BaseTester()
    return tester.get_all_embeddings(dataset, model)


if __name__=="__main__":
    device = torch.device("cuda")

    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomRotation(degrees=(0, 360)),
            transforms.RandomApply(torch.nn.ModuleList([
                                         transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 0.1))])
                ,p=0.5),
            transforms.CenterCrop([30,30]),
            transforms.Resize([args.resize,args.resize])
        ])

    batch_size = 64

    dataset = datasets.ImageFolder('./AVM_center_data_track/contrastive',  transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers = 8)

    model = resnet18(pretrained=True).to(device)
    model.fc = nn.Identity()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    num_epochs = args.epoch


    ### pytorch-metric-learning stuff ###
    distance = distances.CosineSimilarity()
    reducer = reducers.ThresholdReducer(low = 0)
    loss_func = losses.TripletMarginLoss(margin = args.margin, distance = distance, reducer = reducer)
    mining_func = miners.TripletMarginMiner(margin = args.margin, distance = distance, type_of_triplets = "semihard")


    for epoch in range(1, num_epochs+1):
        train(model, loss_func, mining_func, device, train_loader, optimizer, epoch)

    state = model.state_dict()
    torch.save(state,'./deep_sort_pytorch/deep_sort/deep/{}.pth'.format(args.name))