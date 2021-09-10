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

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(50176, 512)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x

### MNIST code originally from https://github.com/pytorch/examples/blob/master/mnist/main.py ### 
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

### compute accuracy using AccuracyCalculator from pytorch-metric-learning ###
def test(train_set, test_set, model, accuracy_calculator):
    train_embeddings, train_labels = get_all_embeddings(train_set, model)
    test_embeddings, test_labels = get_all_embeddings(test_set, model)
    train_labels = train_labels.squeeze(1)
    test_labels = test_labels.squeeze(1)
    print("Computing accuracy")
    accuracies = accuracy_calculator.get_accuracy(test_embeddings, 
                                                train_embeddings,
                                                test_labels,
                                                train_labels,
                                                False)
    print("Test set accuracy (Precision@1) = {}".format(accuracies["precision_at_1"]))


device = torch.device("cuda")

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomRotation(degrees=(0, 360)),
        transforms.RandomPerspective(distortion_scale=0.1, p=0.5),
        transforms.RandomApply(torch.nn.ModuleList([
                                     transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 0.1))])
            ,p=0.5),
        transforms.CenterCrop([30,30]),
        transforms.Resize([64,64])
    ])

batch_size = 64

dataset1 = datasets.ImageFolder('./AVM_center_data_track/contrastive',  transform=transform)
dataset2 = datasets.ImageFolder('./AVM_center_data_track/contrastive', transform=transform)
train_loader = torch.utils.data.DataLoader(dataset1, batch_size=64, shuffle=True, num_workers = 8)
test_loader = torch.utils.data.DataLoader(dataset2, batch_size=64, num_workers = 8)

model = resnet18(pretrained=True).to(device)
model.fc = nn.Identity()
optimizer = optim.Adam(model.parameters(), lr=0.01)
num_epochs = 300


### pytorch-metric-learning stuff ###
distance = distances.CosineSimilarity()
reducer = reducers.ThresholdReducer(low = 0)
loss_func = losses.TripletMarginLoss(margin = 0.5, distance = distance, reducer = reducer)
mining_func = miners.TripletMarginMiner(margin = 0.5, distance = distance, type_of_triplets = "semihard")
# accuracy_calculator = AccuracyCalculator(include = ("precision_at_1",), k = 1)
### pytorch-metric-learning stuff ###


for epoch in range(1, num_epochs+1):
    train(model, loss_func, mining_func, device, train_loader, optimizer, epoch)



state = model.state_dict()
torch.save(state,'./deep_sort_pytorch/deep_sort/deep/{}.pth'.format(args.name))