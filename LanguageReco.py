from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.linear_model import LogisticRegression
import torch
from torch import nn
import torch.nn.functional as F
from manual import GetLoader
from torch.utils.data import DataLoader
learing_rate = 1e-3
epochs = 1000
slice = 40000
device ='cpu'
dim_hidden = 128

class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.layer_hidden1 = nn.Linear(dim_hidden, dim_hidden)
        self.layer_hidden2 = nn.Linear(dim_hidden, dim_hidden)
        self.layer_hidden3 = nn.Linear(dim_hidden,  dim_out)
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        #x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.relu(self.dropout(x))
        x = self.layer_hidden1(x)
        x = self.relu(self.dropout(x))
        #x = self.layer_hidden2(x)
        #x = self.relu(self.dropout(x))
        #x = self.layer_hidden2(x)
        #x = self.relu(self.dropout(x))
        x = self.layer_hidden3(x)
        return self.softmax(x)
    
def train(data, label, global_model, device):
    optimizer = torch.optim.Adam(global_model.parameters(), lr=learing_rate)
    test_dataset = data[:slice//3, :]
    testlabel = np.squeeze(label[:slice//3, :], axis=1)
    train_dataset = data[slice//3:, :]
    trainlabel = np.squeeze(label[slice//3:, :], axis=1)
    torch_data = GetLoader(train_dataset, trainlabel)
    trainloader = DataLoader(torch_data, batch_size=64, shuffle=False, drop_last=False)
    torch_data1 = GetLoader(test_dataset, testlabel)
    testloader = DataLoader(torch_data1, batch_size=64, shuffle=False, drop_last=False, num_workers=1)
    criterion = nn.CrossEntropyLoss().to(device)
    epoch_loss = []
    iflist = []
    for epoch in range(epochs):
        batch_loss = []
        for i, data in enumerate(trainloader):
            images, labels = data[0].float().to(device), data[1].to(device).long()
            optimizer.zero_grad()
            outputs = global_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())
        loss_avg = sum(batch_loss)/len(batch_loss)
        print('Train loss:', loss_avg, epoch)
        epoch_loss.append(loss_avg)
        net = global_model.eval()
        total, correct = 0.0, 0.0
        for t, data1 in enumerate(testloader):
            images1, labels1 = data1[0].float().to(device), data1[1].to(device).long()
            outputs1 = net(images1)
            _, pred_labels = torch.max(outputs1, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels1)).item()
            total += len(labels1)
        accuracy = correct/total
        print('Validation:', accuracy, epoch)
        iflist.append(accuracy)
        plt.figure()
        plt.plot(range(len(epoch_loss)), epoch_loss)
        plt.xlabel('epochs')
        plt.ylabel('Train loss')
        plt.savefig('loss128.png')
        # testing
        plt.figure()
        plt.plot(range(len(iflist)), iflist)
        plt.xlabel('epochs')
        plt.ylabel('Test Accuracy')
        plt.savefig('test128.png')
    return epoch_loss, iflist


arrays1 = np.loadtxt("ia.csv")
arrays1 = arrays1[:slice,:]
(h1,w1) = np.shape(arrays1)
label1 = np.zeros((h1,1),float)
print('h1:',h1,w1)
arrays2 = np.loadtxt("gn.csv")
arrays2 = arrays2[:slice,:]
(h2,w2) = np.shape(arrays2)
label2 = np.ones((h2,1),float)
print('h2:',h2,w2)
arrays3 = np.loadtxt("ceb.csv")
arrays3 = arrays3[:slice,:]
(h3,w3) = np.shape(arrays3)
label3 = np.ones((h3,1),float)*2
print('h3:',h3,w3)
arrays4 = np.loadtxt("sco.csv")
arrays4 = arrays4[:slice,:]
(h4,w4) = np.shape(arrays4)
label4 = np.ones((h4,1),float)*3
print('h4:',h4,w4)
data = np.vstack((arrays1,arrays2,arrays3,arrays4))
data = (data-np.mean(data))/np.std(data)
label = np.vstack((label1,label2,label3,label4)) #y is merged label 0 and 1x_train.shape[0]
permutation = np.random.permutation(label.shape[0])
shuffled_dataset = data[permutation, :]
shuffled_labels = label[permutation, :]
global_model = MLP(26, dim_hidden, 4)
global_model.to(device)
global_model.train()
print(global_model)
epoch_loss, iflist = train(shuffled_dataset, shuffled_labels, global_model, device)
#seed = 12
#test_size = 0.3
#X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=test_size, random_state=seed)

# Training
#model= svm.LinearSVC()
#model.fit(X_train, y_train)

# Inference
#y_pred = model.predict(X_test)
#Conmatrix=confusion_matrix(y_test, y_pred, labels=[0,1,2,3])
#F_measure1=f1_score(y_test, y_pred, average='weighted')
#F_measure2=f1_score(y_test, y_pred, average='macro')
#F_measure3=f1_score(y_test, y_pred, average='micro')
#print('confusion matrix:',Conmatrix)
#print('f1 score:weighted',F_measure1)
#print('f1 score:macro',F_measure2)
#print('f1 score:micro',F_measure3)
# Evaluation
#accuracy = accuracy_score(y_pred, y_test)
#print("Accuracy: %.2f%%" % (accuracy * 100.0))
