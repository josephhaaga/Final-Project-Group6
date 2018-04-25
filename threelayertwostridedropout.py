import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
import time
import pandas as pd


SVHN_train = torchvision.datasets.SVHN('~/ML2FinalProject/',split='train', download=True, transform=torchvision.transforms.ToTensor())
SVHN_test = torchvision.datasets.SVHN('~/ML2FinalProject/', split='test', download=True, transform=torchvision.transforms.ToTensor())

#Separate training into training and validation
print(len(SVHN_train), len(SVHN_test))
a = list(range(len(SVHN_train)))
np.random.seed(5)
np.random.shuffle(a)
val_indices = a[0:7000]
train1_indices = a[7000:]
val = [SVHN_train[i] for i in val_indices]
train1 = [SVHN_train[i] for i in train1_indices]
print(len(val), len(train1))


def conv_train(train_data=SVHN_train, num_epochs=10, batch_size=100, val_data=None, validate=0):
    start = time.time() 
    criterion = torch.nn.CrossEntropyLoss()
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    if validate==1:
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
    for epoch in range(num_epochs):
        if epoch%10==0: print("epoch ", epoch)
        model.train()
        for data in train_loader:
            inputs, labels = data
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        if validate==1:
            model.eval()
            for i, data in enumerate(val_loader):
                inputs, labels = data
                inputs = Variable(inputs.cuda())
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                if i==0:
                    df = pd.DataFrame({'Predicted':list(preds), 'Actual': list(labels)})
                if i>0:
                    df1 = pd.DataFrame({'Predicted':list(preds), 'Actual': list(labels)})
                    df = pd.concat([df, df1])
            epoch_acc = accuracy_score(df['Actual'], df['Predicted'])
            acc1 = pd.DataFrame({'Epoch': [epoch], 'Accuracy': [epoch_acc] })
            if epoch==0:
                acc = acc1
            else:
                acc = pd.concat([acc, acc1])
            model.eval()
            for i, data in enumerate(train_loader):
                inputs, labels = data
                inputs = Variable(inputs.cuda())
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                if i==0:
                    df_train = pd.DataFrame({'Predicted':list(preds), 'Actual': list(labels)})
                if i>0:
                    df1_train = pd.DataFrame({'Predicted':list(preds), 'Actual': list(labels)})
                    df_train = pd.concat([df_train, df1_train])
            epoch_acc_train = accuracy_score(df_train['Actual'], df_train['Predicted'])
            acc1_train = pd.DataFrame({'Epoch': [epoch], 'Accuracy': [epoch_acc_train] })
            if epoch==0:
                acc_train = acc1_train
            else:
                acc_train = pd.concat([acc_train, acc1_train])
    end = time.time()
    t1 = end-start
    print str(t1) + ' seconds'
    if validate==1: 
        acc = acc.rename(columns={'Accuracy': 'Validation accuracy'}).merge(acc_train.rename(columns={'Accuracy': 'Training accuracy'}), on='Epoch').set_index('Epoch')
        #acc.plot(title='Training and validation accuracy')
        return acc 

class Flatten(torch.nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
model = torch.nn.Sequential(
        torch.nn.Conv2d(3,20,5),
        torch.nn.MaxPool2d(2, stride=2),
        torch.nn.ReLU(),
        torch.nn.Conv2d(20,50,5),
        torch.nn.MaxPool2d(2, stride=2),
        torch.nn.ReLU(),
        torch.nn.Dropout2d(0.25),
        Flatten(),
        torch.nn.Linear(1250, 500), 
        torch.nn.ReLU(),
        torch.nn.Linear(500, 10), 
        torch.nn.LogSoftmax()
        ).cuda() 
"""
optimizer =torch.optim.Adam(model.parameters(),lr=0.001)
acc = conv_train(train_data=train1, num_epochs=10, batch_size=1000, val_data=val, validate=1)
"""

def conv_test(batch_size=1000, testds=SVHN_test, report=0):
    model.eval()
    test_loader = torch.utils.data.DataLoader(testds, batch_size=batch_size, shuffle=True)
    for i,data in enumerate(test_loader):
        inputs, labels = data
        inputs = Variable(inputs.cuda())
        labels = labels.cuda()
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        if i==0:
            df = pd.DataFrame({'Predicted':list(preds), 'Actual': list(labels)})
        if i>0:
            df1 = pd.DataFrame({'Predicted':list(preds), 'Actual': list(labels)})
            df = pd.concat([df, df1])
    print(testds)
    if report==1:
        print(classification_report(df['Actual'], df['Predicted']))
    print(accuracy_score(df['Actual'], df['Predicted']))
    return df
"""
ctest_dropout = conv_test(report=1)
pd.crosstab(ctest_dropout['Actual'], ctest_dropout['Predicted'])
"""

"""
optimizer =torch.optim.Adam(model.parameters(),lr=0.001)
acc_nodropout = conv_train(train_data=train1, num_epochs=10, batch_size=1000, val_data=val, validate=1)

ctest_nodropout = conv_test(report=1)
pd.crosstab(ctest_dropout['Actual'], ctest_dropout['Predicted'])
"""

model = torch.nn.Sequential(
        torch.nn.Conv2d(3,20,5,padding=2),
        torch.nn.MaxPool2d(2, stride=2),
        torch.nn.ReLU(),
        torch.nn.Conv2d(20,40,5,padding=2),
        torch.nn.MaxPool2d(2, stride=2),
        torch.nn.ReLU(),
	torch.nn.Conv2d(40,40,5,padding=2),
        torch.nn.MaxPool2d(2, stride=2),
        torch.nn.ReLU(),
	torch.nn.Dropout2d(0.47),
        Flatten(),
	torch.nn.Linear(640, 320), 
        torch.nn.ReLU(),
        torch.nn.Linear(320, 10), 
        torch.nn.LogSoftmax()
        ).cuda() 
optimizer =torch.optim.Adam(model.parameters(),lr=0.001)
acc = conv_train(train_data=train1, num_epochs=100, batch_size=1000, val_data=val, validate=1)

print("alpha=0.001, 0.47 dropout, three layers w padding=2")
ctest_dropout = conv_test(report=1)

pd.crosstab(ctest_dropout['Actual'], ctest_dropout['Predicted'])




