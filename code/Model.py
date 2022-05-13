### YOUR CODE HERE
import os, time
import numpy as np
from Network import MyNetwork
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from tqdm import tqdm # for progress bar
from torch.optim.lr_scheduler import MultiStepLR

"""This script defines the training, validation and testing process.
"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MyModel(object):

    def __init__(self, model_configs, checkpoint_path = None):
        # test for speeding up
        cudnn.benchmark = True

        self.model_configs = model_configs
        self.network = MyNetwork(model_configs).to(device)

        self.cross_entropy = nn.CrossEntropyLoss().to(device)
        self.optimizer = torch.optim.SGD(
            self.network.parameters(), 
            lr=0.1,
            momentum=0.9,
            nesterov=True, # paper set this True
            weight_decay=5e-4,
        )
        if checkpoint_path is not None:
            try:
                # loads a model's parameter dictionary using a deserialized state_dict.  
                self.network.load_state_dict(torch.load(checkpoint_path))
            except:
                print("Checkpoint path does not exit: ", checkpoint_path)    
               
        # for learning_rate decay. by using this, we don't need to manually write a logic to decrease LR every certain epochs
        # args: optimizer, milestones: list of epoch indices, must be increase, gamma: mltiplicative factor of LR decay.
        self.scheduler = MultiStepLR(self.optimizer, milestones=[75, 150, 175], gamma=0.1)

    def train(self, training_loader, training_configs, testing_loader):
        # reuse some code from HW2 Model.py
        print('### Training... ###')
        for epoch in range(training_configs["epochs"]):
            correct = 0.
            total = 0.
            # tqdm takes iterable object and display the progress status
            training_progress = tqdm(training_loader)
            for i, (images, labels) in enumerate(training_progress):
                training_progress.set_description('Epoch ' + str(epoch))
                images, labels = images.to(device), labels.to(device)

                self.network.zero_grad()
                pred = self.network(images)
                loss = self.cross_entropy(pred, labels)
                loss.backward()

                self.optimizer.step()

                # Calculate running average of accuracy
                pred = torch.max(pred.data, 1)[1]
                total += labels.size(0)
                correct += (pred == labels.data).sum().item()
                accuracy = correct / total

                training_progress.set_postfix(
                acc='%.3f' % accuracy)

            test_acc = self.evaluate(testing_loader)
            tqdm.write('test_acc: %.3f' % (test_acc))

            self.scheduler.step()

            output = {'epoch': str(epoch), 'train_acc': str(accuracy), 'test_acc': str(test_acc)}
            print(output)

            if epoch % 10 == 0:
                name = "checkpoint_epoch" + str(epoch) + ".pt"
                path = f'./checkpoint'
                checkpoint_path = os.path.join(path,name)
                if not os.path.exists(path):
                    os.makedirs(path)
                print(path)
                torch.save(self.network.state_dict(), checkpoint_path)



    def evaluate(self, testing_loader):
        # reuse
        self.network.eval()
        correct = 0.
        total = 0.

        for images, labels in testing_loader:
            #images, labels = images.cuda(), labels.cuda() 
            images, labels = images.to(device), labels.to(device)

            with torch.no_grad():
                pred = self.network(images)

            pred = torch.max(pred.data, 1)[1]
            total += labels.size(0)
            correct += (pred == labels).sum().item()

        val_acc = correct / total
        self.network.train()
        return val_acc

    def predict_prob(self, private_test_loader):
        self.network.eval()
        pred_list = []

        for images in private_test_loader:
            images = images.cuda()

            with torch.no_grad():
                pred_list.extend(self.network(images).cpu().numpy())

        pred_linear = np.asarray(pred_list)

        # softmax to get the probablities
        pred_exp = np.exp(pred_linear)
        pred = pred_exp/np.sum(pred_exp, axis=-1, keepdims=True)
        return pred



### END CODE HERE