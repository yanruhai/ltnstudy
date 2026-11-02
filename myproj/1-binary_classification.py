import torch
import matplotlib.pyplot as plt
nr_samples = 100
dataset = torch.rand((nr_samples, 2))
labels_dataset = torch.sum(torch.square(dataset - torch.tensor([.5, .5])), dim=1) < .09
plt.figure(figsize=(4,4))
plt.scatter(dataset[labels_dataset][:,0],dataset[labels_dataset][:,1],label='A')
plt.scatter(dataset[torch.logical_not(labels_dataset)][:,0],dataset[torch.logical_not(labels_dataset)][:,1],label='~A')
plt.title("Ground truth")
plt.legend()
#plt.show()

import ltn
# we define predicate A
class ModelA(torch.nn.Module):
    def __init__(self):
        super(ModelA, self).__init__()
        self.sigmoid = torch.nn.Sigmoid()
        self.layer1 = torch.nn.Linear(2, 16)
        self.layer2 = torch.nn.Linear(16, 16)
        self.layer3 = torch.nn.Linear(16, 1)
        self.elu = torch.nn.ELU()

    def forward(self, x):
        x = self.elu(self.layer1(x))
        x = self.elu(self.layer2(x))
        return self.sigmoid(self.layer3(x))


A = ltn.Predicate(ModelA())

# we define the connectives, quantifiers, and the SatAgg
Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
SatAgg = ltn.fuzzy_ops.SatAgg()

from sklearn.metrics import accuracy_score
import numpy as np

# this is a standard PyTorch DataLoader to load the dataset for the training and testing of the model
class DataLoader(object):
    def __init__(self,
                 data,
                 labels,
                 batch_size=1,
                 shuffle=True):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return int(np.ceil(self.data.shape[0] / self.batch_size))

    def __iter__(self):
        n = self.data.shape[0]
        idxlist = list(range(n))
        if self.shuffle:
            np.random.shuffle(idxlist)

        for _, start_idx in enumerate(range(0, n, self.batch_size)):
            end_idx = min(start_idx + self.batch_size, n)
            data = self.data[idxlist[start_idx:end_idx]]
            labels = self.labels[idxlist[start_idx:end_idx]]

            yield data, labels


# define metrics for evaluation of the model

# it computes the overall satisfaction level on the knowledge base using the given data loader (train or test)
def compute_sat_level(loader):
    mean_sat = 0
    for data, labels in loader:
        x_A = ltn.Variable("x_A", data[torch.nonzero(labels)])  # positive examples
        x_not_A = ltn.Variable("x_not_A",
                               data[torch.nonzero(torch.logical_not(labels))])  # negative examples
        mean_sat += SatAgg(
            Forall(x_A, A(x_A)),
            Forall(x_not_A, Not(A(x_not_A)))
        )
    mean_sat /= len(loader)
    return mean_sat

# it computes the overall accuracy of the predictions of the trained model using the given data loader
# (train or test)
def compute_accuracy(loader):
    mean_accuracy = 0.0
    for data, labels in loader:
        predictions = A.model(data).detach().numpy()
        predictions = np.where(predictions > 0.5, 1., 0.).flatten()
        mean_accuracy += accuracy_score(labels, predictions)

    return mean_accuracy / len(loader)

# create train and test loader, 50 points each
# batch size is 64, meaning there is only one batch for epoch
train_loader = DataLoader(dataset[:50], labels_dataset[:50], 64, True)
test_loader = DataLoader(dataset[50:], labels_dataset[50:], 64, False)