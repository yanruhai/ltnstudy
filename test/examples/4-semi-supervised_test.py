import torch
import pandas as pd
import torchvision

def get_mnist_dataset_for_digits_addition(single_digit=True):
    """
    It prepares the dataset for the MNIST single digit or multi digits addition example of the LTN paper.

    :param single_digit: whether the dataset has to be generated for the single digit or multi digits example (please,
    carefully read the examples in the paper to understand the differences between the two).
    :return: a tuple of two elements. The first element is the training set, while the second element is the test set.
    Both training set and test set are lists that contain the following information:
        1. a list [left_operands, right_operands], where left_operands is a list of MNIST images that are used as the
        left operand of the addition, while right_operands is a list of MNIST images that are used as the right operand
        of the addition;
        2. a list containing the summation of the labels of the images contained in the list at point 1. The label of
        the left operand is added to the label of the right operand, and the target label is generated. This represents
        the target of the digits addition task.
    Note that this is the output of the process for the single digit case. In the multi digits case the list at point
    1 will have 4 elements since in the multi digits case four digits are involved in each addition (two digits
    represent the first operand and two digits the second operand).
    """
    if single_digit:
        n_train_examples = 30000
        n_test_examples = 5000
        n_operands = 2
    else:
        n_train_examples = 15000
        n_test_examples = 2500
        n_operands = 4

    mnist_train = torchvision.datasets.MNIST("./datasets/", train=True, download=True,
                                             transform=torchvision.transforms.ToTensor())
    mnist_test = torchvision.datasets.MNIST("./datasets/", train=False, download=True,
                                            transform=torchvision.transforms.ToTensor())

    train_imgs, train_labels, test_imgs, test_labels = mnist_train.data, mnist_train.targets, \
                                                       mnist_test.data, mnist_test.targets

    train_imgs, test_imgs = train_imgs / 255.0, test_imgs / 255.0

    train_imgs, test_imgs = torch.unsqueeze(train_imgs, 1), torch.unsqueeze(test_imgs, 1)#增加channel，Prepare for cnn operate late on

    imgs_operand_train = [train_imgs[i * n_train_examples:i * n_train_examples + n_train_examples]
                          for i in range(n_operands)]
    labels_operand_train = [train_labels[i * n_train_examples:i * n_train_examples + n_train_examples]
                            for i in range(n_operands)]

    imgs_operand_test = [test_imgs[i * n_test_examples:i * n_test_examples + n_test_examples]
                         for i in range(n_operands)]
    labels_operand_test = [test_labels[i * n_test_examples:i * n_test_examples + n_test_examples]
                           for i in range(n_operands)]

    if single_digit:
        label_addition_train = labels_operand_train[0] + labels_operand_train[1]
        label_addition_test = labels_operand_test[0] + labels_operand_test[1]
    else:
        label_addition_train = 10 * labels_operand_train[0] + labels_operand_train[1] + \
                               10 * labels_operand_train[2] + labels_operand_train[3]

        label_addition_test = 10 * labels_operand_test[0] + labels_operand_test[1] + \
                              10 * labels_operand_test[2] + labels_operand_test[3]

    train_set = [torch.stack(imgs_operand_train, dim=1), label_addition_train]#运行后的维度是(30000,2,1,28,28),其中的2是变量数
    test_set = [torch.stack(imgs_operand_test, dim=1), label_addition_test]

    return train_set, test_set

# single digit dataset
single_d_train_set, single_d_test_set = get_mnist_dataset_for_digits_addition(single_digit=True)
# multi digit dataset
multi_d_train_set, multi_d_test_set = get_mnist_dataset_for_digits_addition(single_digit=False)

import matplotlib.pyplot as plt
first_example_images = single_d_train_set[0][0]
first_example_label = single_d_train_set[1][0]

print("Operands are displayed in the following images:")
fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
imgplot = plt.imshow(first_example_images[0].permute(1, 2, 0))
ax.set_title('First operand')
ax = fig.add_subplot(1, 2, 2)
imgplot = plt.imshow(first_example_images[1].permute(1, 2, 0))
ax.set_title('Second operand')
plt.show()
print("The target label (sum) for these operands is: %d" % first_example_label.item())

from torch.nn.init import xavier_uniform_, normal_, kaiming_uniform_
import ltn

# we define the variables
d_1 = ltn.Variable("d_1", torch.tensor(range(10)))
d_2 = ltn.Variable("d_2", torch.tensor(range(10)))
# these are used only in the multi digit case
d_3 = ltn.Variable("d_3", torch.tensor(range(10)))
d_4 = ltn.Variable("d_4", torch.tensor(range(10)))

# we define predicate digit
class MNISTConv(torch.nn.Module):
    """
    CNN that returns linear embeddings for MNIST images. It is used in the single digit and multi digits addition
    examples of the LTN paper.
    Args:
        conv_channels_sizes: tuple containing the number of channels of the convolutional layers of the model. The first
        element of the tuple must be the number of input channels of the first conv layer, while the last element
        of the tuple must be the number of output channels of the last conv layer. Specifically, the number of conv
        layers constructed is equal to `len(conv_channels_sizes) - 1`;
        kernel_sizes: tuple containing the sizes of the kernels used in the conv layers of the architecture;
        linear_layers_sizes: tuple containing the sizes of the linear layers used as the final layers of the
        architecture. The first element of the tuple must be the number of features in input to the first linear layer,
        while the last element of the tuple must be the number of output features of the last linear layer. Specifically,
        the number of layers constructed is equal to `len(linear_layers_sizes) - 1`.
    """
    def __init__(self, conv_channels_sizes=(1, 6, 16), kernel_sizes=(5, 5), linear_layers_sizes=(256, 100)):
        super(MNISTConv, self).__init__()
        self.conv_layers = torch.nn.ModuleList([torch.nn.Conv2d(conv_channels_sizes[i - 1], conv_channels_sizes[i],
                                                                kernel_sizes[i - 1])
                                                  for i in range(1, len(conv_channels_sizes))])
        self.relu = torch.nn.ReLU()  # relu is used as activation for the conv layers
        self.tanh = torch.nn.Tanh()  # tanh is used as activation for the linear layers
        self.maxpool = torch.nn.MaxPool2d((2, 2))
        self.linear_layers = torch.nn.ModuleList([torch.nn.Linear(linear_layers_sizes[i - 1], linear_layers_sizes[i])
                                                  for i in range(1, len(linear_layers_sizes))])
        self.batch_norm_layers = torch.nn.ModuleList([torch.nn.BatchNorm1d(linear_layers_sizes[i])
                                                      for i in range(1, len(linear_layers_sizes))])

        self.init_weights()

    def forward(self, x):
        for conv in self.conv_layers:
            x = self.relu(conv(x))
            x = self.maxpool(x)
        x = torch.flatten(x, start_dim=1)
        for i in range(len(self.linear_layers)):
            x = self.tanh(self.batch_norm_layers[i](self.linear_layers[i](x)))
        return x

    def init_weights(self):
        r"""Initialize the weights of the network.
        Weights of conv layers are initialized with the :py:func:`torch.nn.init.kaiming_uniform_` initializer,
        weights of linear layers are initialized with the :py:func:`torch.nn.init.xavier_uniform_` initializer,
        while biases are initialized with the :py:func:`torch.nn.init.normal_` initializer.
        """
        for layer in self.conv_layers:
            kaiming_uniform_(layer.weight)
            normal_(layer.bias)

        for layer in self.linear_layers:
            xavier_uniform_(layer.weight)
            normal_(layer.bias)


class SingleDigitClassifier(torch.nn.Module):
    """
    Model classifying one MNIST digit image into 10 possible classes. It outputs the logits, so it is not a normalized
    output. It has a convolutional part in the initial layers of the architecture and a linear part in the last
    layers of the architecture. To build the convolutional part a pre-configured convolutional model is used.
    Args:
        layers_sizes: tuple containing the sizes of the linear layers used as the final layers of the
        architecture. The first element of the tuple must be the number of features in input to the first linear layer,
        while the last element of the tuple must be the number of output features of the last linear layer. Specifically,
        the number of layers constructed is equal to `len(layers_sizes) - 1`.
    """
    def __init__(self, layers_sizes=(100, 84, 10)):
        super(SingleDigitClassifier, self).__init__()
        self.mnistconv = MNISTConv()  # this is the convolutional part of the architecture
        self.tanh = torch.nn.Tanh()  # tanh is used as activation for the linear layers
        self.linear_layers = torch.nn.ModuleList([torch.nn.Linear(layers_sizes[i - 1], layers_sizes[i])
                                                  for i in range(1, len(layers_sizes))])
        self.batch_norm_layers = torch.nn.ModuleList([torch.nn.BatchNorm1d(layers_sizes[i])
                                                      for i in range(1, len(layers_sizes))])
        self.init_weights()

    def forward(self, x):
        x = self.mnistconv(x)
        for i in range(len(self.linear_layers) - 1):
            x = self.tanh(self.batch_norm_layers[i](self.linear_layers[i](x)))
        return self.linear_layers[-1](x)  # in the last layer a sigmoid or a softmax has to be applied

    def init_weights(self):
        """Initialize the weights of the linear layers of the network.
        Weights are initialized with the :py:func:`torch.nn.init.xavier_uniform_` initializer,
        while biases are initialized with the :py:func:`torch.nn.init.normal_` initializer.
        """
        for layer in self.linear_layers:
            xavier_uniform_(layer.weight)
            normal_(layer.bias)


class LogitsToPredicate(torch.nn.Module):
    """
    This model has inside a logits model, that is a model which compute logits for the classes given an input example x.
    The idea of this model is to keep logits and probabilities separated. The logits model returns the logits for an example,
    while this model returns the probabilities given the logits model.

    In particular, it takes as input an example x and a class label d. It applies the logits model to x to get the logits.
    Then, it applies a softmax function to get the probabilities per classes. Finally, it returns only the probability related
    to the given class d.
    """
    def __init__(self, logits_model):
        super(LogitsToPredicate, self).__init__()
        self.logits_model = logits_model
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x, d):
        logits = self.logits_model(x)
        probs = self.softmax(logits)
        out = torch.gather(probs, 1, d)#(320，10),(320,1)
        #torch.gather 会沿着 dim=1 这个维度，去 probs 里查找 d 里指定的索引，然后把对应位置的概率值提取出来。
        return out

# single digit
cnn_s_d = SingleDigitClassifier()
Digit_s_d = ltn.Predicate(LogitsToPredicate(cnn_s_d)).to(ltn.device)
# multi digit
cnn_m_d = SingleDigitClassifier()
Digit_m_d = ltn.Predicate(LogitsToPredicate(cnn_m_d)).to(ltn.device)

# we define the connectives, quantifiers, and the SatAgg
And = ltn.Connective(ltn.fuzzy_ops.AndProd())
Exists = ltn.Quantifier(ltn.fuzzy_ops.AggregPMean(p=2), quantifier="e")
Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
SatAgg = ltn.fuzzy_ops.SatAgg()

import numpy as np

# this is a standard PyTorch DataLoader to load the dataset for the training and testing of the model
class DataLoader(object):
    def __init__(self,
                 fold,
                 batch_size=1,
                 shuffle=True):
        self.fold = fold
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return int(np.ceil(self.fold[0].shape[0] / self.batch_size))

    def __iter__(self):
        n = self.fold[0].shape[0]
        idxlist = list(range(n))
        if self.shuffle:
            np.random.shuffle(idxlist)

        for _, start_idx in enumerate(range(0, n, self.batch_size)):
            end_idx = min(start_idx + self.batch_size, n)
            digits = self.fold[0][idxlist[start_idx:end_idx]]
            addition_labels = self.fold[1][idxlist[start_idx:end_idx]]

            yield digits, addition_labels

# create train and test loader
single_train_loader = DataLoader(single_d_train_set, 32, shuffle=True)
single_test_loader = DataLoader(single_d_test_set, 32, shuffle=False)
multi_d_train_loader = DataLoader(multi_d_train_set, 32, shuffle=True)
multi_d_test_loader = DataLoader(multi_d_test_set, 32, shuffle=False)

optimizer = torch.optim.Adam(Digit_s_d.parameters(), lr=0.001)

for epoch in range(20):
    # scheduling of the parameter p for the existential quantifier as described in the LTN paper
    if epoch in range(0, 4):
        p = 1
    if epoch in range(4, 8):
        p = 2
    if epoch in range(8, 12):
        p = 4
    if epoch in range(12, 20):
        p = 6
    train_loss, test_loss = 0.0, 0.0
    train_sat, test_sat = 0.0, 0.0
    train_acc, test_acc = 0.0, 0.0
    # train step
    for batch_idx, (operand_images, addition_label) in enumerate(single_train_loader):
        optimizer.zero_grad()
        # ground variables with current batch data
        images_x = ltn.Variable("x", operand_images[:, 0])
        images_y = ltn.Variable("y", operand_images[:, 1])
        labels_z = ltn.Variable("z", addition_label)
        sat_agg = Forall(
            ltn.diag(images_x, images_y, labels_z),
            Exists(
                [d_1, d_2],
                And(Digit_s_d(images_x, d_1), Digit_s_d(images_y, d_2)),
                cond_vars=[d_1, d_2, labels_z],
                cond_fn=lambda d1, d2, z: torch.eq(d1.value + d2.value, z.value),
                p=p
            )).value
        #diag先将images_x,image_y,labels_z维度合并，进入digit_s_d谓词分别运算，进入and时保留3个
        #维度，即diag，d1,d2,再进入mask,对d1,d2,diag后的labels_z做掩码，然后用exists消除d1,d2两个
        #维度，最后只有diag维度，用forall消除
        loss = 1. - sat_agg
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_sat += sat_agg
        # compute train accuracy
        predictions_x = torch.argmax(cnn_s_d(operand_images[:, 0].to(ltn.device)), dim=1)
        predictions_y = torch.argmax(cnn_s_d(operand_images[:, 1].to(ltn.device)), dim=1)
        predictions = predictions_x + predictions_y
        train_acc += torch.count_nonzero(torch.eq(addition_label.to(ltn.device), predictions)) / predictions.shape[0]
    train_loss = train_loss / len(single_train_loader)
    train_sat = train_sat / len(single_train_loader)
    train_acc = train_acc / len(single_train_loader)

    # test step
    for batch_idx, (operand_images, addition_label) in enumerate(single_test_loader):
        # ground variables with current batch data
        images_x = ltn.Variable("x", operand_images[:, 0])
        images_y = ltn.Variable("y", operand_images[:, 1])
        labels_z = ltn.Variable("z", addition_label)
        sat_agg = Forall(
            ltn.diag(images_x, images_y, labels_z),
            Exists(
                [d_1, d_2],
                And(Digit_s_d(images_x, d_1), Digit_s_d(images_y, d_2)),
                cond_vars=[d_1, d_2, labels_z],
                cond_fn=lambda d1, d2, z: torch.eq(d1.value + d2.value, z.value),
                p=p
            )).value
        loss = 1. - sat_agg
        test_loss += loss.item()
        test_sat += sat_agg
        # compute test accuracy
        predictions_x = torch.argmax(cnn_s_d(operand_images[:, 0].to(ltn.device)), dim=1)
        predictions_y = torch.argmax(cnn_s_d(operand_images[:, 1].to(ltn.device)), dim=1)
        predictions = predictions_x + predictions_y
        test_acc += torch.count_nonzero(torch.eq(addition_label.to(ltn.device), predictions)) / predictions.shape[0]
    test_loss = test_loss / len(single_test_loader)
    test_sat = test_sat / len(single_test_loader)
    test_acc = test_acc / len(single_test_loader)

    # we print metrics every epoch of training

    print(" epoch %d | Train loss %.4f | Train Sat %.4f | Train Acc %.4f | Test loss %.4f | Test Sat %.4f |"
                " Test Acc %.4f " %
          (epoch, train_loss, train_sat, train_acc, test_loss, test_sat, test_acc))