import torch
import torch.nn as nn

class MultiLayerPerceptron():
    model = any
    loss_fun = nn.CrossEntropyLoss()
    optim = any

    def create_model(self, inputs):
        activation, opt, r = inputs
        self.model = nn.Sequential()
        self.model.add.module("dense1", nn.linear(42, 100)) #input dim, output dim
        self.model.add.module("act1", nn.ReLU())
        self.model.add.module("dense2", nn.linear(100, 42))
        self.model.add.module("act2", nn.ReLU())
        self.model.add.module("dense3", nn.linear(42, 1))
        self.model.add.module("out", nn.Sigmoid())
        if (opt == 1):
            self.optim = torch.optim.Adam(self.model.parameters(), lr = r)
        elif (opt == 2):
            self.optim = torch.optim.NAdam(self.model.parameters(), lr = r)
        elif (opt == 3):
            self.optim = torch.optim.SGD(self.model.parameters(), lr = r)
        elif (opt == 4):
            self.optim = torch.optim.RMSprop(self.model.parameters(), lr = r)

    def train_model(self, trainset, epochs):
        data, label = trainset
        for n in range(epochs):
            ouptuts = self.model(data)
            loss = self.loss_fun(outputs, label)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

    def do_training(self, data):
        with torch.no_grad():
            self.model.eval()
            prediction = self.model(data)
        _, predicted_class = torch.max(prediction, 1)
        return predicted_class
    
    def __init__(self, inputs) -> None:
        self.create_model(inputs)