from game.gameInterface import ConnectFour
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
import pathlib
import os

def get_result(dataset: tuple, game_outcome):
    player_that_placed = int(dataset[2][0][0])
    # a version that uses one hot 1d matrices rather than class indices of classification
    # if game_outcome == player_that_placed:
    #     probability_win = 1
    #     probability_loss = 0
    #     probability_draw = 0
    # elif game_outcome == 2:
    #     probability_win = 0
    #     probability_loss = 0
    #     probability_draw = 1
    # else:
    #     probability_win = 0
    #     probability_loss = 1
    #     probability_draw = 0
    # return [probability_win, probability_loss, probability_draw]

    if game_outcome == player_that_placed:
        return 0       # win
    elif game_outcome == 2:
        return 2       # draw
    else:
        return 1       # loss

class  CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=4, 
            padding=1
        )

        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=0
        )

        self.relu2 = nn.ReLU()

        self.flatten = nn.Flatten()

        self.dense1 = nn.Linear(64*3*4, 128)

        self.reluDense = nn.ReLU()

        self.output = nn.Linear(128, 3)

    def forward(self, input):
        x = self.relu1(self.conv1(input))
        x = self.relu2(self.conv2(x))
        x = self.flatten(x)
        x = self.reluDense(self.dense1(x))
        x = self.output(x)

        return x

class Overall_dataset(Dataset):
    def __init__(self):
        self.data_list = []

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        tensor, number = self.data_list[index]
        return tensor, number

script_dir = pathlib.Path(__file__).resolve().parent

sub1batchNumber = 1
datapath = str(pathlib.Path(script_dir).parent.parent) + r"/training_data_kaggle/batch_" + str(sub1batchNumber) + r"/compiled.pt"
data = torch.load(pathlib.Path(datapath), weights_only=False)

print("getting data")
model_path = str(pathlib.Path(script_dir).parent.parent) + r"/models"
test_model_file_path = model_path + r"/test_model.pt"

print("initiating data")
positions = []
labels = []
for i in range(0, data.__len__()):
    positions.append(data.__getitem__(i)[0])
    labels.append(get_result(data.__getitem__(i)[0], data.__getitem__(i)[1]))
dataset = TensorDataset(torch.stack(positions), torch.tensor(labels, dtype=torch.long))
        
model = CNN()
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr = 0.0001
)

dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)

# input = 0
# game_outcome = 0
# for training_step in range(2000):
#     predicted_value = model(input)

#     loss = criterion(predicted_value, get_result(game_outcome))
#     if training_step % 100 == 99:
#         print(training_step, loss.item())

#     optimizer.zero_grad()
#     loss.backward()

#     optimizer.step()'

device = torch.device("cpu")
model.to(device)

checkpoint = 0

epoch = 5 
for epoch in range(epoch):
    model.train()
    running_loss = 0

    for batch_index, (batch_input, batch_output) in enumerate(dataloader):

        batch_input = batch_input.to(device)
        batch_output = batch_output.to(device)

        predictions = model(batch_input)
        loss = criterion(predictions, batch_output)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        checkpoint +=1

        running_loss += loss

        if checkpoint % 1000 == 999:
            torch.save(model.state_dict(), str(model_path) + r"/testcheckpointmodel_" + str(checkpoint / 1000) + r".pt")

        if checkpoint % 500 == 499:

            print("checkpoint:" + str(checkpoint) + "loss" + str(running_loss / 500))
            running_loss = 0
torch.save(model.state_dict(), test_model_file_path)
print("training have finished, model saved.")

    
