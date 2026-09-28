from game.gameInterface import ConnectFour
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
import pathlib
from modelHandling import model
from dataManagement.dataManager import Overall_dataset, get_result

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
        
model = model.CNN()
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

    
