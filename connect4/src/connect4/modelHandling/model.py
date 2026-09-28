import torch
from torch import nn
from torch.utils.data import TensorDataset, Dataset, DataLoader
from pathlib import Path
from connect4.dataManagement import dataManager

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

    def trainModel(
            self,
            dataset: TensorDataset,
            model_file_path: str,
            checkpoint_file_path = None,
            batch_size: int = 32, 
            epoch: int = 5,
            device: str = "cpu",
            checkpoint_frequency: int = 5000,
            report_frequency: int = 100,
            shuffle: bool = True,
            learning_rate: float = 0.001
            ):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr = learning_rate
            )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle
        )
        train_turn = 0
        running_loss = 0
        self.train()
        self.to(device)
        for epoch in range(1, epoch + 1):
            for batch_index, (batch_input, batch_output) in enumerate(dataloader):

                batch_input = batch_input.to(device)
                batch_output = batch_output.to(device)

                predictions = self(batch_input)
                loss = criterion(predictions, batch_output)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_turn +=1
                running_loss += loss.item()

                if train_turn % checkpoint_frequency == checkpoint_frequency:
                    torch.save(self.state_dict(), self.get_default_path(r"checkpointmodel_" + str(int(checkpoint_frequency / 1000)) + r".pt"))
                if train_turn % report_frequency == 0:
                    print(f"turn: {train_turn}. Average loss over the last {report_frequency} steps: " + 
                          str(running_loss / report_frequency))
                    running_loss = 0
        torch.save(self.state_dict(), Path(model_file_path))
        print("training finished, model saved in:" + model_file_path)

    def get_default_path(self, modelname: str):
        if ".pt" not in modelname:
            modelname = modelname + r".pt"
        script_dir = Path(__file__).resolve().parent
        model_dir = str(script_dir.parent.parent.parent) + r"/models"
        return Path(model_dir + "/" + modelname)

    def load_model(self, model_path):
        state_dict = torch.load(Path(model_path), weights_only=True)
        self.load_state_dict(state_dict=state_dict)

    def inference(self, input):
        self.eval()
        x = self.forward(input)
        x=  nn.Softmax(x)
        print(x)
        return x
