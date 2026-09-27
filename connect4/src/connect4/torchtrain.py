from game.gameInterface import ConnectFour
import torch
import torch.nn as nn


board = ConnectFour()

board.print_board()

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

        self.output = nn.Linear(128, 3)

    def forward(self, input):
        x = self.relu1(self.conv1(input))
        x = self.relu2(self.conv2(x))
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.output(x)

        return x

model = CNN()

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr = 0.001
)


input = 0
game_outcome = 0
for training_step in range(2000):
    predicted_value = model(input)

    loss = criterion(predicted_value, game_outcome)
    if training_step % 100 == 99:
        print(training_step, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    



