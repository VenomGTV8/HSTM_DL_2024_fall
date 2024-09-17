import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error

num_epoches = 1
batch_size = 1024
learning_rate = 100

train_df = pd.read_csv('practices/data_practice1/train.csv')
val_df = pd.read_csv('practices/data_practice1/val.csv')
test_df = pd.read_csv('practices/data_practice1/test.csv')

X_train = train_df[['F', 'x']].values
y_train = train_df['y'].values

X_val = val_df[['F', 'x']].values
y_val = val_df['y'].values

X_test = test_df[['F', 'x']].values
y_test = test_df['y'].values

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


model = MLP()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


def train(model, criterion, optimizer, X_train, y_train, X_val, y_val, epochs):
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for i in range(0, X_train.size(0), batch_size):
            X_batch = X_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]

            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f'\t Train: Epoch {epoch}, train Loss: {loss.item()}')
            train_loss += loss.item()

        train_loss /= X_train.size(0)
        if epoch % 1 == 0:
            model.eval()
            val_outputs = model(X_val).squeeze()
            val_loss = criterion(val_outputs, y_val)
            print(f'Val: Epoch {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss.item()}')


train(model, criterion, optimizer, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, num_epoches)

model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor).squeeze()
    test_outputs = test_outputs.numpy()
    test_loss = mean_squared_error(y_test, test_outputs)
    print(f'Test MSE: {test_loss}')
