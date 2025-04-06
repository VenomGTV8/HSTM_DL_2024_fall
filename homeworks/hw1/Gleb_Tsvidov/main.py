import argparse
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler


def load_data(train_csv, val_csv, test_csv):
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    X_train = train_df.drop(['order0', 'order1', 'order2'], axis=1)
    y_train = train_df['order0']

    X_val = val_df.drop(['order0', 'order1', 'order2'], axis=1)
    y_val = val_df['order0']

    X_test = test_df

    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_val = ss.transform(X_val)
    X_test = ss.transform(X_test)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.int64)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val.values, dtype=torch.int64)
    X_test = torch.tensor(X_test, dtype=torch.float32)

    return X_train, y_train, X_val, y_val, X_test


class MLP(nn.Module):
    def __init__(self, ):
        super().__init__()

        self.fc1 = nn.Linear(360, 720)
        self.fc2 = nn.Linear(720, 720)
        self.fc3 = nn.Linear(720, 3)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


def init_model(lr):
    model = MLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    return model, criterion, optimizer


def evaluate(model, X, y, criterion):
    model.eval()
    with torch.no_grad():
        outputs = model(X)
        loss = criterion(outputs, y)
        predictions = outputs.argmax(dim=1)
        accuracy = accuracy_score(y, predictions)
        conf_matrix = confusion_matrix(y, predictions)

    return predictions, accuracy, loss, conf_matrix


def predict(model, X):
    model.eval()
    with torch.no_grad():
        outputs = model(X)
        predictions = outputs.argmax(dim=1)

    return predictions


def train(model, criterion, optimizer, X_train, y_train, X_val, y_val, epochs, batch_size):
    TRAIN_LOSSES = []
    VAL_LOSSES = []

    for epoch in range(epochs):
        train_loss = 0
        model.train()
        for i in range(0, X_train.size(0), batch_size):
            X_batch = X_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= X_train.size(0)
        print(f'\t Train: Epoch {epoch}, train Loss: {train_loss}')

        predictions, val_accuracy, val_loss, conf_matrix = evaluate(model, X_val, y_val, criterion)
        TRAIN_LOSSES.append(train_loss)
        VAL_LOSSES.append(val_loss.item())
        print(f'\t val Loss {val_loss.item()} val accuracy: {val_accuracy}')

    return model, TRAIN_LOSSES, VAL_LOSSES


def main(args):
    X_train, y_train, X_val, y_val, X_test = load_data(args.train_csv, args.val_csv, args.test_csv)
    model, criterion, optimizer = init_model(args.lr)
    model, train_losses, val_losses = train(model,
                                            criterion,
                                            optimizer,
                                            X_train,
                                            y_train,
                                            X_val,
                                            y_val,
                                            args.num_epoches,
                                            args.batch_size)
    predictions = predict(model, X_test)

    pd.DataFrame(predictions).to_csv(args.out_csv)

    # plt.plot(val_losses)
    # plt.plot(train_losses)
    # plt.legend(['Train_loss', 'Val_loss'])
    # plt.show()
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_csv', default='../data/train.csv')
    parser.add_argument('--val_csv', default='../data/val.csv')
    parser.add_argument('--test_csv', default='../data/test.csv')
    parser.add_argument('--out_csv', default='submission.csv')
    parser.add_argument('--lr', default=3e-4)
    parser.add_argument('--batch_size', default=512)
    parser.add_argument('--num_epoches', default=5)

    args = parser.parse_args()
    main(args)
