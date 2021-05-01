'''Model training program'''
import torch
import torch.nn as nn
from data_loader import get_data_loader
from model import PCRNN
import os
import time
import math
from tqdm import tqdm


def write_val_to_csv(val, name):
    '''Append value to file, if not exist, create one'''
    with open("{}.csv".format(name), "a", encoding="utf-8") as fh:
        fh.write("{}\n".format(val))
    fh.close()


def train(model=None, n_epochs=1, batch_size=64):
    '''Train the model given'''
    if model is None:
        raise Exception("Model cannot be empty!")

    train_loader = get_data_loader(mode="train", batch_size=batch_size)
    test_loader = get_data_loader(mode="validation", batch_size=batch_size)

    lr = 5e-4
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    best_accuracy = 0

    for epoch in range(1, n_epochs + 1):

        # Training
        train_loss = 0
        model.train()
        for datas, target in tqdm(train_loader, desc="Training..."):
            optimizer.zero_grad()
            output = model.forward(datas)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Evaluation
        accuracy = 0
        valid_loss = 0
        model.eval()
        for data, val_target in tqdm(test_loader, desc="Validating..."):
            val_output = model.forward(data)
            loss = criterion(val_output, val_target)
            valid_loss += loss.item()
            _, preds = torch.max(val_output, 1)
            accuracy += sum((preds == val_target).numpy())

        train_loss /= math.ceil(len(test_loader.dataset)/batch_size)
        valid_loss /= math.ceil(len(test_loader.dataset)/batch_size)
        accuracy /= len(test_loader.dataset)

        print("Epoch: {:3}/{:3} Train Loss: {:.4f} Validation Loss: {:.4f} Accuracy: {:.2f}%".format(
            epoch, n_epochs, train_loss, valid_loss, accuracy*100))

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), "model2.pth")

        write_val_to_csv(valid_loss, "validation_loss2")
        write_val_to_csv(accuracy, "accuracy2")
        write_val_to_csv(train_loss, "train_loss2")

        if accuracy >= 0.98:
            print('Performance condition satisfied, stopping..')
            torch.save(model.state_dict(), "model.pth")
            print("Run time: {:.3f} min".format(
                (time.time() - start)/60))
            return model

    return model


if __name__ == "__main__":
    model = PCRNN()
    model.load_state_dict(torch.load("model.pth"))
    train(model, n_epochs=100)
