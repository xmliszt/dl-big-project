'''Model training program'''
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_loader import AudioDataset
from crnn import Pcrnn
import time


train_set = AudioDataset(mode="train")
validation_set = AudioDataset(mode="validation")
print(train_set)
print(validation_set)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(validation_set, batch_size=64, shuffle=True)


def write_val_to_csv(val, name):

    with open("{}.csv".format(name), "a", encoding="utf-8") as fh:
        fh.write("{}\n".format(val))
    fh.close()


def train(model=None, n_epochs=1):
    '''Train the model given'''
    if model is None:
        raise Exception("Model cannot be empty!")

    lr = 5e-4
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    best_accuracy = 0

    for epoch in range(1, n_epochs + 1):
        train_loss = 0

        # Training
        model.train()
        for datas, target in train_loader:
            optimizer.zero_grad()
            output = model.forward(datas)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            accuracy = 0

            # Evaluation
            valid_loss = 0
            model.eval()
            for val_step, (data, val_target) in enumerate(test_loader):
                data, tmp_target = data, val_target
                val_output = model.forward(data)
                loss = criterion(val_output, tmp_target)
                valid_loss += loss.item()
                _, preds = torch.max(val_output, 1)
                accuracy += sum((preds == tmp_target).numpy())

            valid_loss /= (len(test_loader.dataset)//64)
            accuracy /= (len(test_loader.dataset))

            print("Epoch: {:3}/{:3} Train Loss: {:.6f} Validation Loss: {:.6f} Accuracy: {:.4f}".format(
                epoch, n_epochs, len(train_loader)*10, train_loss, valid_loss, accuracy))

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), "model.pth")

            write_val_to_csv(valid_loss, "validation_loss")
            write_val_to_csv(accuracy, "accuracy")
            write_val_to_csv(train_loss, "train_loss")

            model.train()

            if accuracy >= 0.98:
                print('Performance condition satisfied, stopping..')
                torch.save(model.state_dict(), "model.pth")
                print("Run time: {:.3f} min".format(
                    (time.time() - start)/60))
                return model, train_loss_list, validation_loss_list, accuracy_list

    return model, train_loss_list, validation_loss_list, accuracy_list


if __name__ == "__main__":
    model = Pcrnn()
    train(model)
