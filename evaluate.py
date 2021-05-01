'''Evaluate the model by running through test set'''
import torch
from seaborn import sns
from model import PCRNN
from data_loader import get_data_loader, AudioDataset
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score, recall_score, precision_score


def show_confusion_matrix(y_true, y_pred, genre_list):
    '''Plot a confusion matrix'''
    mat = confusion_matrix(y_true, y_pred)
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
                xticklabels=genre_list,
                yticklabels=genre_list)
    plt.xlabel('true label')
    plt.ylabel('predicted label')


def test(model_path=None):
    '''Run model on test set and evaluate the model'''
    '''Return y_true, y_pred'''
    if model_path is None:
        raise Exception("model_path cannot be empty!")

    model = PCRNN()
    model.load_state_dict(torch.load(model_path))
    test_loader = get_data_loader(mode="test", batch_size=1)

    model.eval()
    y_true = []
    y_pred = []
    for data, target in tqdm(test_loader, desc="Testing..."):
        output = model.forward(data)
        _, preds = torch.max(output, 1)
        y_true.extend(list(target.numpy()))
        y_pred.extend(list(preds.numpy()))

    return y_true, y_pred


def evaluate(model_path=None):
    '''Evaluate the model with visualization'''
    if model_path is None:
        raise Exception("model_path cannot be empty!")
    y_true, y_pred = test(model_path)
    dataset = AudioDataset()
    genre_list = list(dataset._class.keys())

    # Classification report
    print("Classification Report: ")
    print(classification_report(y_true, y_pred, target_names=genre_list))

    # Scores
    f1 = f1_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    precision = precision_score(y_true, y_pred, average="weighted")
    accuracy = accuracy_score(y_true, y_pred)
    print("F1: {:6f}".format(f1))
    print("Recall: {:6f}".format(recall))
    print("Precision: {:6f}".format(precision))
    print("Accuracy: {:6f}".format(accuracy))

    # Confusion Matrix
    show_confusion_matrix(y_true, y_pred, genre_list)


def predict(model_path=None, audio_path=None):
    '''Get top-5 predictions for an audio file'''
    if model_path is None:
        raise Exception("model_path cannot be empty!")
    if audio_path is None:
        raise Exception("Audio file path cannot be empty!")

    model = PCRNN()
    model.load_state_dict(torch.load(model_path))
    dataset = AudioDataset()

    # Process audio
    spect = dataset.create_spectrogram(audio_path)
    spect_expanded = np.expand_dims(spect, axis=0)
    inputs = torch.Tensor(spect_expanded).unsqueeze(0)

    model.eval()
    predictions = model.forward(inputs)
    prediction_map = dict()
    genre_map = dataset._class
    index_to_genre = dict()
    for k, v in genre_map.items():
        index_to_genre[v] = k

    predictions = predictions.numpy()
    for idx, confidence in enumerate(predictions):
        prediction_map[index_to_genre[idx]] = confidence

    sorted_predictions = sorted(
        prediction_map.items(), key=lambda item: item[1], reverse=True)
    top_5 = sorted_predictions[:5]
    return top_5
