'''Data Loader that loads the raw WAV dataset, Generate Mel-spectrogram for inputs'''

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import librosa
import librosa.display

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


class AudioDataset(Dataset):
    '''Dataset Object, please put all audio files under data/genres_original folder in each genre folder'''

    def __init__(self, mode="train"):
        if mode not in ["train", "validation", "test"]:
            raise ValueError("Mode must be either train or validation or test")
        self._mode = mode
        self._class = {
            "blues": 0,
            "classical": 1,
            "country": 2,
            "disco": 3,
            "hiphop": 4,
            "jazz": 5,
            "metal": 6,
            "pop": 7,
            "reggae": 8,
            "rock": 9,
        }

    def get_dataset_path(self, _class):
        '''Get the path to a particular class folder'''
        audio_path = os.path.join(DIR_PATH, "data", _class)
        if self._mode == "test":
            audio_path = os.path.join(DIR_PATH, "data", "test", _class)
        return audio_path

    def get_dataset_count(self, dataset_path):
        '''Get the number of audio files in a particular dataset'''
        for root, _, files in os.walk(dataset_path):
            total_count = len(files)
            train_count = total_count // 10 * 8
            if self._mode == "train":
                return train_count
            elif self._mode == "validation":
                return total_count - train_count
            elif self._mode == "test":
                return total_count

    def create_spectrogram(self, audio_path):
        '''Output a Mel-spectrogram in 2D array from an audio file'''
        y, sr = librosa.load(audio_path)
        spect = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=2048, hop_length=1024)
        spect = librosa.power_to_db(spect, ref=np.max)
        return spect.T[:640]

    def show_spectrogram(self, spect):
        '''Display the Mel-spectrogram'''
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(
            spect.T, y_axis='mel', fmax=8000, x_axis='time')
        plt.colorbar(format='%+2.0f dB')
        plt.show()

    def show_item(self, index):
        spect, label = self.__getitem__(index)
        print("Genre: {}".format(label))
        self.show_spectrogram(spect.T[0].T)

    def __str__(self):
        '''Print description about this dataset'''
        _str = "This is an audio mel-spectrogram dataset that takes in WAV files.\n"
        _str += "This dataset is a {} dataset\n".format(self._mode)
        _str += "It contains {} data samples\n".format(len(self))
        _str += "It contains the following labels: {}\n".format(
            list(self._class.keys()))
        return _str

    def __len__(self):
        '''Return count of dataset'''
        length = 0
        for key in self._class:
            dataset_path = self.get_dataset_path(key)
            length += self.get_dataset_count(dataset_path)
        return length

    def __getitem__(self, index):
        '''Return inputs, label'''
        all_count = len(self)
        if index >= all_count:
            raise IndexError(
                "Index out of bound! Total number of items in this dataset is: {}".format(all_count))
        all_audio_files = []
        for key in self._class:
            dataset_path = self.get_dataset_path(key)
            sub_audio_files = []
            for root, _, files in os.walk(dataset_path):
                total_count = len(files)
                train_count = total_count // 10 * 8
                if self._mode == "train":
                    sub_audio_files = files[: train_count]
                elif self._mode == "validation":
                    sub_audio_files = files[train_count:]
                elif self._mode == "test":
                    sub_audio_files = files
            for audio_file in sub_audio_files:
                all_audio_files.append({
                    "path": os.path.join(dataset_path, audio_file),
                    "_class": key,
                })

        target_path = all_audio_files[index]["path"]
        target_label = all_audio_files[index]["_class"]
        spect = self.create_spectrogram(target_path)
        # reshape and expand dims for conv2d
        spect_expand = np.expand_dims(spect, axis=0)
        return spect_expand, self._class[target_label]


def get_data_loader(mode="train", batch_size=64):
    '''Get the Dataloader, mode can be train or validation'''
    dataset = AudioDataset(mode=mode)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


if __name__ == "__main__":
    train_loader = get_data_loader(mode="validation")
    print(len(train_loader.dataset))
