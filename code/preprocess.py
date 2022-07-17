import os
import pickle
from typing import Dict

import librosa
import numpy as np
from tqdm import tqdm

from config import cfg


class Loader:
    """Loader is responsible for loading an audio file."""

    def __init__(self, sample_rate: int, duration: float):
        self.sample_rate: int = sample_rate
        self.duration: float = duration

    def load(self, file_path: str) -> np.ndarray:
        signal = librosa.load(file_path,
                              sr=self.sample_rate,
                              duration=self.duration)[0]
        return signal


class Padder:
    """Padder is responsible to apply padding to an array."""

    def __init__(self, pad_mode: str = "constant"):
        self.pad_mode: str = pad_mode

    def pad(self, array: np.ndarray, pad_size_left: int = 0, pad_size_right: int = 0) -> np.ndarray:
        padded_array = np.pad(array,
                              (pad_size_left, pad_size_right),
                              mode=self.pad_mode)
        return padded_array


class SpectrogramExtractor:
    """SpectrogramExtractor extracts log spectrograms (not in dB) from a
    time-series signal.
    """

    def __init__(self, n_fft: int, hop_length: int):
        self.n_fft: int = n_fft
        self.hop_length: int = hop_length

    def extract(self, signal: np.ndarray) -> np.ndarray:
        spectrogram = librosa.stft(signal,
                                   n_fft=self.n_fft,
                                   hop_length=self.hop_length)[:-1]

        spectrogram = librosa.power_to_db(spectrogram, ref=np.max, top_db=cfg.TOP_DB)

        return spectrogram


class MinMaxNormaliser:
    """MinMaxNormaliser applies min max normalisation to an array."""

    @staticmethod
    def normalise(array: np.ndarray,
                  new_min: float = cfg.NORM_RANGE[0],
                  new_max: float = cfg.NORM_RANGE[1]) -> np.ndarray:
        return np.interp(x=array,
                         xp=(array.min(), array.max()),
                         fp=(new_min, new_max))

    @staticmethod
    def denormalise(array: np.ndarray,
                    new_min: float = cfg.DB_RANGE[0],
                    new_max: float = cfg.DB_RANGE[1]) -> np.ndarray:
        if len(array.shape) == len(cfg.SPECTROGRAM_SHAPE):
            return np.interp(x=array,
                             xp=(array.min(), array.max()),
                             fp=(new_min, new_max))

        else:
            assert len(array.shape) == len(cfg.SPECTROGRAM_SHAPE) + 1

            for i in range(len(array)):
                array[i] = np.interp(x=array[i],
                                     xp=(array[i].min(), array[i].max()),
                                     fp=(new_min, new_max))
            return array


class Saver:
    """saver is responsible to save features, and the min max values."""

    def __init__(self, feature_save_dir: str):
        self.feature_save_dir = feature_save_dir

    def save_feature(self, feature: np.ndarray, file_path: str) -> str:
        save_path = self._generate_save_path(file_path)
        np.save(save_path, feature)
        return save_path

    @staticmethod
    def _save(data: Dict[str, Dict[str, float]], save_path: str) -> None:
        with open(save_path, "wb") as f:
            pickle.dump(data, f)

    def _generate_save_path(self, file_path: str) -> str:
        file_name = os.path.split(file_path)[1]
        save_path = os.path.join(self.feature_save_dir, file_name + ".npy")
        return save_path


class PreprocessingPipeline:
    """PreprocessingPipeline processes audio files in a directory, applying
    the following steps to each file:
        1- load a file
        2- pad the signal (if necessary)
        3- extracting log spectrogram from signal
        4- normalise spectrogram
        5- save the normalised spectrogram

    Storing the min max values for all the log spectrograms.
    """

    def __init__(self, *,
                 loader: Loader,
                 padder: Padder,
                 extractor: SpectrogramExtractor,
                 normaliser: MinMaxNormaliser,
                 saver: Saver):

        self._loader: Loader = loader
        self._padder: Padder = padder
        self._extractor: SpectrogramExtractor = extractor
        self._normaliser: MinMaxNormaliser = normaliser
        self._saver: Saver = saver
        self._num_expected_samples: int = int(loader.sample_rate * loader.duration)

    def process(self, audio_files_dir: str) -> None:
        for root, _, files in os.walk(audio_files_dir):
            for file in tqdm(files):
                file_path = os.path.join(root, file)
                self._process_file(file_path)
                # print(f"Processed file {file_path}")

    def _process_file(self, file_path: str) -> None:
        signal = self._loader.load(file_path)

        # Pad signal
        if self._is_padding_necessary(signal):
            signal = self._apply_padding(signal)

        # Extract spectrogram in dB
        feature = self._extractor.extract(signal)

        # Pad spectrogram to be in desired size
        feature = np.pad(feature,
                         ((0, cfg.SPECTROGRAM_SHAPE[0] - feature.shape[0]),
                          (0, cfg.SPECTROGRAM_SHAPE[1] - feature.shape[1])),
                         mode='constant',
                         constant_values=-cfg.TOP_DB)

        # Normalize spectrogram values
        feature = self._normaliser.normalise(feature)

        feature = feature[..., None]
        assert feature.shape == cfg.SPECTROGRAM_SHAPE
        self._saver.save_feature(feature, file_path)

    def _is_padding_necessary(self, signal: np.ndarray) -> bool:
        return len(signal) < self._num_expected_samples

    def _apply_padding(self, signal: np.ndarray) -> np.ndarray:
        num_missing_samples = self._num_expected_samples - len(signal)
        padded_signal = self._padder.pad(signal, pad_size_right=num_missing_samples)
        return padded_signal


if __name__ == "__main__":
    # instantiate all objects
    preprocessing_pipeline = PreprocessingPipeline(
        loader=Loader(sample_rate=cfg.SAMPLE_RATE, duration=cfg.DURATION),
        padder=Padder(),
        extractor=SpectrogramExtractor(n_fft=cfg.N_FFT, hop_length=cfg.HOP_LENGTH),
        normaliser=MinMaxNormaliser(),
        saver=Saver(feature_save_dir=cfg.SPECTROGRAMS_SAVE_DIR))

    preprocessing_pipeline.process(cfg.FILES_DIR)
