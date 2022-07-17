import os
from time import sleep
from typing import List, Tuple, Optional, Union

import librosa
import numpy as np
import soundfile as sf
from pyaudio import PyAudio

from autoencoder import VAE
from config import cfg
from preprocess import MinMaxNormaliser


class SoundGenerator:
    """
    SoundGenerator is responsible for generating audios from spectrograms.
    """

    def __init__(self, vae: Optional[VAE]):
        self.vae: VAE = vae

    def generate_from_spec(
            self,
            spectrograms: np.ndarray) \
            -> Tuple[List[np.ndarray], np.ndarray]:
        assert spectrograms.shape[1:] == cfg.SPECTROGRAM_SHAPE
        generated_spectrograms, latent_representations = self.vae.reconstruct(spectrograms)
        signals = convert_spectrograms_to_audio(generated_spectrograms)
        return signals, latent_representations

    def generate_from_latent(
            self,
            latent_representations: np.ndarray) \
            -> Tuple[List[np.ndarray], np.ndarray]:
        assert latent_representations.shape[1:] == (cfg.LATENT_SPACE_DIM,)
        generated_spectrograms, latent_representations = self.vae.decoder.predict(latent_representations)
        signals = convert_spectrograms_to_audio(generated_spectrograms)
        return signals, latent_representations


def convert_spectrograms_to_audio(
        spectrograms: np.ndarray) \
        -> List[np.ndarray]:
    assert spectrograms.shape[-3:] == cfg.SPECTROGRAM_SHAPE
    assert spectrograms.min() >= cfg.NORM_RANGE[0], "Given spectrogram is expected to be Normalized"
    spectrograms = spectrograms

    signals = []
    for spectrogram in spectrograms:

        if np.array_equal(spectrogram, cfg.EMPTY_SPEC):
            signal = (np.zeros(int(cfg.DURATION * cfg.SAMPLE_RATE))).astype(np.int16)

        else:
            # Convert used spectrogram from arbitrary log-scale to absolute linear-scale
            spectrogram = MinMaxNormaliser.denormalise(spectrogram)[..., 0]
            spectrogram_amp = librosa.db_to_power(spectrogram)

            # Covert spectrogram to signal with phase reconstruction
            signal = librosa.griffinlim(spectrogram_amp,
                                        n_iter=150,
                                        hop_length=cfg.HOP_LENGTH,
                                        momentum=0.,
                                        # n_fft=cfg.N_FFT,
                                        # window=librosa.filters.get_window(window='hann', Nx=cfg.N_FFT - 2),
                                        init=None,
                                        )

            # Trim length
            signal = signal[:int(cfg.DURATION * cfg.SAMPLE_RATE)]

            # Normalize audio signal
            normalization_factor = cfg.NORMALIZATION_FACTOR / np.max(np.abs(signal))
            signal = (normalization_factor * signal).astype(np.int16)

        signals.append(signal)

    return signals


def play_sounds(
        data: Union[np.ndarray, List[np.ndarray]],
        is_spec: bool = True,
        save_dir: Optional[str] = None,
        save_joined: bool = False,
        print_names: Optional[List[str]] = None,
        sleep_duration: float = 0.) \
        -> None:

    if is_spec:
        assert data.shape[1:] == cfg.SPECTROGRAM_SHAPE
        if print_names is not None: assert data.shape[0] == len(print_names)
        signals = convert_spectrograms_to_audio(data)
    else:
        assert isinstance(data, list)
        assert len(data[0].shape) == 1
        if print_names is not None: assert len(data) == len(print_names)
        signals = data

    p = PyAudio()
    stream = p.open(format=2,
                    channels=1,
                    rate=cfg.SAMPLE_RATE,
                    output=True)

    if print_names is None:
        for signal in signals:
            stream.write(np.frombuffer(signal, dtype=np.int16))
            sleep(sleep_duration)
    else:
        for signal, print_name in zip(signals, print_names):
            print(print_name)
            stream.write(np.frombuffer(signal, dtype=np.int16))
            sleep(sleep_duration)

    stream.stop_stream()
    stream.close()

    if save_dir:
        if save_joined:
            signals = [np.concatenate(signals)]
        save_signals(signals, save_dir)


def save_signals(signals: List[np.ndarray], save_dir: str, sample_rate: int = cfg.SAMPLE_RATE) -> None:
    for i, signal in enumerate(signals):
        save_path = os.path.join(save_dir, str(i) + ".wav")
        sf.write(save_path, signal, sample_rate)
