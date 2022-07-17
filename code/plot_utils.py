import os
import wave
from time import sleep
from typing import List

from librosa.display import specshow, waveshow
from matplotlib import pyplot as plt
from pyaudio import PyAudio
from rainbowgram.rain2graph import rain2graph
from rainbowgram.wave_rain import wave2rain
from tqdm import tqdm

from autoencoder import VAE
from config import *
from preprocess import MinMaxNormaliser
from soundgenerator import convert_spectrograms_to_audio, play_sounds
from train import load_dataset

chunk = cfg.SAMPLE_RATE


def play_all_original_sounds():
    directory = "../datasets/one_shot_percussive_sounds/audio"

    p = PyAudio()
    stream = p.open(format=2,
                    channels=1,
                    rate=cfg.SAMPLE_RATE,
                    output=True)

    for file in tqdm(os.listdir(directory)):
        filename = os.fsdecode(file)
        if filename.endswith(".wav"):
            f = wave.open(os.path.join(directory, filename), "rb")
            data = np.frombuffer(f.readframes(chunk), dtype=np.int16)
            data = np.pad(data, (0, chunk - f.getnframes()), 'constant')
            stream.write(np.frombuffer(data, dtype=np.int16))
        else:
            continue

    stream.stop_stream()
    stream.close()


def play_all_spectrogrammed_sounds():
    x_train, _ = load_dataset(num_samples=100)
    play_sounds(x_train, is_spec=True)


def play_all_reconstructed_sounds():
    x_train, file_paths = load_dataset(num_samples=200)
    vae = VAE.load("model_220621")
    play_sounds(
        vae.model.predict(x_train),
        is_spec=True,
        # print_names=file_paths
    )


def show_rain(filename: str):
    f = wave.open(filename, "rb")

    data_original = np.frombuffer(f.readframes(np.inf), dtype=np.int16)

    data_original = np.pad(data_original, (0, chunk - len(data_original)), 'constant')
    normalization_factor = np.max(np.abs(data_original)).astype(np.float64)
    data_original = data_original / normalization_factor
    rain_original = wave2rain(data_original, sr=cfg.SAMPLE_RATE, n_fft=cfg.N_FFT, stride=cfg.HOP_LENGTH)

    rain2graph(rain_original)
    plt.show(dpi=600)


def pretty_plot_spectrogram(
        spectrogram: np.ndarray,
        signal: Optional[np.ndarray] = None,
        title: str = ''):
    if signal is not None:
        # plt.figure()
        fig, axs = plt.subplots(2, 1, constrained_layout=True)
        fig.suptitle(title, fontsize=16)

        waveshow(signal / cfg.NORMALIZATION_FACTOR, sr=cfg.SAMPLE_RATE, ax=axs[0])
        # divider = make_axes_locatable(axs[0])
        # cax = divider.append_axes("right", size="5%", pad=.05)
        axs[0].set_xticks(np.linspace(0, 1, 11))
        axs[0].set_xlim([0, 1])
        axs[0].set_title('Signal')
        axs[0].set_xlabel('Time [s]')
        axs[0].set_ylabel('Amplitude')

        img = specshow(spectrogram[..., 0], ax=axs[1],
                       x_axis='time', y_axis='log', sr=cfg.SAMPLE_RATE, n_fft=cfg.N_FFT, hop_length=cfg.HOP_LENGTH)

        cax = axs[1].inset_axes([1.01, 0.0, 0.02, 1.0])
        cbar = plt.colorbar(img, format="%+2.f dB", ax=axs[1], cax=cax)
        cbar.ax.tick_params(labelsize=6)
        axs[1].set_xticks(np.linspace(0, 1, 11))
        axs[1].set_xlim([0, 1])
        axs[1].set_title('Spectrogram')
        axs[1].set_xlabel('Time [s]')
        axs[1].set_ylabel('Frequency [Hz]')

        # plt.tight_layout()
        plt.show()

    else:
        fig, ax = plt.subplots()
        ax.set(title=title)

        img = specshow(spectrogram[..., 0], ax=ax,
                       x_axis='time', y_axis='log', sr=cfg.SAMPLE_RATE, n_fft=cfg.N_FFT, hop_length=cfg.HOP_LENGTH)

        ax.set_xticks(np.linspace(0, 1, 11))
        ax.set_xlim([0, 1])
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Frequency [Hz]')

        fig.colorbar(img, ax=ax, format="%+2.f dB")

        plt.gcf().set_dpi(300)
        plt.show()


def show_reconstruction_examples():
    num_samples = 1
    show = True
    play = False

    vae = VAE.load("model_220623")
    m, _ = load_dataset(num_samples=num_samples)
    z = vae.model.predict(m)

    if play:
        play_sounds(convert_spectrograms_to_audio(m), is_spec=False)
        sleep(1)
        play_sounds(convert_spectrograms_to_audio(z), is_spec=False)

    if show:
        m_signals = convert_spectrograms_to_audio(m)
        z_signals = convert_spectrograms_to_audio(z)
        m_specs = MinMaxNormaliser.denormalise(m)
        z_specs = MinMaxNormaliser.denormalise(z)
        for i in range(num_samples):
            pretty_plot_spectrogram(m_specs[i], m_signals[i], title=f"Sample - {i + 1}(/{num_samples})")
            pretty_plot_spectrogram(z_specs[i], z_signals[i], title=f"Sample reconstruction - {i + 1}(/{num_samples})")

        # Zero response
        pretty_plot_spectrogram(vae.model.predict(np.zeros((1, 128, 128, 1)))[0], title=f"Zeros reconstruction")


def show_interpolation(
        interpolated_specs: np.ndarray,
        interpolated_signals: List[np.ndarray],
        title: str) \
        -> None:
    assert len(interpolated_specs) == len(interpolated_signals) == 9, \
        "It is hardcoded having 9 total steps of interpolation including edges"

    pretty_plot_spectrogram(interpolated_specs[0], interpolated_signals[0], title=f"Interpolation {title} - source")
    pretty_plot_spectrogram(interpolated_specs[2], interpolated_signals[2], title=f"Interpolation {title} - 25%")
    pretty_plot_spectrogram(interpolated_specs[4], interpolated_signals[4], title=f"Interpolation {title} - 50%")
    pretty_plot_spectrogram(interpolated_specs[6], interpolated_signals[6], title=f"Interpolation {title} - 75%")
    pretty_plot_spectrogram(interpolated_specs[8], interpolated_signals[8], title=f"Interpolation {title} - target")


def play_interpolation(
        interpolated_specs: np.ndarray,
        save_dir: Optional[str] = None) \
        -> None:
    output_specs = np.stack([
        interpolated_specs[0], interpolated_specs[0], cfg.EMPTY_SPEC,
        interpolated_specs[-1], interpolated_specs[-1], cfg.EMPTY_SPEC,
        *interpolated_specs
    ])

    play_sounds(output_specs, save_dir=save_dir, save_joined=True)
