import os
import wave

import librosa
from librosa import amplitude_to_db, db_to_amplitude, stft, istft

from autoencoder import VAE
from config import *
from interpolate import interpolate_specs
from plot_utils import pretty_plot_spectrogram, show_interpolation, play_interpolation
from preprocess import MinMaxNormaliser
from soundgenerator import convert_spectrograms_to_audio, play_sounds
from train import load_dataset

chunk = cfg.SAMPLE_RATE


def waveform_reconstruction_loss():
    directory = "../datasets/one_shot_percussive_sounds/audio"

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".wav"):
            f = wave.open(os.path.join(directory, filename), "rb")
            data_original = np.frombuffer(f.readframes(np.inf), dtype=np.int16)

            data_original = np.pad(data_original, (0, chunk - f.getnframes()), 'constant')
            normalization_factor = np.max(np.abs(data_original)).astype(np.float64)
            data_original = data_original / normalization_factor
            spectrogram = stft(data_original, n_fft=cfg.N_FFT, hop_length=cfg.HOP_LENGTH)
            data_reconstructed = istft(spectrogram, n_fft=cfg.N_FFT, hop_length=cfg.HOP_LENGTH)
            waves_diff = np.abs(data_original - data_reconstructed)

            assert len(waves_diff) == len(data_original) == len(data_reconstructed) == chunk
            assert np.mean(waves_diff ** 2) <= 1e-20
            assert np.max(waves_diff) <= 1e-10

        else:
            continue


def griffin_lim(
        spec_db: np.ndarray,
        num_iters: int = 10,
        phase_angle: float = 0.0,
        n_fft: int = cfg.N_FFT,
        hop_length: int = cfg.HOP_LENGTH) \
        -> np.ndarray:
    def _db_to_amp(x):
        return np.power(10.0, x * 0.05)

    def _inv_magphase(mag, phase_angle):
        phase = np.cos(phase_angle) + 1.j * np.sin(phase_angle)
        return mag * phase

    assert num_iters > 0
    mag = _db_to_amp(spec_db)
    complex_spec = _inv_magphase(mag, phase_angle)

    audio = librosa.istft(complex_spec, win_length=n_fft, hop_length=hop_length, center=True)
    for i in range(num_iters):
        complex_spec = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        _, phase = librosa.magphase(complex_spec)
        phase_angle = np.angle(phase)
        complex_spec = _inv_magphase(mag, phase_angle)
        audio = librosa.istft(complex_spec, win_length=n_fft, hop_length=hop_length, center=True)

    return audio


def complex_absolute_to_db():
    spec_amp_stacked = 2. * (np.random.random((128, 128, 2)) - 0.5)
    spec_amp_stacked /= np.max(np.linalg.norm(spec_amp_stacked))

    spec_amp_complex = 1. * spec_amp_stacked[..., 0] + 1.j * spec_amp_stacked[..., 1]
    spec_db_complex = np.exp(1j * np.angle(spec_amp_complex)) * amplitude_to_db(np.abs(spec_amp_complex) - cfg.EPS,
                                                                                ref=np.max, amin=1e-8)
    spec_db_stacked = np.stack([np.real(spec_db_complex), np.imag(spec_db_complex)], axis=-1)

    spec_db_complex_2 = 1. * spec_db_stacked[..., 0] + 1.j * spec_db_stacked[..., 1]
    spec_amp_complex_2 = np.exp(1j * np.angle(-spec_db_complex_2)) * (
            db_to_amplitude(-np.abs(spec_db_complex_2)) + cfg.EPS)
    spec_amp_stacked_2 = np.stack([np.real(spec_amp_complex_2), np.imag(spec_amp_complex_2)], axis=-1)

    assert np.max(np.abs(spec_db_complex - spec_db_complex_2)) < 1e-7
    assert np.max(np.abs(spec_amp_complex - spec_amp_complex_2)) < 1e-7
    assert np.max(spec_amp_stacked - spec_amp_stacked_2) < 1e-7


def find_best_reconstructions():
    num_samples = 256
    k = 32

    vae = VAE.load("model_220624")
    m, _ = load_dataset(num_samples=num_samples)
    z = vae.model.predict(m)
    mse = (np.square(m - z)).mean(axis=(1, 2, 3))
    signal_length = np.count_nonzero(m.sum(axis=(1, 3)), axis=1)
    score = (signal_length / 128) / mse

    best_score_idxs = np.argpartition(score, len(score) - k)[-k:]

    m = m[best_score_idxs]
    z = z[best_score_idxs]
    m_signals = convert_spectrograms_to_audio(m)
    z_signals = convert_spectrograms_to_audio(z)
    m_specs = MinMaxNormaliser.denormalise(m)
    z_specs = MinMaxNormaliser.denormalise(z)
    for i, idx in enumerate(best_score_idxs):
        pretty_plot_spectrogram(m_specs[i], m_signals[i], title=f"Sample - {idx}")
        pretty_plot_spectrogram(z_specs[i], z_signals[i], title=f"Sample reconstruction - {idx}")


def generate_demonstration_data():
    idxs = [13, 16, 19, 20, 64, 75]

    vae = VAE.load("model_220624")
    original_specs, file_paths = load_dataset(num_samples=max(idxs) + 1, specific_file_idxs=idxs)
    reconstruction_specs = vae.model.predict(original_specs)
    play_sounds(original_specs,
                save_dir='../output/reconstruction_examples/original', print_names=file_paths, sleep_duration=1.)
    play_sounds(reconstruction_specs,
                save_dir='../output/reconstruction_examples/reconstruction', print_names=file_paths, sleep_duration=1.)

    original_specs = MinMaxNormaliser.denormalise(original_specs)
    reconstruction_specs = MinMaxNormaliser.denormalise(reconstruction_specs)
    for i in range(len(idxs)):
        pretty_plot_spectrogram(original_specs[i], title=f"Sample - {i + 1}")
        pretty_plot_spectrogram(reconstruction_specs[i], title=f"Sample reconstruction - {i + 1}")

    # Interpolation 1
    interpolated_specs_1, interpolated_latent_representations_1 = \
        interpolate_specs(num_samples=256, interpolation_size=9, vae=vae, spec_idxs=idxs[0:2])
    show_interpolation(interpolated_specs_1, convert_spectrograms_to_audio(interpolated_specs_1), title='1')
    play_interpolation(interpolated_specs_1, '../output/interpolation_examples')

    # Interpolation 2
    interpolated_specs_2, interpolated_latent_representations_2 = \
        interpolate_specs(num_samples=256, interpolation_size=9, vae=vae, spec_idxs=idxs[2:4])
    show_interpolation(interpolated_specs_2, convert_spectrograms_to_audio(interpolated_specs_2), title='2')
    play_interpolation(interpolated_specs_2, '../output/interpolation_examples')

    # Interpolation 3
    interpolated_specs_3, interpolated_latent_representations_3 = \
        interpolate_specs(num_samples=256, interpolation_size=9, vae=vae, spec_idxs=idxs[4:6])
    show_interpolation(interpolated_specs_3, convert_spectrograms_to_audio(interpolated_specs_3), title='3')
    play_interpolation(interpolated_specs_3, '../output/interpolation_examples')


if __name__ == '__main__':
    find_best_reconstructions()
