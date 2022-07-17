import numpy as np

from autoencoder import VAE
from config import cfg
from soundgenerator import SoundGenerator, convert_spectrograms_to_audio, play_sounds, save_signals
from train import load_dataset


def select_spectrograms(num_spectrograms: int = 10) -> np.ndarray:

    spectrograms, spectrograms_paths = load_dataset(num_samples=100)

    num_spectrograms = min(num_spectrograms, len(spectrograms))

    sampled_indexes = np.random.choice(range(len(spectrograms)), num_spectrograms, replace=False)
    sampled_spectrograms = spectrograms[sampled_indexes]
    spectrograms_paths = [spectrograms_paths[index] for index in sampled_indexes]
    print(spectrograms_paths)

    return sampled_spectrograms


if __name__ == '__main__':
    # initialise sound generator
    vae = VAE.load("model_220609")
    sound_generator = SoundGenerator(vae)

    # Sample spectrograms
    sampled_specs = select_spectrograms(num_spectrograms=10)
    # sampled_specs = specs

    plain_reconstructed_signals = convert_spectrograms_to_audio(sampled_specs)

    vae_reconstructed_signals, _ = sound_generator.generate_from_spec(sampled_specs)

    # for spec, file_path in zip(specs, file_paths):
    #     s = spec[..., 0] + spec[..., 1] * 1.j
    #     fig, ax = plt.subplots()
    #     img = librosa.display.specshow(librosa.amplitude_to_db(s, ref=np.max), y_axis='linear', x_axis='time', ax=ax)
    #     ax.set_title(f"Power spectrogram - {os.path.basename(file_path)}")
    #     fig.colorbar(img, ax=ax, format="%+2.0f dB")
    #     plt.xlim(left=0., right=1.)
    #     plt.ylim(bottom=0.)
    #     plt.show()

    # save audio signals
    save_signals(plain_reconstructed_signals, cfg.SAVE_DIR_ORIGINAL)  # Plain ISTFT
    save_signals(vae_reconstructed_signals, cfg.SAVE_DIR_GENERATED)  # Through VAE

    output_signals = []
    for a, b in zip(plain_reconstructed_signals, vae_reconstructed_signals):
        output_signals.append(a)
        output_signals.append(b)
    play_sounds(output_signals, is_spec=False)
