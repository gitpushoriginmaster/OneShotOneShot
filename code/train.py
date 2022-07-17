import os
import sys
from typing import List, Tuple, Optional

import numpy as np
from tensorflow.python.keras.callbacks import History

from autoencoder import VAE
from config import cfg


def load_dataset(
        spectrograms_path: str = cfg.SPECTROGRAMS_PATH,
        num_samples: int = sys.maxsize,
        num_rand: int = sys.maxsize,
        specific_file_idxs: Optional[List[int]] = None) \
        -> Tuple[np.ndarray, List[str]]:

    assert not(num_rand < sys.maxsize and specific_file_idxs is not None)

    x_train = []
    file_paths = []

    root, _, file_names = next(os.walk(spectrograms_path))
    file_names = file_names[:num_samples]

    if num_rand < num_samples:
        # Random sample from specific set of files
        random_idxs = np.random.choice(range(num_samples), size=num_rand, replace=False)
        file_names = np.take(file_names, random_idxs)
        num_samples = num_rand
    if specific_file_idxs is not None:
        file_names = [file_names[i] for i in specific_file_idxs]

    for file_name in file_names[:num_samples]:

        file_path = os.path.join(root, file_name)
        file_paths.append(file_path)

        spec = np.load(file_path)  # (n_bins, n_frames, 1)
        x_train.append(spec)

    x_train = np.array(x_train)  # -> (num_samples, n_bins, n_frames, 1)

    return x_train, file_paths


def train(
        x_train: np.ndarray,
        learning_rate: float,
        batch_size: int,
        epochs: int,
        should_pretrain: bool = False) \
        -> Tuple[VAE, History]:
    vae = VAE()
    if cfg.VERBOSE:
        vae.summary()

    vae.compile(learning_rate=learning_rate)

    # Pre-train
    if should_pretrain:
        x_pretrain = cfg.PRETRAIN_DATA
        vae.train(x_pretrain, 1, cfg.PRETRAIN_EPOCHS, should_callback=False)

    # Train
    history = vae.train(x_train, batch_size, epochs)
    return vae, history


if __name__ == '__main__':
    num_samples = 512  # sys.maxsize

    x_train, _ = load_dataset(spectrograms_path=cfg.SPECTROGRAMS_PATH, num_samples=num_samples)
    autoencoder, _ = train(x_train, cfg.LEARNING_RATE, cfg.BATCH_SIZE, cfg.EPOCHS, should_pretrain=True)
    autoencoder.save()
