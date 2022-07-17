from typing import Tuple, Optional, List

import numpy as np

from autoencoder import VAE
from config import cfg
from soundgenerator import play_sounds
from train import load_dataset


def interpolate_specs(
        num_samples: int,
        interpolation_size: int,
        vae: VAE,
        spec_idxs: Optional[List[int]] = None) \
        -> Tuple[np.ndarray, np.ndarray]:
    if spec_idxs is None:
        specs, spec_idxs = load_dataset(num_samples=num_samples, num_rand=2)
    else:
        specs, spec_idxs = load_dataset(num_samples=num_samples, specific_file_idxs=spec_idxs)
    latent_representations = vae.encoder.predict(specs)

    print(f"Interpolating from {spec_idxs[0]} to {spec_idxs[1]} in {interpolation_size} steps")
    print(f"Source latent embedding: {latent_representations[0]}")
    print(f"Target latent embedding: {latent_representations[1]}")

    interpolated_latent_representations = \
        np.linspace(latent_representations[0], latent_representations[1], interpolation_size)

    interpolated_specs = vae.decoder.predict(interpolated_latent_representations)
    return interpolated_specs, interpolated_latent_representations


if __name__ == '__main__':
    # Generate latent representations
    interpolated_specs, interpolated_latent_representations = \
        interpolate_specs(num_samples=256, interpolation_size=11, vae=VAE.load("model_220620"))

    # Add additional source and target examples
    output_specs = np.stack([
        interpolated_specs[0], interpolated_specs[0], cfg.EMPTY_SPEC,
        interpolated_specs[-1], interpolated_specs[-1], cfg.EMPTY_SPEC,
        *interpolated_specs
    ])

    # Play
    play_sounds(output_specs)
