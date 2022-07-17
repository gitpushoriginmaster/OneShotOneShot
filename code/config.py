from dataclasses import dataclass, field
from os import getcwd, pardir
from os.path import join
from typing import Dict, Any, Tuple, Optional

import numpy as np
from openpyxl.compat.singleton import Singleton


@dataclass
class Config(metaclass=Singleton):
    ##############
    # PREPROCESS #
    ##############
    N_FFT: int = 2 ** 8
    HOP_LENGTH: int = 2 ** 7
    DURATION: float = 1.0
    SAMPLE_RATE: int = 16_000
    EPS: float = 1e-10
    SPECTROGRAMS_SAVE_DIR: str = "../datasets/one_shot_percussive_sounds/spectrograms_db/"
    FILES_DIR: str = "../datasets/one_shot_percussive_sounds/audio/"
    TOP_DB: float = 80.
    NORM_RANGE: Tuple[float, float] = (0., 1.)
    DB_RANGE: Tuple[float, float] = (-TOP_DB, 0.)
    SPECTROGRAM_SHAPE: Tuple[int, int, int] = (128, 128, 1)
    EMPTY_SPEC: np.ndarray = np.zeros(SPECTROGRAM_SHAPE)

    ###############
    # AUTOENCODER #
    ###############
    IS_VARIATIONAL: bool = True
    RECONSTRUCTION_LOSS_WEIGHT: float = 1e-7
    NUM_HIDDEN_CONV_LAYERS: int = 7
    FILTER_SIZE_CONV_LAYERS: int = 4
    LATENT_SPACE_DIM: int = 4
    CONV_PARAMS_MIDDLE: Dict[str, Any] = field(default_factory=lambda: {
        'kernel_size': (6, 3),
        'strides': (2, 1),
        'padding': 'same',
        'activation': 'elu',
        'kernel_initializer': 'glorot_normal',
    })
    CONV_PARAMS_END: Dict[str, Any] = field(default_factory=lambda: {
        'kernel_size': 6,
        'strides': 1,
        'padding': 'same',
        'activation': 'softplus',
        'kernel_initializer': 'glorot_normal',
    })
    USE_MOCK_ENCODER: bool = False
    USE_MOCK_DECODER: bool = False
    ARCH_PLOT_ARGS: Dict[str, bool] = field(default_factory=lambda: {
        'show_layer_activations': True,
        'show_shapes': True,
    })
    LOGS_DIR: str = join(getcwd(), pardir, "logs")
    CLIP_PREDICTION: bool = True
    LOSS_SQUARED: bool = False
    LOSS_MSE_LINEAR_WEIGHT: Optional[Tuple[float, float]] = (2., 1.)
    USE_LSTM: bool = True

    #########
    # TRAIN #
    #########
    VERBOSE: bool = True
    LEARNING_RATE: float = 1e-3
    BATCH_SIZE: int = 2 ** 6
    PRETRAIN_EPOCHS: int = 500
    EPOCHS: int = 2000
    SPECTROGRAMS_PATH: str = "../datasets/one_shot_percussive_sounds/spectrograms_db/"
    PRETRAIN_DATA: np.ndarray = \
        np.tile(np.linspace(NORM_RANGE[1], NORM_RANGE[0], SPECTROGRAM_SHAPE[0])[..., None], SPECTROGRAM_SHAPE[1]).T[None, ..., None]

    ############
    # GENERATE #
    ############
    SAVE_DIR_ORIGINAL: str = "../output/samples/original/"
    SAVE_DIR_GENERATED: str = "../output/samples/generated/"
    NORMALIZATION_FACTOR: float = 2 ** 16


cfg = Config()
