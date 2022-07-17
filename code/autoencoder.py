from __future__ import annotations

import pickle
from datetime import datetime
from os import makedirs
from os.path import join
from typing import Optional, Tuple, Dict, Any

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Reshape, Conv2DTranspose, Lambda, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras import losses, metrics
from tensorflow.python.keras.callbacks import TensorBoard, History
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.losses import LossFunctionWrapper
from tensorflow.python.keras.utils.vis_utils import plot_model

from config import cfg

tf.compat.v1.disable_eager_execution()


# NOTE: Can be upgraded to introduce Kapre integration.

class VAE:
    """
    VAE represents a Deep Convolutional variational autoencoder architecture
    with mirrored encoder and decoder components.
    """

    def __init__(self,
                 input_shape: Tuple[int, int, int] = cfg.SPECTROGRAM_SHAPE,
                 latent_space_dim: int = cfg.LATENT_SPACE_DIM,
                 is_variational: bool = cfg.IS_VARIATIONAL,
                 reconstruction_loss_weight: float = cfg.RECONSTRUCTION_LOSS_WEIGHT,
                 num_hidden_conv_layers: int = cfg.NUM_HIDDEN_CONV_LAYERS,
                 filter_size_conv_layers: int = cfg.FILTER_SIZE_CONV_LAYERS,
                 conv_params_middle: Optional[Dict[str, Any]] = None,
                 conv_params_end: Optional[Dict[str, Any]] = None,
                 use_mock_encoder: bool = cfg.USE_MOCK_ENCODER,
                 use_mock_decoder: bool = cfg.USE_MOCK_DECODER,
                 clip_prediction: bool = cfg.CLIP_PREDICTION,
                 loss_squared: bool = cfg.LOSS_SQUARED,
                 loss_mse_linear_weight: Optional[Tuple[float, float]] = cfg.LOSS_MSE_LINEAR_WEIGHT,
                 use_lstm: bool = cfg.USE_LSTM,
                 ):

        if conv_params_middle is None:
            conv_params_middle = cfg.CONV_PARAMS_MIDDLE
        if conv_params_end is None:
            conv_params_end = cfg.CONV_PARAMS_END
        assert len(input_shape) == 3

        # Input parameters
        self.input_shape: Tuple[int, int, int] = input_shape
        self.latent_space_dim: int = latent_space_dim
        self._is_variational: bool = is_variational
        self._reconstruction_loss_weight: float = reconstruction_loss_weight if self._is_variational else 0.
        self._num_hidden_conv_layers: int = num_hidden_conv_layers
        self._filter_size_conv_layers: int = filter_size_conv_layers
        self._conv_params_middle: Dict[str, Any] = conv_params_middle
        self._conv_params_end: Dict[str, Any] = conv_params_end
        self._use_mock_encoder: bool = use_mock_encoder
        self._use_mock_decoder: bool = use_mock_decoder
        self._clip_prediction: bool = clip_prediction
        self._loss_squared: bool = loss_squared
        self._loss_mse_linear_weight: Optional[Tuple[float, float]] = loss_mse_linear_weight
        self._loss_mse_weight_factor: np.ndarray = \
            np.linspace(self._loss_mse_linear_weight[0], self._loss_mse_linear_weight[1], self.input_shape[0])[
                None, ..., None, None] if self._loss_mse_linear_weight is not None else np.ones((1, *self.input_shape))
        self._use_lstm: bool = use_lstm

        # Build model
        self.encoder: Optional[Model] = None
        self.decoder: Optional[Model] = None
        self.model: Optional[Model] = None
        self._encoder_input_layer: Input = None
        self._loss_mse: LossFunctionWrapper = losses.MeanSquaredError()
        self._loss_kl: LossFunctionWrapper = losses.KLDivergence()
        self._build()

        # Set TensorBoard
        self._log_dir: str = join(cfg.LOGS_DIR, f"model_{datetime.now().strftime('%y%m%d')}")
        self._tensorboard_callback: TensorBoard = tf.keras.callbacks.TensorBoard(log_dir=self._log_dir,
                                                                                 histogram_freq=1)

    def summary(self) -> None:

        self.model.summary(line_length=100)
        print()
        self.encoder.summary(line_length=100)
        print()
        self.decoder.summary(line_length=100)
        print()

    def compile(self, learning_rate: float) -> None:
        if cfg.VERBOSE:
            print(f"Compiling VAE with {learning_rate=}")
        self.model.compile(optimizer=Adam(learning_rate=learning_rate),
                           loss=self._calculate_combined_loss,
                           metrics=[metrics.MeanSquaredError(),
                                    metrics.KLDivergence()])

    def _calculate_combined_loss(self, y_target: np.ndarray, y_predicted: np.ndarray) -> float:

        if self._loss_squared:
            y_target = tf.math.square(y_target)
            y_predicted = tf.math.square(y_predicted)

        y_target = y_target * self._loss_mse_weight_factor
        y_predicted = y_predicted * self._loss_mse_weight_factor

        return self._loss_mse(y_target, y_predicted) + \
               self._loss_kl(y_target, y_predicted) * self._reconstruction_loss_weight

    def train(self, x_train: np.ndarray, batch_size: int, num_epochs: int, should_callback: bool = True) -> History:
        history = self.model.fit(x=x_train,
                                 y=x_train,
                                 batch_size=batch_size,
                                 epochs=num_epochs,
                                 shuffle=True,
                                 callbacks=[self._tensorboard_callback] if should_callback else None,
                                 verbose=cfg.VERBOSE,
                                 )
        return history

    def reconstruct(self, images: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        assert images.shape[1:] == self.input_shape
        latent_representations = self.encoder.predict(images)
        reconstructed_images = self.decoder.predict(latent_representations)
        if self._clip_prediction:
            reconstructed_images = np.clip(reconstructed_images, a_min=cfg.NORM_RANGE[0], a_max=cfg.NORM_RANGE[1])
        return reconstructed_images, latent_representations

    def save(self) -> None:

        print(f"tensorboard --logdir {self._tensorboard_callback.log_dir}")
        print(f"Model saved to {self._log_dir}")

        makedirs(self._log_dir, exist_ok=True)
        # Save weights
        self.model.save_weights(join(self._log_dir, "weights.h5"))
        # Save parameters
        self._save_parameters()

    def _save_parameters(self):
        parameters = {
            'input_shape': self.input_shape,
            'latent_space_dim': self.latent_space_dim,
            'is_variational': self._is_variational,
            'reconstruction_loss_weight': self._reconstruction_loss_weight,
            'num_hidden_conv_layers': self._num_hidden_conv_layers,
            'filter_size_conv_layers': self._filter_size_conv_layers,
            'conv_params_middle': self._conv_params_middle,
            'conv_params_end': self._conv_params_end,
            'use_mock_encoder': self._use_mock_encoder,
            'use_mock_decoder': self._use_mock_decoder,
            'clip_prediction': self._clip_prediction,
            'loss_squared': self._loss_squared,
            'loss_mse_linear_weight': self._loss_mse_linear_weight,
            'use_lstm': self._use_lstm,
        }

        with open(join(self._log_dir, "parameters.pkl"), "wb") as f:
            pickle.dump(parameters, f)

    @classmethod
    def load(cls, save_folder: str) -> VAE:
        save_folder = join(cfg.LOGS_DIR, save_folder)

        with open(join(save_folder, "parameters.pkl"), "rb") as f:
            parameters = pickle.load(f)
        vae = VAE(**parameters)

        weights_path = join(save_folder, "weights.h5")
        vae.model.load_weights(weights_path)
        print(f"Loaded VAE from {save_folder}")
        return vae

    def _build(self) -> None:
        self._build_encoder()
        self._build_decoder()
        self._build_autoencoder()

    def _build_autoencoder(self) -> None:
        decoder_output = self.decoder(self.encoder(self._encoder_input_layer))
        self.model = Model(self._encoder_input_layer, decoder_output, name="vae")

    def _build_encoder(self) -> None:
        self._encoder_input_layer = Input(shape=self.input_shape, name='encoder_input')

        # Convolution layers
        x = Conv2D(
            filters=self._filter_size_conv_layers,
            **self._conv_params_middle,
            name="encoder_conv2d_1")(self._encoder_input_layer)
        for i in range(self._num_hidden_conv_layers - 1):
            x = Conv2D(
                filters=self._filter_size_conv_layers,
                **self._conv_params_middle,
                name=f"encoder_conv2d_{i + 2}")(x)

        self._smallest_convolution_shape = x.shape[1:]  # Calculate value for the decoder to use

        if self._use_mock_encoder:
            x = Flatten(name="mock_encoder_flatten")(self._encoder_input_layer)
            bottleneck = Dense(self.latent_space_dim, trainable=False, name="mock_encoder_dense")(x)
            self.encoder = Model(self._encoder_input_layer, bottleneck, name="mock_encoder")

        else:
            if self._use_lstm:
                if x.shape[1] > 1:
                    x = Conv2D(
                        filters=self._filter_size_conv_layers,
                        kernel_size=(x.shape[1], 1),
                        strides=(x.shape[1], 1),
                        padding=self._conv_params_middle['padding'],
                        activation=self._conv_params_middle['activation'],
                        kernel_initializer=self._conv_params_middle['kernel_initializer'],
                        name=f"encoder_conv2d_last")(x)
                x = Reshape(x.shape[2:], name="encoder_reshape_lstm")(x)
                x = LSTM(128, return_sequences=False, name='encoder_lstm')(x)
            x = Flatten(name="encoder_flatten_1")(x)
            bottleneck = self._add_bottleneck(x)
            self.encoder = Model(self._encoder_input_layer, bottleneck, name="encoder")

    def _add_bottleneck(self, x: Layer) -> Layer:
        """Flatten data and add bottleneck with Gaussian sampling (Dense layer)."""
        if self._is_variational:
            # Implement a VARIATIONAL autoencoder
            self.mu = Dense(self.latent_space_dim, name="encoder_mu")(x)
            self.log_variance = Dense(self.latent_space_dim, name="encoder_log_variance")(x)

            def _sample_point_from_normal_distribution(args):
                mu, log_variance = args
                epsilon = K.random_normal(shape=K.shape(mu), mean=0., stddev=1.)
                sampled_point = mu + K.exp(log_variance / 2) * epsilon
                return sampled_point

            x = Lambda(_sample_point_from_normal_distribution, name="encoder_output")([self.mu, self.log_variance])

        else:
            # Implement a NON-VARIATIONAL autoencoder
            x = Dense(self.latent_space_dim, name="encoder_output")(x)

        return x

    def _build_decoder(self) -> None:
        decoder_input = Input(shape=(self.latent_space_dim,), name='decoder_input')
        if self._use_mock_decoder:
            x = Dense(np.product(self.input_shape), trainable=False, name="mock_decoder_dense")(decoder_input)
            x = Reshape(self.input_shape, name="mock_decoder_reshape")(x)
            self.decoder = Model(decoder_input, x, name="mock_decoder")
        else:
            x = Dense(np.product(self._smallest_convolution_shape), name="decoder_dense_1")(decoder_input)
            x = Reshape(self._smallest_convolution_shape, name="decoder_reshape_1")(x)

            # Convolution layers
            for i in range(self._num_hidden_conv_layers-1):
                x = Conv2DTranspose(
                    filters=self._filter_size_conv_layers,
                    **self._conv_params_middle,
                    name=f"decoder_conv2d_t_{i + 1}")(x)
            x = Conv2DTranspose(
                filters=self.input_shape[-1],
                **self._conv_params_middle,
                name=f"decoder_conv2d_t_{self._num_hidden_conv_layers}")(x)

            x = Conv2D(
                filters=self.input_shape[-1],
                **self._conv_params_end,
                name="decoder_output")(x)
            self.decoder = Model(decoder_input, x, name="decoder")

    def _plot_architecture(self):
        plot_model(self.model, to_file='ae_model.png', **cfg.ARCH_PLOT_ARGS)
        plot_model(self.model, to_file='ae_nested_model.png', expand_nested=False, **cfg.ARCH_PLOT_ARGS)
        plot_model(self.encoder, to_file='encoder_model.png', **cfg.ARCH_PLOT_ARGS)
        plot_model(self.decoder, to_file='decoder_model.png', **cfg.ARCH_PLOT_ARGS)


if __name__ == '__main__':
    autoencoder = VAE(
        input_shape=cfg.SPECTROGRAM_SHAPE,  # (Frequency, Time, Channels),
    )
    autoencoder.summary()
