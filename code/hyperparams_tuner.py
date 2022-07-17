import pandas as pd
import talos

from config import cfg
from train import load_dataset, train

cfg.VERBOSE = False


def experiment_convolution():
    num_samples = 100
    cfg.EPOCHS = 100
    x_train, _ = load_dataset(spectrograms_path=cfg.SPECTROGRAMS_PATH, num_samples=num_samples)

    talos_params = {
        'reconstruction_loss_weight': [1e-2, 1e-3, 1e-4],
        'num_hidden_conv_layers': [4, 5, 6],
        'latent_space_dim': [2, 3, 4],
        'conv_params_middle_kernel_size': [(6, 6), (6, 3), (6, 1), (3, 3), (3, 1)],
        'conv_params_middle_activation': ['relu', 'elu', 'tanh'],
        'conv_params_end_kernel_size': [6, 3],
        'conv_params_end_activation': ['relu', 'sigmoid', 'softplus'],
    }

    def vae_model(x_train, y_train, x_val, y_val, params):
        cfg.RECONSTRUCTION_LOSS_WEIGHT = params['reconstruction_loss_weight']
        cfg.NUM_HIDDEN_CONV_LAYERS = params['num_hidden_conv_layers']
        cfg.LATENT_SPACE_DIM = params['latent_space_dim']
        cfg.CONV_PARAMS_MIDDLE['kernel_size'] = params['conv_params_middle_kernel_size']
        cfg.CONV_PARAMS_MIDDLE['activation'] = params['conv_params_middle_activation']
        cfg.CONV_PARAMS_END['kernel_size'] = params['conv_params_end_kernel_size']
        cfg.CONV_PARAMS_END['activation'] = params['conv_params_end_activation']

        autoencoder, out = \
            train(x_train, cfg.LEARNING_RATE, cfg.BATCH_SIZE, cfg.EPOCHS, should_pretrain=True)

        return out, autoencoder

    scan_object = talos.Scan(x=x_train,
                             y=x_train,
                             model=vae_model,
                             params=talos_params,
                             experiment_name='vae',

                             fraction_limit=0.999,
                             time_limit='2022-06-22 14:00',
                             print_params=True,

                             # reduction_method='correlation',
                             # reduction_interval=50,
                             # reduction_window=25,
                             # reduction_threshold=0.2,
                             # reduction_metric='mae',
                             # minimize_loss=True,
                             )

    return scan_object


def experiment_loss():
    num_samples = 256
    cfg.EPOCHS = 200
    x_train, _ = load_dataset(spectrograms_path=cfg.SPECTROGRAMS_PATH, num_samples=num_samples)

    talos_params = {
        'reconstruction_loss_weight': [1e-3, 1e-5, 1e-7],
        'reduce_depth': [True, False],
        'loss_squared': [True, False],
        'loss_mse_linear_weight': [(5., 1.), (2., 1.), (1., 1.)],
    }

    def vae_model(x_train, y_train, x_val, y_val, params):
        cfg.RECONSTRUCTION_LOSS_WEIGHT = params['reconstruction_loss_weight']
        cfg.REDUCE_DEPTH = params['reduce_depth']
        cfg.LOSS_SQUARED = params['loss_squared']
        cfg.LOSS_MSE_LINEAR_WEIGHT = params['loss_mse_linear_weight']

        autoencoder, out = \
            train(x_train, cfg.LEARNING_RATE, cfg.BATCH_SIZE, cfg.EPOCHS, should_pretrain=True)

        return out, autoencoder

    scan_object = talos.Scan(x=x_train,
                             y=x_train,
                             model=vae_model,
                             params=talos_params,
                             experiment_name='vae',

                             fraction_limit=0.999,
                             time_limit='2022-06-23 23:30',
                             print_params=True,

                             # reduction_method='correlation',
                             # reduction_interval=50,
                             # reduction_window=25,
                             # reduction_threshold=0.2,
                             # reduction_metric='mae',
                             # minimize_loss=True,
                             )

    return scan_object


if __name__ == '__main__':
    scan = experiment_loss()
    data = talos.Analyze(scan).data
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    print(data)
