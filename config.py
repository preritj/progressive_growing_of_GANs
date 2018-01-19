from easydict import EasyDict as edict


# Training/validation parameters
TRAIN = True  # set to True for training + validation and False for testing
N_ITERS = 5000  # number of iterations to train
BATCH_SIZE = 64  # batch size for training
DISPLAY_PERIOD = 5  # Interval to display loss
SAVE_PERIOD = 1  # Interval to save model
SAVE_IMAGES = True  # if True, save images when model is saved
SAVE_DIR = r'D:\Data\pgan\models'  # path to save trained models
IMAGE_SAVE_DIR = r'D:\Data\pgan\gan_images'  # path to save GAN images
SUMMARY_DIR = r'D:\Data\pgan\train'

# Data Preprocessing and augmentation parameters
INPUT_SHAPE = (512, 512, 3)  # size of random crops used for training
FLIP = True  # applies random horizontal and vertical flips
ROTATE = True  # applies random rotations
PREPROCESS = 'min-max'  # can be 'min-max' or 'standard'

# Adam optimizer parameters:
LEARNING_RATE = 0.001
BETA1 = 0.
BETA2 = 0.99

# Other parameters
LEAKY_RELU_ALPHA = 0.2  # alpha in leaky relu
SMOOTH_LABEL = True  # uses 0.9 instead of 1 for positive labels
NOISE_STDDEV = 0.01  # standard deviation for noise added to images
Z_DIM = 128  # dim of latent space
# LOSS_MODE can be 'js' or 'wgan_gp'
# 'js' : Jensen-Shannon loss as in the original GAN paper
# 'wgan_gp' Wasserstein GAN loss with gradient penalty (Gulrajani et al)
LOSS_MODE = 'wgan_gp'
N_CRITIC = 3  # number of times to train disc for every gen train step
LAMBDA_GP = 10.  # Gradient penalty lambda hyperparameter
GAMMA_GP = 1.  # Gradient penalty gamma hyperparameter
MINIBATCH_STDDEV = True  # include minibatch std deviation as feature
NORM_D = None  # options are None, pixel_norm, batch_norm and layer_norm
NORM_G = 'pixel_norm'  # options are None, pixel_norm, batch_norm and layer_norm
USE_TANH = False  # use tanh in the final layer of generator
WEIGHT_SCALE = True  # use weight scaling for equalized learning rate
DRIFT_LOSS = True  # add a drift loss term
EPS_DRIFT = 0.001  # epsilon for drift loss term
FADE_ALPHA = 0.  # starting alpha to use for transition

RESOLUTION = 16  # this is the current resolution of network
MIN_RESOLUTION = 4  # min spatial resolution of features
NF_MIN = 32  # min depth of features
NF_MAX = 512  # max depth of features

TRANSITION = False  # whether to train in transition mode
LOAD_MODEL = None  # if not None, specify load sub-dir e.g. '16x16_transition'


###############################################################
# dataset specific parameters
###############################################################

# RGB mean
IMAGE_MEAN = [184.02, 157.45, 215.96]
IMAGE_STDDEV = [42.37, 48.23, 29.98]

# class encodings
CLASSES = {'Normal': 0,
           'Benign': 1,
           'InSitu': 2,
           'Invasive': 3}

# abreviations
CLASS_ABR = {'Normal': 'n',
             'Benign': 'b',
             'InSitu': 'is',
             'Invasive': 'iv'}

# validation files
VALIDATION_SET = {'Normal': ([i for i in range(46, 52)]
                             + [i for i in range(61, 69)]),
                  'Benign': [i for i in range(45, 59)],
                  'InSitu': [i for i in range(40, 54)],
                  'Invasive': ([i for i in range(50, 54)]
                               + [i for i in range(64, 74)])}

# Data directory
DATA_DIR = r'C:\Data\img'


#################################################################
cfg = edict({'data_dir': DATA_DIR,
             'summary_dir': SUMMARY_DIR,
             'image_mean': IMAGE_MEAN,
             'image_stddev': IMAGE_STDDEV,
             'preprocess': PREPROCESS,
             'classes': CLASSES,
             'class_abr': CLASS_ABR,
             'validation_set': VALIDATION_SET,
             'train': TRAIN,
             'input_shape': INPUT_SHAPE,
             'flip': FLIP,
             'rotate': ROTATE,
             'smooth_label': SMOOTH_LABEL,
             'noise_stddev': NOISE_STDDEV,
             'z_dim': Z_DIM,
             'loss_mode': LOSS_MODE,
             'lambda_gp': LAMBDA_GP,
             'gamma_gp': GAMMA_GP,
             'n_iters': N_ITERS,
             'batch_size': BATCH_SIZE,
             'leakyRelu_alpha': LEAKY_RELU_ALPHA,
             'learning_rate': LEARNING_RATE,
             'beta1': BETA1,
             'beta2': BETA2,
             'norm_d': NORM_D,
             'norm_g': NORM_G,
             'weight_scale': WEIGHT_SCALE,
             'drift_loss': DRIFT_LOSS,
             'eps_drift': EPS_DRIFT,
             'n_critic': N_CRITIC,
             'use_tanh': USE_TANH,
             'fade_alpha': FADE_ALPHA,
             'resolution': RESOLUTION,
             'min_resolution': MIN_RESOLUTION,
             'nf_min': NF_MIN,
             'nf_max': NF_MAX,
             'transition': TRANSITION,
             'load_model': LOAD_MODEL,
             'minibatch_stddev': MINIBATCH_STDDEV,
             'display_period': DISPLAY_PERIOD,
             'save_images': SAVE_IMAGES,
             'model_save_dir': SAVE_DIR,
             'image_save_dir': IMAGE_SAVE_DIR,
             'save_period': SAVE_PERIOD})
