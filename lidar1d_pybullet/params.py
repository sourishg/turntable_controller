from tensorflow.python.platform import flags

# Network parameters

FLAGS = flags.FLAGS

flags.DEFINE_integer('seq_length', 5,
                     'Length of input sequence')
flags.DEFINE_integer('pred_length', 10,
                     'Length of prediction')
flags.DEFINE_integer('num_rays', 100,
                     'Length of prediction')
flags.DEFINE_float('train_val_split', 0.8,
                   'Training/validation split ratio')
flags.DEFINE_float('latent_multiplier', 1e-03,
                   'Multiplier for KL cost')
flags.DEFINE_float('latent_std_min', -5.0,
                   'Min log var of latent variables')
flags.DEFINE_float('tr_half_width', 30,
                   'Half width range of ray from which to select task-relevant feature')
flags.DEFINE_bool('task_relevant', False,
                  'Whether or not to predict task relevant features')

TRAINED = False
USE_ONLY_DECODER = False
VARIATIONAL_SAMPLES = 30

# World and control paramters

NUM_OBSTACLES = 10
WORLD_SAMPLES = 1
CONTROL_SAMPLES = 50
MAX_ANGULAR_VELOCITY = 1.0
TIME_INCREMENT = 0.1
TOTAL_CONTROL_TIMESTEPS = 10
LIDAR_POS = (0.0, 0.0, 0.2)
LIDAR_THETA_RANGE_DEG = 120
NUM_RAYS = 100
LIDAR_RANGE = 5.0
