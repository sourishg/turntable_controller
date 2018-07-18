from tensorflow.python.platform import flags

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
flags.DEFINE_float('tr_half_width', 20,
                   'Half width range of ray from which to select task-relevant feature')
flags.DEFINE_bool('task_relevant', True,
                  'Whether or not to predict task relevant features')

TRAINED = True
USE_ONLY_DECODER = False
VARIATIONAL_SAMPLES = 30
