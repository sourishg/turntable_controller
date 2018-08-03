from __future__ import print_function
import sys
import params
from prepare_data import get_dataset_testing, get_task_relevant_feature
from model import TRFModel

import matplotlib.pyplot as plt
import numpy as np

FLAGS = params.FLAGS
H = FLAGS.seq_length
F = FLAGS.pred_length
num_rays = FLAGS.num_rays
num_samples = params.VARIATIONAL_SAMPLES


if __name__ == '__main__':
    x_test, y_test, u_test = get_dataset_testing(sys.argv[1])

    model = TRFModel(FLAGS.num_rays, FLAGS.seq_length,
                     FLAGS.pred_length, var_samples=num_samples,
                     task_relevant=FLAGS.task_relevant)

    if FLAGS.task_relevant:
        model.load_weights("vae_weights_tr_p1.h5")
    else:
        model.load_weights("vae_weights_p1.h5")

    encoder = model.get_encoder_model()
    decoder = model.get_decoder_model()
    latent_model = model.get_latent_model()
    transition_model = model.get_transition_model()

    for k in range(0, y_test.shape[0], 1):
        fig = plt.figure(figsize=(15, 8))
        plots = []

        output_dim = num_rays
        if FLAGS.task_relevant:
            output_dim = 1
            plt.ylim([0.0, 1.0])
        else:
            for p in range(F):
                ax = fig.add_subplot(3, F / 3 + 1, p + 1)
                ax.set_ylim([0.0, 1.0])
                ax.set_title("Timestep " + str(p + 1))
                plots.append(ax)

        y_true = np.reshape(y_test[k], (F, output_dim))

        for i in range(num_samples):
            b = encoder.predict(np.array([x_test[k][H-1], ]), batch_size=1)[0]
            y_pred = []
            for j in range(F):
                u = u_test[k][H-1+j]
                b = transition_model.predict([np.array([b, ]), np.array([u, ])], batch_size=1)[0]
                _, _, z = latent_model.predict(np.array([b, ]), batch_size=1)
                z = z[0]
                y = decoder.predict(np.array([z, ]), batch_size=1)[0]
                y_pred.append(y)

            if FLAGS.task_relevant:
                plt.plot([j for j in range(F)], [float(u) for u in y_pred], 'b.')
            else:
                for p in range(F):
                    plots[p].plot([j for j in range(num_rays)], [float(u) for u in y_pred[p]], 'b.')

        if FLAGS.task_relevant:
            plt.plot([j for j in range(F)], [float(u) for u in y_test[k]], 'r.')
        else:
            for p in range(F):
                plots[p].plot([j for j in range(FLAGS.num_rays)], [float(u) for u in y_true[p]], 'r.')

        plt.show()
