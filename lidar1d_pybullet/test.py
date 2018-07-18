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
                     epochs=15, batch_size=1000, task_relevant=False)

    model_tr = TRFModel(FLAGS.num_rays, FLAGS.seq_length,
                        FLAGS.pred_length, var_samples=num_samples,
                        epochs=15, batch_size=1000, task_relevant=True)

    model_tr.load_weights("vae_weights_tr_p2.h5")
    model.load_weights("vae_weights_p2.h5")

    gen_model_tr = model_tr.get_gen_model()
    vae_model_tr = model_tr.get_vae_model()
    gen_model = model.get_gen_model()
    vae_model = model.get_vae_model()

    if FLAGS.task_relevant:
        latent_dim = model_tr.latent_dim
    else:
        latent_dim = model.latent_dim

    for k in range(0, y_test.shape[0], 1):
        fig = plt.figure(figsize=(15, 8))
        plots = []

        output_dim = num_rays
        if FLAGS.task_relevant:
            output_dim = 1
            plt.ylim([-1.0, 1.0])
        else:
            for p in range(F):
                ax = fig.add_subplot(3, F / 3 + 1, p + 1)
                ax.set_ylim([0, 1.0])
                ax.set_title("Timestep " + str(p + 1))
                plots.append(ax)

        y = np.reshape(y_test[k], (F, output_dim))

        if params.USE_ONLY_DECODER:
            for i in range(num_samples):
                prev_y = x_test[k][-1]
                if FLAGS.task_relevant:
                    prev_y = get_task_relevant_feature(prev_y, FLAGS.tr_half_width)
                outputs = []
                for p in range(F):
                    z = np.random.standard_normal(latent_dim)
                    u = u_test[k][H - 1 + p]
                    if FLAGS.task_relevant:
                        y_pred = gen_model_tr.predict([np.array([prev_y, ]),
                                                       np.array([u, ]),
                                                       np.array([z, ])],
                                                      batch_size=1)[0]
                    else:
                        y_pred = gen_model.predict([np.array([prev_y, ]),
                                                    np.array([u, ]),
                                                    np.array([z, ])],
                                                   batch_size=1)[0]
                    prev_y = y_pred

                    if not FLAGS.task_relevant:
                        plots[p].plot([j for j in range(FLAGS.num_rays)], [float(u) for u in y_pred], 'b.')
                    else:
                        outputs.append(y_pred)
                if FLAGS.task_relevant:
                    plt.plot([j for j in range(F)], [float(u) for u in outputs], 'b.')
        else:
            init_outputs = []
            prev_y = x_test[k][-1]
            for p in range(F):
                z = np.random.standard_normal(model.latent_dim)
                u = u_test[k][H - 1 + p]
                y_pred = gen_model.predict([np.array([prev_y, ]), np.array([u, ]), np.array([z, ])], batch_size=1)[0]
                init_outputs.append(y_pred)
                prev_y = y_pred

            init_outputs = np.array(init_outputs).astype('float32')
            prev_x = np.array(x_test[k]).astype('float32')
            x_enc = np.concatenate((prev_x, init_outputs))
            for i in range(num_samples):
                if FLAGS.task_relevant:
                    y_pred = vae_model_tr.predict([np.array([x_enc, ]), np.array([u_test[k], ])], batch_size=1)[0]
                    outputs = y_pred[H - 1:]
                    plt.plot([j for j in range(F)], [float(u) for u in outputs], 'b.')
                else:
                    y_pred = vae_model.predict([np.array([x_enc, ]), np.array([u_test[k], ])], batch_size=1)[0]
                    y_pred = np.reshape(y_pred, (H + F - 1, num_rays))
                    y_pred = y_pred[H - 1:, :]
                    for p in range(F):
                        plots[p].plot([j for j in range(num_rays)], [float(u) for u in y_pred[p]], 'b.')

        if FLAGS.task_relevant:
            plt.plot([j for j in range(F)], [float(u) for u in y_test[k]], 'r.')
        else:
            for p in range(F):
                plots[p].plot([j for j in range(FLAGS.num_rays)], [float(u) for u in y[p]], 'r.')

        plt.show()
