import sys
import numpy as np
import matplotlib.pyplot as plt
from prepare_data import get_dataset_training, get_task_relevant_feature
from model import TRFModel
import params

FLAGS = params.FLAGS
H = FLAGS.seq_length
F = FLAGS.pred_length
num_rays = FLAGS.num_rays
num_samples = params.VARIATIONAL_SAMPLES
epochs = 50
batch_size = 1000

if __name__ == '__main__':
    train_file = sys.argv[1]
    test_file = sys.argv[2]

    if not params.TRAINED:
        x, y, u = get_dataset_training(train_file)
        idx = np.arange(x.shape[0])
        np.random.shuffle(idx)
        train_samples = int(FLAGS.train_val_split * x.shape[0])
        train_idx = idx[:train_samples]
        val_idx = idx[train_samples:]

        x_train, x_val = x[train_idx, :], x[val_idx, :]
        y_train, y_val = y[train_idx, :], y[val_idx, :]
        u_train, u_val = u[train_idx, :], u[val_idx, :]

    x_test, y_test, u_test = get_dataset_training(test_file)

    if not params.TRAINED:
        print(x_train.shape, y_train.shape, u_train.shape)
        print(x_val.shape, y_val.shape, u_val.shape)

    model = TRFModel(FLAGS.num_rays, FLAGS.seq_length,
                     FLAGS.pred_length, var_samples=num_samples,
                     epochs=epochs, batch_size=batch_size, task_relevant=False)

    if params.TRAINED:
        # load weights into new model
        if FLAGS.task_relevant:
            model.load_weights("vae_weights_p0.h5")
        else:
            model.load_weights("vae_weights_p0.h5")
    else:
        model.train_model(x_train, x_val,
                          y_train, y_val,
                          u_train, u_val)

    vae = model.get_vae_model()

    if FLAGS.task_relevant:
        for k in range(0, y_test.shape[0], 1):
            fig = plt.figure()
            plt.ylim((0.0, 1.0))
            y_true = y_test[k]
            for i in range(num_samples):
                y_pred = vae.predict([np.array([x_test[k], ]), np.array([u_test[k], ])], batch_size=1)[0]
                y_pred = np.reshape(y_pred, (H + F - 1, model.output_dim))
                y_tr = []
                for yp in y_pred:
                    y_tr.append(get_task_relevant_feature(yp, FLAGS.tr_half_width))
                plt.plot([j for j in range(H + F - 1)], [float(u) for u in y_tr], 'b.')

            plt.plot([j for j in range(H + F - 1)], [float(u) for u in y_true], 'r.')

            plt.show()
    else:
        for k in range(0, y_test.shape[0], 1):
            fig = plt.figure(figsize=(15, 8))
            y = np.reshape(y_test[k], (H + F - 1, model.output_dim))
            plots = []
            num_plots = H + F - 1
            for p in range(num_plots):
                ax = fig.add_subplot(3, num_plots / 3 + 1, p + 1)
                ax.set_ylim([0, 1.0])
                ax.set_title("Timestep " + str(p + 1))
                plots.append(ax)

            for i in range(num_samples):
                y_pred = vae.predict([np.array([x_test[k], ]), np.array([u_test[k], ])], batch_size=1)
                y_pred = np.reshape(y_pred[0], (H + F - 1, model.output_dim))
                for p in range(num_plots):
                    plots[p].plot([j for j in range(num_rays)], [float(u) for u in y_pred[p]], 'b.')

            for p in range(num_plots):
                plots[p].plot([j for j in range(num_rays)], [float(u) for u in y[p]], 'r.')

            plt.show()
