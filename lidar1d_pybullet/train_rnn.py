from vae_rnn import DeepVAE
import numpy as np
import sys

# Constants and params
theta_range = np.deg2rad(120.0)
num_rays = 100

trained = True
H = 9  # no of past observations
F = 1  # no of future predictions
num_samples = 30
training_data_fraction = 0.7

def prepareDataset(train_file, test_file):
    x_train, x_val, y_train, y_val, x_test, y_test = [], [], [], [], [], []
    u_train, u_val, u_test = [], [], []

    if not trained:
        f1 = open(train_file, "r")
        x_raw = []
        y_raw = []
        u_raw = []
        lines = [line.rstrip('\n') for line in f1]
        for i in range(len(lines) - H - F):
            print("preparing training data point", i)
            x = []
            y = []
            u = []
            for j in range(H):
                parts = lines[i + j].split(" ")
                x_sum = 0
                x1 = []
                for k in range(num_rays):
                    x_sum = x_sum + float(parts[k+2])
                    x1.append(x_sum)
                x.append(x1)
                #x.append(parts[1:num_rays+2:1])
                #u.append(float(parts[0]))
            for j in range(F):
                parts = lines[i + j + H].split(" ")
                y_sum = 0
                y1 = []
                for k in range(num_rays):
                    y_sum = y_sum + float(parts[k+2])
                    y1.append(y_sum)
                y.append(y1)
                #y.append(parts[2:num_rays+2:1])
                u.append(float(parts[1]))
            
            x = np.asarray(x)
            # u = np.asarray([u])
            # x = np.concatenate((x, u.T), axis=1)
            # x = x.flatten('F')

            y = np.asarray(y)
            y = y.flatten('F')

            x_raw.append(x)
            y_raw.append(y)
            u_raw.append(u)

        x = np.asarray(x_raw, dtype=np.float32)
        y = np.asarray(y_raw, dtype=np.float32)
        u = np.asarray(u_raw, dtype=np.float32)

        n = len(lines) - H - F
        n_train_samples = int(training_data_fraction * n)
        n_val_samples = n - n_train_samples

        
        training_idx = np.random.randint(x.shape[0], size=n_train_samples)
        val_idx = np.random.randint(x.shape[0], size=n_val_samples)

        x_train, x_val = x[training_idx, :], x[val_idx, :]
        y_train, y_val = y[training_idx, :], y[val_idx, :]
        u_train, u_val = u[training_idx, :], u[val_idx, :]
        
        '''
        x_train, x_val = x[:n_train_samples], x[n_train_samples:]
        y_train, y_val = y[:n_train_samples], y[n_train_samples:]
        u_train, u_val = u[:n_train_samples], u[n_train_samples:]
        '''
        print("Prepared training dataset!")

    f2 = open(test_file, "r")
    x_raw = []
    y_raw = []
    u_raw = []
    lines = [line.rstrip('\n') for line in f2]
    for i in range(len(lines) - H - F):
        print("preparing testing data point", i)
        x = []
        y = []
        u = []
        for j in range(H):
            parts = lines[i + j].split(" ")
            x_sum = 0
            x1 = []
            for k in range(num_rays):
                x_sum = x_sum + float(parts[k+2])
                x1.append(x_sum)
            x.append(x1)
            #x.append(parts[1:num_rays+2:1])
            #u.append(float(parts[0]))
        for j in range(F):
            parts = lines[i + j + H].split(" ")
            y_sum = 0
            y1 = []
            for k in range(num_rays):
                y_sum = y_sum + float(parts[k+2])
                y1.append(y_sum)
            y.append(y1)
            #y.append(parts[2:num_rays+2:1])
            u.append(float(parts[1]))

        x = np.asarray(x)
        # u = np.asarray([u])
        # x = np.concatenate((x, u.T), axis=1)
        # x = x.flatten('F')

        y = np.asarray(y)
        y = y.flatten('F')
        
        x_raw.append(x)
        y_raw.append(y)
        u_raw.append(u)

    x_test = np.asarray(x_raw, dtype=np.float32)
    y_test = np.asarray(y_raw, dtype=np.float32)
    u_test = np.asarray(u_raw, dtype=np.float32)

    print("Prepared testing dataset!")

    return x_train, y_train, x_val, y_val, x_test, y_test, u_train, u_val, u_test

if __name__ == '__main__':
    x_train, y_train, x_val, y_val, x_test, y_test, u_train, u_val, u_test = prepareDataset(sys.argv[1], sys.argv[2])
    '''
    if not trained:
        u_train = np.tile(u_train[...,:], (1, num_rays))
        u_val = np.tile(u_val[...,:], (1, num_rays))
    u_test = np.tile(u_test[...,:], (1, num_rays))
    '''
    if not trained:
        x_train = x_train / 500.0
        x_val = x_val / 500.0
        y_train = y_train / 500.0
        y_val = y_val / 500.0

    x_test = x_test / 500.0
    y_test = y_test / 500.0

    vae = DeepVAE(num_rays, H, F, num_samples, epochs=10, batch_size=128)

    if trained:
        # load weights into new model
        vae.load_weights("vae_weights.h5")
    else:
        vae.fit(x_train, x_val,
                y_train, y_val,
                u_train, u_val)

    vae.plot_results(x_test, u_test, y_test)