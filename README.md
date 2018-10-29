## Turntable Controller

![](img/demo.gif)

## Dependencies

It is recommended to work on a `virtualenv`. Install [`virtualenvwrapper`](http://virtualenvwrapper.readthedocs.io/en/latest/install.html) and activate a virtual environment. Install CUDA 9.0 [[Tutorial](https://yangcha.github.io/CUDA90/)]. Install other dependencies by running:
```bash 
(env) $ pip install numpy matplotlib pybullet tensorflow-gpu keras cvxpy==0.4.11
```

## Collect Dataset

Example datasets are provided in `data/`. Data can also be collected by running: 
```bash 
(env) $ python collect_data.py [path/to/data.txt]
```

## Train Model

The VAE can be trained by running:
```bash 
(env) $ python train.py [path/to/data_training.txt] [path/to/data_testing.txt]
```

Learned weights are provided in `weights/`. Set `TRAINED = True` in `params.py` to use them. Visualize the predictions by running:
```bash 
(env) $ python test.py [path/to/data_testing.txt]
```

## Run MPC controller

Once the model is trained, run the MPC controller to point the turntable to an obstacle-free direction:
```bash 
(env) $ python mpc.py
```

Each time a random world is generated.