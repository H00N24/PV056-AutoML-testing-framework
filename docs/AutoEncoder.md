## AutoEncoder in Keras
****
Trains defined Neural Network on the whole dataset and evaulates output for each instance. The output is mean squared error [MSE](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html) between the instance and its prediction.   

#### Configuration:
```
"detectors": {
    "AutoEncoder" : {
      "batch_size": BATCH_SIZE,
      "epochs": EPOCHS,
      "optimizer": OPTIMIZER,
      "layers": NN_LAYOUT,
      "activation": ACTIVATION,
      "loss": LOSS
    }
  },
```

, where 
* 1 <= BATCH_SIZE(int) >= SIZE_OF_DATASET, default = 40
* 1 <= EPOCHS(int), default = 200
* OPTIMIZER(str) from [keras.optimizers](https://keras.io/optimizers/), default: "keras.optimizers.adam()"
* LAYERS(str) -- list of number of neurons in the given layers, default = "[20,5,20]" for 3 layers
* ACTIVATION(str) -- name of activation function [activations](https://keras.io/activations/), default = "tanh"
* LOSS -- name of a loss function for training neural net from [losses](https://keras.io/losses/), default = "mean_squared_error"