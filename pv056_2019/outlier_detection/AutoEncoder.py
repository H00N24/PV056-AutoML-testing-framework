import keras
from keras import layers
from sklearn.metrics import mean_squared_error as mse
from sklearn import preprocessing
import pandas as pd


class AutoEncoder:
    def __init__(self, df, params):
        self.params = params

        self.params.setdefault("batch_size", 40)
        self.params.setdefault("epochs", 200)
        self.params.setdefault("optimizer", "keras.optimizers.adam()")
        self.params.setdefault("loss", "mean_squared_error")
        self.params.setdefault("layers", "[20,5,20]")
        self.params.setdefault("activation", "tanh")

        self.params["layers"] = eval(self.params["layers"])

        x = df.values
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        self.df = pd.DataFrame(x_scaled)

        self.auto_enc = keras.models.Sequential()

        self.auto_enc.add(
            layers.Dense(
                input_dim=df.shape[1],
                units=self.params["layers"][0],
                activation=self.params["activation"],
            )
        )

        if len(self.params["layers"]) > 1:
            for l in self.params["layers"][1:]:
                self.auto_enc.add(
                    layers.Dense(units=l, activation=self.params["activation"])
                )
        self.auto_enc.add(
            layers.Dense(units=df.shape[1], activation=self.params["activation"])
        )

        self.auto_enc.compile(
            loss=self.params["loss"], optimizer=eval(self.params["optimizer"])
        )

        self.auto_enc.summary()

    def compute_values(self, classes):
        self.auto_enc.fit(
            self.df.values,
            self.df.values,
            batch_size=int(self.params["batch_size"]),
            epochs=int(self.params["epochs"]),
        )
        return mse(
            self.df.transpose(),
            self.auto_enc.predict(self.df.values).transpose(),
            multioutput="raw_values",
        )
