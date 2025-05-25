import tensorflow as tf
from keras.api.models import Sequential
from keras.api.layers import (
    Conv1D,
    MaxPooling1D,
    Flatten,
    Dense,
    Dropout,
    LeakyReLU,
    ReLU,
    BatchNormalization,
)
from keras.api.regularizers import l2


class CNNModel:
    def build(num_classes) -> Sequential:
        model = Sequential(
            [
                Conv1D(
                    16,
                    kernel_size=3,
                    activation=None,
                    padding="same",
                    kernel_regularizer=l2(0.001),
                    input_shape=(16, 1),
                ),
                LeakyReLU(),
                MaxPooling1D(pool_size=2),
                Conv1D(
                    32,
                    kernel_size=3,
                    activation=None,
                    padding="same",
                    kernel_regularizer=l2(0.001),
                ),
                LeakyReLU(),
                MaxPooling1D(pool_size=2),
                Conv1D(
                    64,
                    kernel_size=3,
                    activation=None,
                    padding="same",
                    kernel_regularizer=l2(0.001),
                ),
                LeakyReLU(),
                MaxPooling1D(pool_size=2),
                Flatten(),
                Dense(64, kernel_regularizer=l2(0.001)),
                LeakyReLU(),
                Dropout(0.3),
                Dense(num_classes, activation="softmax"),
            ]
        )

        model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )

        return model


class CNNModelV2:
    @staticmethod
    def build(num_classes) -> Sequential:
        model = Sequential(
            [
                # First Conv1D block
                Conv1D(
                    12,
                    kernel_size=3,
                    padding="same",
                    kernel_regularizer=l2(0.001),
                    input_shape=(16, 1),
                ),
                ReLU(),
                BatchNormalization(),
                MaxPooling1D(pool_size=2),
                # Second Conv1D block
                Conv1D(
                    16,
                    kernel_size=3,
                    padding="same",
                    kernel_regularizer=l2(0.001),
                ),
                ReLU(),
                BatchNormalization(),
                MaxPooling1D(pool_size=2),
                # Third Conv1D block
                Conv1D(
                    20,
                    kernel_size=3,
                    padding="same",
                    kernel_regularizer=l2(0.001),
                ),
                ReLU(),
                BatchNormalization(),
                MaxPooling1D(pool_size=2),
                # Flatten and Dense layers
                Flatten(),
                Dense(20, kernel_regularizer=l2(0.001)),
                ReLU(),
                BatchNormalization(),
                Dropout(0.5),
                Dense(num_classes, activation="softmax"),
            ]
        )

        model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )

        return model
