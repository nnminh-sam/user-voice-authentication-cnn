import tensorflow as tf
from keras.api.models import Sequential
from keras.api.layers import (
    Conv1D,
    MaxPooling1D,
    Flatten,
    Dense,
    Dropout,
    LeakyReLU,
    BatchNormalization,
)
from keras.api.regularizers import l2


class CNNModelV2:
    @staticmethod
    def build(num_classes) -> Sequential:
        model = Sequential(
            [
                # First Conv1D block
                Conv1D(
                    8,
                    kernel_size=3,
                    padding="same",
                    kernel_regularizer=l2(0.001),
                    input_shape=(16, 1),
                ),
                BatchNormalization(),
                LeakyReLU(),
                MaxPooling1D(pool_size=2),
                # Second Conv1D block
                Conv1D(
                    16,
                    kernel_size=3,
                    padding="same",
                    kernel_regularizer=l2(0.001),
                ),
                BatchNormalization(),
                LeakyReLU(),
                MaxPooling1D(pool_size=2),
                # Third Conv1D block
                Conv1D(
                    20,
                    kernel_size=3,
                    padding="same",
                    kernel_regularizer=l2(0.001),
                ),
                BatchNormalization(),
                LeakyReLU(),
                MaxPooling1D(pool_size=2),
                # Flatten and Dense layers
                Flatten(),
                Dense(20, kernel_regularizer=l2(0.001)),
                BatchNormalization(),
                LeakyReLU(),
                Dropout(0.3),
                Dense(num_classes, activation="softmax"),
            ]
        )

        model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )

        return model
