import tensorflow as tf
from tensorflow.keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, LeakyReLU, Dropout
from tensorflow.keras.regularizers import l2


class CNNModel:
    def build(num_classes) -> Sequential:
        model = Sequential(
            [
                Conv1D(
                    32,
                    kernel_size=3,
                    activation=None,
                    padding="same",
                    kernel_regularizer=l2(0.001),
                    input_shape=(16, 1),
                ),
                LeakyReLU(alpha=0.1),
                MaxPooling1D(pool_size=2),
                Conv1D(
                    64,
                    kernel_size=3,
                    activation=None,
                    padding="same",
                    kernel_regularizer=l2(0.001),
                ),
                LeakyReLU(alpha=0.1),
                MaxPooling1D(pool_size=2),
                Conv1D(
                    128,
                    kernel_size=3,
                    activation=None,
                    padding="same",
                    kernel_regularizer=l2(0.001),
                ),
                LeakyReLU(alpha=0.1),
                MaxPooling1D(pool_size=2),
                Flatten(),
                Dense(64, kernel_regularizer=l2(0.001)),
                LeakyReLU(alpha=0.1),
                Dropout(0.3),
                Dense(num_classes, activation="softmax"),
            ]
        )

        model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )

        return model


# ----


class CNNModelV2:
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
                LeakyReLU(alpha=0.1),
                MaxPooling1D(pool_size=2),
                Conv1D(
                    32,
                    kernel_size=3,
                    activation=None,
                    padding="same",
                    kernel_regularizer=l2(0.001),
                ),
                LeakyReLU(alpha=0.1),
                MaxPooling1D(pool_size=2),
                Conv1D(
                    64,
                    kernel_size=3,
                    activation=None,
                    padding="same",
                    kernel_regularizer=l2(0.001),
                ),
                LeakyReLU(alpha=0.1),
                MaxPooling1D(pool_size=2),
                Flatten(),
                Dense(64, kernel_regularizer=l2(0.001)),
                LeakyReLU(alpha=0.1),
                Dropout(0.3),
                Dense(num_classes, activation="softmax"),
            ]
        )

        model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )

        return model
