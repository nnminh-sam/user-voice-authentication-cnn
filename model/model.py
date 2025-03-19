import tensorflow as tf
from tensorflow.keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, LeakyReLU, Dropout
from tensorflow.keras.regularizers import l2


# * Version 1
# def build_voice_auth_cnn(num_classes):
#     model = Sequential()

#     # Input shape: (time_steps, features) -> (None, 16, 1)
#     input_shape = (16, 1)  # 16 extracted features, 1D signal

#     # First Conv Layer
#     model.add(Conv1D(filters=32, kernel_size=3, padding='same', input_shape=input_shape))
#     model.add(LeakyReLU(alpha=0.3))
#     model.add(MaxPooling1D(pool_size=2))

#     # Second Conv Layer
#     model.add(Conv1D(filters=64, kernel_size=3, padding='same'))
#     model.add(LeakyReLU(alpha=0.3))
#     model.add(MaxPooling1D(pool_size=2))

#     # Third Conv Layer
#     model.add(Conv1D(filters=128, kernel_size=3, padding='same'))
#     model.add(LeakyReLU(alpha=0.3))
#     model.add(MaxPooling1D(pool_size=2))

#     # Flatten Layer
#     model.add(Flatten())

#     # Fully Connected Dense Layer
#     model.add(Dense(128))
#     model.add(LeakyReLU(alpha=0.3))
#     model.add(Dropout(0.3))  # Prevent overfitting

#     # Output Layer with Softmax
#     model.add(Dense(num_classes, activation='softmax'))

#     # Compile the model
#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#         loss='categorical_crossentropy',
#         metrics=['accuracy']
#     )

#     return model


# ----

# * Version 2
def build_voice_auth_cnn(num_classes) -> Sequential:
    model = Sequential([
        Conv1D(32, kernel_size=3, activation=None, padding='same', kernel_regularizer=l2(0.001), input_shape=(16, 1)),
        LeakyReLU(alpha=0.1),
        MaxPooling1D(pool_size=2),

        Conv1D(64, kernel_size=3, activation=None, padding='same', kernel_regularizer=l2(0.001)),
        LeakyReLU(alpha=0.1),
        MaxPooling1D(pool_size=2),

        Conv1D(128, kernel_size=3, activation=None, padding='same', kernel_regularizer=l2(0.001)),
        LeakyReLU(alpha=0.1),
        MaxPooling1D(pool_size=2),

        Flatten(),
        Dense(64, kernel_regularizer=l2(0.001)), 
        LeakyReLU(alpha=0.1),
        Dropout(0.3),

        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

