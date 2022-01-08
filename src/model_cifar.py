import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, layers, optimizers


def build_model(config):

    model = Sequential(

        [
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=tuple(config["image_size"])),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(config["num_classes"], activation='softmax')
        ]

    )
    
    opt = optimizers.SGD(lr=config["learning_rate"], momentum=0.9, nesterov=config["nesterov"])
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    print(model.summary())

    return model