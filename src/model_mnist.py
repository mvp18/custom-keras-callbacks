import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, layers, optimizers


def build_model(config):
    
    model = Sequential(

        [
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=tuple(config["image_size"])),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(config["num_classes"], activation="softmax")
        ]

    )
    
    opt = optimizers.SGD(lr=config["learning_rate"], momentum=0.9, nesterov=config["nesterov"])
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    print(model.summary())

    return model