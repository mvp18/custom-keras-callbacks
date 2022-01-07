import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, layers, optimizers


def build_model(config):
    
    # model = Sequential(

    # 	[
    # 		layers.Conv2D(6, (5, 5), padding='valid', activation = 'relu', kernel_initializer='he_normal', input_shape=tuple(config["image_size"])),
    # 		layers.MaxPooling2D((2, 2), strides=(2, 2)),
    # 		layers.Conv2D(16, (5, 5), padding='valid', activation = 'relu', kernel_initializer='he_normal'),
    # 		layers.MaxPooling2D((2, 2), strides=(2, 2)),
    # 		layers.Flatten(),
    # 		layers.Dense(120, activation = 'relu', kernel_initializer='he_normal'),
    # 		layers.Dense(84, activation = 'relu', kernel_initializer='he_normal'),
    # 		layers.Dense(config["num_classes"], activation = 'softmax', kernel_initializer='he_normal')
    # 	]

    # )

    model = Sequential(

        [
            layers.Conv2D(64, (3, 3), activation='relu', input_shape=tuple(config["image_size"])),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dense(config["num_classes"], activation='softmax')
        ]

    )
    
    opt = optimizers.SGD(lr=config["learning_rate"], momentum=0.9, nesterov=config["nesterov"])
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    print(model.summary())

    return model