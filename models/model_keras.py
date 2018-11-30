import logging
import os

import keras
from keras.models import load_model

logger = logging.getLogger(__name__)


class ModelKeras:

    def __init__(self, n_actions, x_shape, y_shape, batch_size,
                 model_path=None):
        self.n_actions = n_actions
        self.x_shape = x_shape
        self.y_shape = y_shape
        self.batch_size = batch_size

        if model_path is None or not os.path.isfile(model_path):
            logger.info('creating new model')
            self.model_impl = self._create_model()
        else:
            logger.info('restoring model from {}'.format(model_path))
            self.model_impl = load_model(model_path)

    def _create_model(self):
        frames_input = keras.layers.Input(
            [self.n_actions, self.y_shape, self.x_shape], name='frames')
        actions_input = keras.layers.Input((self.n_actions,),
                                           name='action_one-hot')
        normalized = keras.layers.Lambda(
            lambda x: (x / 255.0), name='normalized')(frames_input)

        conv_1 = keras.layers.Conv2D(
            filters=16, kernel_size=(8, 8), strides=(2, 2), activation='relu',
            padding='valid', data_format='channels_first',
            name='conv1')(normalized)
        conv_2 = keras.layers.Conv2D(
            filters=32, kernel_size=(4, 4), strides=(2, 2), activation='relu',
            padding='valid', data_format='channels_first',
            name='conv2')(conv_1)
        conv_flattened = keras.layers.core.Flatten(name='flat')(conv_2)
        hidden = keras.layers.Dense(256, activation='relu',
                                    name='hidden')(conv_flattened)
        output = keras.layers.Dense(self.n_actions, activation='sigmoid',
                                    name='output')(hidden)
        # output has 4 rewards (one per action) - next we multiply by the mask
        filtered_output = keras.layers.multiply([output, actions_input])
        model = keras.models.Model(
            inputs=[frames_input, actions_input], outputs=filtered_output)
        model.compile(optimizer='rmsprop', loss='mse')
        return model

    def predict(self, frames, actions):
        return self.model_impl.predict([frames, actions])

    def fit(self, frames, actions, target):
        self.model_impl.fit(
            [frames, actions],
            target,
            epochs=1, batch_size=self.batch_size, verbose=0
        )

    def save(self, path):
        self.model_impl.save(path)
