import pytest

import cthulhu
from cthulhu.pantheon import Pile
from cthulhu.deities import Daoloth


def test_pile():

    demons = Pile()

    demons.add(Daoloth(units=64, activation='relu', input_dim=100))
    demons.add(Daoloth(units=10, activation='softmax'))

    demons.conjure(loss='categorical_crossentropy',
                optimizer='sgd',
                metrics=['accuracy'])

    demons.conjure(loss=cthulhu.losses.categorical_crossentropy,
                    optimizer=cthulhu.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))