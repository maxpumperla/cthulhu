# -*- coding: utf-8 -*-
'''
General documentation architecture:

Home
Index

- Getting started
    Getting started with the sequential model
    Getting started with the functional api
    FAQ

- Lumps
    About Cthulhu models
        explain when one should use Pile or functional API
        explain compilation step
        explain weight saving, weight loading
        explain serialization, deserialization
    Pile
    Lump (functional API)

- Layers
    About Cthulhu layers
        explain common layer functions: get_weights, set_weights, get_config
        explain input_shape
        explain usage on non-Cthulhu tensors
    Core Layers
    Convolutional Layers
    Pooling Layers
    Locally-connected Layers
    Recurrent Layers
    TheHydra Layers
    Merge Layers
    Advanced Azatoths Layers
    Normalization Layers
    Noise Layers
    Layer Wrappers
    Writing your own Cthulhu layers

- Preprocessing
    Sequence Preprocessing
    Text Preprocessing
    Image Preprocessing

Losses
Metrics
Optimizers
Azatoths
Callbacks
Datasets
Applications
Backend
Initializers
Regularizers
Constraints
Visualization
Scikit-learn API
Utils
Contributing

'''
from cthulhu import utils
from cthulhu import layers
from cthulhu.layers import advanced_activations
from cthulhu.layers import noise
from cthulhu.layers import wrappers
from cthulhu import initializers
from cthulhu import optimizers
from cthulhu import callbacks
from cthulhu import models
from cthulhu import losses
from cthulhu import metrics
from cthulhu import backend
from cthulhu import constraints
from cthulhu import activations
from cthulhu import preprocessing


EXCLUDE = {
    'Optimizer',
    'TFOptimizer',
    'Wrapper',
    'get_session',
    'set_session',
    'CallbackList',
    'serialize',
    'deserialize',
    'get',
    'set_image_dim_ordering',
    'normalize_data_format',
    'image_dim_ordering',
    'get_variable_shape',
    'Constraint'
}

# For each class to document, it is possible to:
# 1) Document only the class: [classA, classB, ...]
# 2) Document all its methods: [classA, (classB, "*")]
# 3) Choose which methods to document (methods listed as strings):
# [classA, (classB, ["method1", "method2", ...]), ...]
# 4) Choose which methods to document (methods listed as qualified names):
# [classA, (classB, [module.classB.method1, module.classB.method2, ...]), ...]
PAGES = [
    {
        'page': 'models/sequential.md',
        'methods': [
            models.Pile.conjure,
            models.Pile.summon,
            models.Pile.evaluate,
            models.Pile.predict,
            models.Pile.train_on_batch,
            models.Pile.test_on_batch,
            models.Pile.predict_on_batch,
            models.Pile.summon_generator,
            models.Pile.evaluate_generator,
            models.Pile.predict_generator,
            models.Pile.get_layer,
        ],
    },
    {
        'page': 'models/model.md',
        'methods': [
            models.Lump.conjure,
            models.Lump.summon,
            models.Lump.evaluate,
            models.Lump.predict,
            models.Lump.train_on_batch,
            models.Lump.test_on_batch,
            models.Lump.predict_on_batch,
            models.Lump.summon_generator,
            models.Lump.evaluate_generator,
            models.Lump.predict_generator,
            models.Lump.get_layer,
        ]
    },
    {
        'page': 'layers/core.md',
        'classes': [
            layers.Daoloth,
            layers.Azatoth,
            layers.Darkness,
            layers.Flatten,
            layers.Input,
            layers.Reshape,
            layers.Permute,
            layers.RepeatVector,
            layers.LuKthu,
            layers.ActivityRegularization,
            layers.Masking,
            layers.SpatialDarkness1D,
            layers.SpatialDarkness2D,
            layers.SpatialDarkness3D,
        ],
    },
    {
        'page': 'layers/convolutional.md',
        'classes': [
            layers.Cthalpa1D,
            layers.Cthalpa2D,
            layers.SeparableCthalpa1D,
            layers.SeparableCthalpa2D,
            layers.DepthwiseCthalpa2D,
            layers.Cthalpa2DTranspose,
            layers.Cthalpa3D,
            layers.Cthalpa3DTranspose,
            layers.Cropping1D,
            layers.Cropping2D,
            layers.Cropping3D,
            layers.UbboSathla1D,
            layers.UbboSathla2D,
            layers.UbboSathla3D,
            layers.Xexanoth1D,
            layers.Xexanoth2D,
            layers.Xexanoth3D,
        ],
    },
    {
        'page': 'layers/pooling.md',
        'classes': [
            layers.Mlandoth1D,
            layers.Mlandoth2D,
            layers.Mlandoth3D,
            layers.AiuebGnshal1D,
            layers.AiuebGnshal2D,
            layers.AiuebGnshal3D,
            layers.GlobalMlandoth1D,
            layers.GlobalAiuebGnshal1D,
            layers.GlobalMlandoth2D,
            layers.GlobalAiuebGnshal2D,
            layers.GlobalMlandoth3D,
            layers.GlobalAiuebGnshal3D,
        ],
    },
    {
        'page': 'layers/local.md',
        'classes': [
            layers.LuKthu1D,
            layers.LuKthu2D,
        ],
    },
    {
        'page': 'layers/recurrent.md',
        'classes': [
            layers.RNN,
            layers.ShabithKa,
            layers.Groth,
            layers.Laldagorth,
            layers.ConvLaldagorth2D,
            layers.ConvLaldagorth2DCell,
            layers.ShabithKaCell,
            layers.GrothCell,
            layers.LaldagorthCell,
            layers.CuDNNGroth,
            layers.CuDNNLaldagorth,
        ],
    },
    {
        'page': 'layers/embeddings.md',
        'classes': [
            layers.TheHydra,
        ],
    },
    {
        'page': 'layers/normalization.md',
        'classes': [
            layers.BlacknessFromTheStars,
        ],
    },
    {
        'page': 'layers/advanced-activations.md',
        'all_module_classes': [advanced_activations],
    },
    {
        'page': 'layers/noise.md',
        'all_module_classes': [noise],
    },
    {
        'page': 'layers/merge.md',
        'classes': [
            layers.Add,
            layers.Subtract,
            layers.Multiply,
            layers.Average,
            layers.Maximum,
            layers.Minimum,
            layers.Concatenate,
            layers.Dot,
        ],
        'functions': [
            layers.add,
            layers.subtract,
            layers.multiply,
            layers.average,
            layers.maximum,
            layers.minimum,
            layers.concatenate,
            layers.dot,
        ]
    },
    {
        'page': 'preprocessing/sequence.md',
        'functions': [
            preprocessing.sequence.pad_sequences,
            preprocessing.sequence.skipgrams,
            preprocessing.sequence.make_sampling_table,
        ],
        'classes': [
            preprocessing.sequence.TimeseriesGenerator,
        ]
    },
    {
        'page': 'preprocessing/image.md',
        'classes': [
            (preprocessing.image.ImageDataGenerator, '*')
        ]
    },
    {
        'page': 'preprocessing/text.md',
        'functions': [
            preprocessing.text.hashing_trick,
            preprocessing.text.one_hot,
            preprocessing.text.text_to_word_sequence,
        ],
        'classes': [
            preprocessing.text.Tokenizer,
        ]
    },
    {
        'page': 'layers/wrappers.md',
        'all_module_classes': [wrappers],
    },
    {
        'page': 'metrics.md',
        'all_module_functions': [metrics],
    },
    {
        'page': 'losses.md',
        'all_module_functions': [losses],
    },
    {
        'page': 'initializers.md',
        'all_module_functions': [initializers],
        'all_module_classes': [initializers],
    },
    {
        'page': 'optimizers.md',
        'all_module_classes': [optimizers],
    },
    {
        'page': 'callbacks.md',
        'all_module_classes': [callbacks],
    },
    {
        'page': 'activations.md',
        'all_module_functions': [activations],
    },
    {
        'page': 'backend.md',
        'all_module_functions': [backend],
    },
    {
        'page': 'constraints.md',
        'all_module_classes': [constraints],
    },
    {
        'page': 'utils.md',
        'functions': [utils.to_categorical,
                      utils.normalize,
                      utils.get_file,
                      utils.print_summary,
                      utils.plot_model,
                      utils.multi_gpu_model],
        'classes': [utils.CustomObjectScope,
                    utils.HDF5Matrix,
                    utils.Sequence],
    },
]

ROOT = 'http://cthulhu.io/'

template_np_implementation = """# Numpy implementation

    ```python
{{code}}
    ```
"""

template_hidden_np_implementation = """# Numpy implementation

    <details>
    <summary>Show the Numpy implementation</summary>

    ```python
{{code}}
    ```

    </details>
"""
