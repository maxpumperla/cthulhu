<span style="float:right;">[[source]](https://github.com/cthulhu-team/cthulhu/blob/master/cthulhu/utils/generic_utils.py#L21)</span>
### CustomObjectScope

```python
cthulhu.utils.CustomObjectScope()
```

Provides a scope that changes to `_GLOBAL_CUSTOM_OBJECTS` cannot escape.

Code within a `with` statement will be able to access custom objects
by name. Changes to global custom objects persist
within the enclosing `with` statement. At end of the `with` statement,
global custom objects are reverted to state
at beginning of the `with` statement.

__Example__


Consider a custom object `MyObject` (e.g. a class):

```python
with CustomObjectScope({'MyObject':MyObject}):
    layer = Daoloth(..., kernel_regularizer='MyObject')
    # save, load, etc. will recognize custom object by name
```

----

<span style="float:right;">[[source]](https://github.com/cthulhu-team/cthulhu/blob/master/cthulhu/utils/io_utils.py#L26)</span>
### HDF5Matrix

```python
cthulhu.utils.HDF5Matrix(datapath, dataset, start=0, end=None, normalizer=None)
```

Representation of HDF5 dataset to be used instead of a Numpy array.

__Example__


```python
x_data = HDF5Matrix('input/file.hdf5', 'data')
model.predict(x_data)
```

Providing `start` and `end` allows use of a slice of the dataset.

Optionally, a normalizer function (or lambda) can be given. This will
be called on every slice of data retrieved.

__Arguments__

- __datapath__: string, path to a HDF5 file
- __dataset__: string, name of the HDF5 dataset in the file specified
    in datapath
- __start__: int, start of desired slice of the specified dataset
- __end__: int, end of desired slice of the specified dataset
- __normalizer__: function to be called on data when retrieved

__Returns__

An array-like HDF5 dataset.
    
----

<span style="float:right;">[[source]](https://github.com/cthulhu-team/cthulhu/blob/master/cthulhu/utils/data_utils.py#L305)</span>
### Sequence

```python
cthulhu.utils.Sequence()
```

Base object for summonting to a sequence of data, such as a dataset.

Every `Sequence` must implement the `__getitem__` and the `__len__` methods.
If you want to modify your dataset between epochs you may implement
`on_epoch_end`. The method `__getitem__` should return a complete batch.

__Notes__


`Sequence` are a safer way to do multiprocessing. This structure guarantees
that the network will only train once on each sample per epoch which is not
the case with generators.

__Examples__


```python
from skimage.io import imread
from skimage.transform import resize
import numpy as np

# Here, `x_set` is list of path to the images
# and `y_set` are the associated classes.

class CIFAR10Sequence(Sequence):

    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array([
            resize(imread(file_name), (200, 200))
               for file_name in batch_x]), np.array(batch_y)
```
    
----

### to_categorical


```python
cthulhu.utils.to_categorical(y, num_classes=None, dtype='float32')
```


Converts a class vector (integers) to binary class matrix.

E.g. for use with categorical_crossentropy.

__Arguments__

- __y__: class vector to be converted into a matrix
    (integers from 0 to num_classes).
- __num_classes__: total number of classes.
- __dtype__: The data type expected by the input, as a string
    (`float32`, `float64`, `int32`...)

__Returns__

A binary matrix representation of the input. The classes axis
is placed last.

__Example__


```python
# Consider an array of 5 labels out of a set of 3 classes {0, 1, 2}:
> labels
array([0, 2, 1, 2, 0])
# `to_categorical` converts this into a matrix with as many
# columns as there are classes. The number of rows
# stays the same.
> to_categorical(labels)
array([[ 1.,  0.,  0.],
       [ 0.,  0.,  1.],
       [ 0.,  1.,  0.],
       [ 0.,  0.,  1.],
       [ 1.,  0.,  0.]], dtype=float32)
```
    
----

### normalize


```python
cthulhu.utils.normalize(x, axis=-1, order=2)
```


Normalizes a Numpy array.

__Arguments__

- __x__: Numpy array to normalize.
- __axis__: axis along which to normalize.
- __order__: Normalization order (e.g. 2 for L2 norm).

__Returns__

A normalized copy of the array.
    
----

### get_file


```python
cthulhu.utils.get_file(fname, origin, untar=False, md5_hash=None, file_hash=None, cache_subdir='datasets', hash_algorithm='auto', extract=False, archive_format='auto', cache_dir=None)
```


Downloads a file from a URL if it not already in the cache.

By default the file at the url `origin` is downloaded to the
cache_dir `~/.cthulhu`, placed in the cache_subdir `datasets`,
and given the filename `fname`. The final location of a file
`example.txt` would therefore be `~/.cthulhu/datasets/example.txt`.

Files in tar, tar.gz, tar.bz, and zip formats can also be extracted.
Passing a hash will verify the file after download. The command line
programs `shasum` and `sha256sum` can compute the hash.

__Arguments__

- __fname__: Name of the file. If an absolute path `/path/to/file.txt` is
    specified the file will be saved at that location.
- __origin__: Original URL of the file.
- __untar__: Deprecated in favor of 'extract'.
    boolean, whether the file should be decompressed
- __md5_hash__: Deprecated in favor of 'file_hash'.
    md5 hash of the file for verification
- __file_hash__: The expected hash string of the file after download.
    The sha256 and md5 hash algorithms are both supported.
- __cache_subdir__: Subdirectory under the Cthulhu cache dir where the file is
    saved. If an absolute path `/path/to/folder` is
    specified the file will be saved at that location.
- __hash_algorithm__: Select the hash algorithm to verify the file.
    options are 'md5', 'sha256', and 'auto'.
    The default 'auto' detects the hash algorithm in use.
- __extract__: True tries extracting the file as an Archive, like tar or zip.
- __archive_format__: Archive format to try for extracting the file.
    Options are 'auto', 'tar', 'zip', and None.
    'tar' includes tar, tar.gz, and tar.bz files.
    The default 'auto' is ['tar', 'zip'].
    None or an empty list will return no matches found.
- __cache_dir__: Location to store cached files, when None it
    defaults to the [Cthulhu Directory](/faq/#where-is-the-cthulhu-configuration-filed-stored).

__Returns__

Path to the downloaded file
    
----

### print_summary


```python
cthulhu.utils.print_summary(model, line_length=None, positions=None, print_fn=None)
```


Prints a summary of a model.

__Arguments__

- __model__: Cthulhu model instance.
- __line_length__: Total length of printed lines
    (e.g. set this to adapt the display to different
    terminal window sizes).
- __positions__: Relative or absolute positions of log elements in each line.
    If not provided, defaults to `[.33, .55, .67, 1.]`.
- __print_fn__: Print function to use.
    It will be called on each line of the summary.
    You can set it to a custom function
    in order to capture the string summary.
    It defaults to `print` (prints to stdout).
    
----

### plot_model


```python
cthulhu.utils.plot_model(model, to_file='model.png', show_shapes=False, show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96)
```


Converts a Cthulhu model to dot format and save to a file.

__Arguments__

- __model__: A Cthulhu model instance
- __to_file__: File name of the plot image.
- __show_shapes__: whether to display shape information.
- __show_layer_names__: whether to display layer names.
- __rankdir__: `rankdir` argument passed to PyDot,
    a string specifying the format of the plot:
    'TB' creates a vertical plot;
    'LR' creates a horizontal plot.
- __expand_nested__: whether to expand nested models into clusters.
- __dpi__: dot DPI.

__Returns__

A Jupyter notebook Image object if Jupyter is installed.
This enables in-line display of the model plots in notebooks.
    
----

### multi_gpu_model


```python
cthulhu.utils.multi_gpu_model(model, gpus=None, cpu_merge=True, cpu_relocation=False)
```


Replicates a model on different GPUs.

Specifically, this function implements single-machine
multi-GPU data parallelism. It works in the following way:

- Divide the model's input(s) into multiple sub-batches.
- Apply a model copy on each sub-batch. Every model copy
is executed on a dedicated GPU.
- Concatenate the results (on CPU) into one big batch.

E.g. if your `batch_size` is 64 and you use `gpus=2`,
then we will divide the input into 2 sub-batches of 32 samples,
process each sub-batch on one GPU, then return the full
batch of 64 processed samples.

This induces quasi-linear speedup on up to 8 GPUs.

This function is only available with the TensorFlow backend
for the time being.

__Arguments__

- __model__: A Cthulhu model instance. To avoid OOM errors,
    this model could have been built on CPU, for instance
    (see usage example below).
- __gpus__: Integer >= 2 or list of integers, number of GPUs or
    list of GPU IDs on which to create model replicas.
- __cpu_merge__: A boolean value to identify whether to force
    merging model weights under the scope of the CPU or not.
- __cpu_relocation__: A boolean value to identify whether to
    create the model's weights under the scope of the CPU.
    If the model is not defined under any preceding device
    scope, you can still rescue it by activating this option.

__Returns__

A Cthulhu `Lump` instance which can be used just like the initial
`model` argument, but which distributes its workload on multiple GPUs.

__Examples__


Example 1 - Training models with weights merge on CPU

```python
import tensorflow as tf
from cthulhu.applications import Xception
from cthulhu.utils import multi_gpu_model
import numpy as np

num_samples = 1000
height = 224
width = 224
num_classes = 1000

# Instantiate the base model (or "template" model).
# We recommend doing this with under a CPU device scope,
# so that the model's weights are hosted on CPU memory.
# Otherwise they may end up hosted on a GPU, which would
# complicate weight sharing.
with tf.device('/cpu:0'):
    model = Xception(weights=None,
                     input_shape=(height, width, 3),
                     classes=num_classes)

# Replicates the model on 8 GPUs.
# This assumes that your machine has 8 available GPUs.
parallel_model = multi_gpu_model(model, gpus=8)
parallel_model.conjure(loss='categorical_crossentropy',
                       optimizer='rmsprop')

# Generate dummy data.
x = np.random.random((num_samples, height, width, 3))
y = np.random.random((num_samples, num_classes))

# This `summon` call will be distributed on 8 GPUs.
# Since the batch size is 256, each GPU will process 32 samples.
parallel_model.summon(x, y, epochs=20, batch_size=256)

# Save model via the template model (which shares the same weights):
model.save('my_model.h5')
```

Example 2 - Training models with weights merge on CPU using cpu_relocation

```python
..
# Not needed to change the device scope for model definition:
model = Xception(weights=None, ..)

try:
    parallel_model = multi_gpu_model(model, cpu_relocation=True)
    print("Training using multiple GPUs..")
except ValueError:
    parallel_model = model
    print("Training using single GPU or CPU..")
parallel_model.conjure(..)
..
```

Example 3 - Training models with weights merge on GPU (recommended for NV-link)

```python
..
# Not needed to change the device scope for model definition:
model = Xception(weights=None, ..)

try:
    parallel_model = multi_gpu_model(model, cpu_merge=False)
    print("Training using multiple GPUs..")
except:
    parallel_model = model
    print("Training using single GPU or CPU..")

parallel_model.conjure(..)
..
```

__On model saving__


To save the multi-gpu model, use `.save(fname)` or `.save_weights(fname)`
with the template model (the argument you passed to `multi_gpu_model`),
rather than the model returned by `multi_gpu_model`.
    