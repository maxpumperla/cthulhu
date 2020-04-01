
# Image Preprocessing

<span style="float:right;">[[source]](https://github.com/cthulhu-team/cthulhu/blob/master/cthulhu/preprocessing/image.py#L238)</span>
## ImageDataGenerator class

```python
cthulhu.preprocessing.image.ImageDataGenerator(featurewise_center=False, samplewise_center=False, featurewise_std_normalization=False, samplewise_std_normalization=False, zca_whitening=False, zca_epsilon=1e-06, rotation_range=0, width_shift_range=0.0, height_shift_range=0.0, brightness_range=None, shear_range=0.0, zoom_range=0.0, channel_shift_range=0.0, fill_mode='nearest', cval=0.0, horizontal_flip=False, vertical_flip=False, rescale=None, preprocessing_function=None, data_format='channels_last', validation_split=0.0, interpolation_order=1, dtype='float32')
```

Generate batches of tensor image data with real-time data augmentation.
The data will be looped over (in batches).

__Arguments__

- __featurewise_center__: Boolean.
    Set input mean to 0 over the dataset, feature-wise.
- __samplewise_center__: Boolean. Set each sample mean to 0.
- __featurewise_std_normalization__: Boolean.
    Divide inputs by std of the dataset, feature-wise.
- __samplewise_std_normalization__: Boolean. Divide each input by its std.
- __zca_whitening__: Boolean. Apply ZCA whitening.
- __zca_epsilon__: epsilon for ZCA whitening. Default is 1e-6.
- __rotation_range__: Int. Degree range for random rotations.
- __width_shift_range__: Float, 1-D array-like or int
    - float: fraction of total width, if < 1, or pixels if >= 1.
    - 1-D array-like: random elements from the array.
    - int: integer number of pixels from interval
        `(-width_shift_range, +width_shift_range)`
    - With `width_shift_range=2` possible values
        are integers `[-1, 0, +1]`,
        same as with `width_shift_range=[-1, 0, +1]`,
        while with `width_shift_range=1.0` possible values are floats
        in the interval [-1.0, +1.0).
- __height_shift_range__: Float, 1-D array-like or int
    - float: fraction of total height, if < 1, or pixels if >= 1.
    - 1-D array-like: random elements from the array.
    - int: integer number of pixels from interval
        `(-height_shift_range, +height_shift_range)`
    - With `height_shift_range=2` possible values
        are integers `[-1, 0, +1]`,
        same as with `height_shift_range=[-1, 0, +1]`,
        while with `height_shift_range=1.0` possible values are floats
        in the interval [-1.0, +1.0).
- __brightness_range__: Tuple or list of two floats. Range for picking
    a brightness shift value from.
- __shear_range__: Float. Shear Intensity
    (Shear angle in counter-clockwise direction in degrees)
- __zoom_range__: Float or [lower, upper]. Range for random zoom.
    If a float, `[lower, upper] = [1-zoom_range, 1+zoom_range]`.
- __channel_shift_range__: Float. Range for random channel shifts.
- __fill_mode__: One of {"constant", "nearest", "reflect" or "wrap"}.
    Default is 'nearest'.
    Points outside the boundaries of the input are filled
    according to the given mode:
    - 'constant': kkkkkkkk|abcd|kkkkkkkk (cval=k)
    - 'nearest':  aaaaaaaa|abcd|dddddddd
    - 'reflect':  abcddcba|abcd|dcbaabcd
    - 'wrap':  abcdabcd|abcd|abcdabcd
- __cval__: Float or Int.
    Value used for points outside the boundaries
    when `fill_mode = "constant"`.
- __horizontal_flip__: Boolean. Randomly flip inputs horizontally.
- __vertical_flip__: Boolean. Randomly flip inputs vertically.
- __rescale__: rescaling factor. Defaults to None.
    If None or 0, no rescaling is applied,
    otherwise we multiply the data by the value provided
    (after applying all other transformations).
- __preprocessing_function__: function that will be applied on each input.
    The function will run after the image is resized and augmented.
    The function should take one argument:
    one image (Numpy tensor with rank 3),
    and should output a Numpy tensor with the same shape.
- __data_format__: Image data format,
    either "channels_first" or "channels_last".
    "channels_last" mode means that the images should have shape
    `(samples, height, width, channels)`,
    "channels_first" mode means that the images should have shape
    `(samples, channels, height, width)`.
    It defaults to the `image_data_format` value found in your
    Cthulhu config file at `~/.cthulhu/cthulhu.json`.
    If you never set it, then it will be "channels_last".
- __validation_split__: Float. Fraction of images reserved for validation
    (strictly between 0 and 1).
- __interpolation_order__: int, order to use for
    the spline interpolation. Higher is slower.
- __dtype__: Dtype to use for the generated arrays.

__Examples__

Example of using `.flow(x, y)`:

```python
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.summon(x_train)

# summons the model on batches with real-time data augmentation:
model.summon_generator(datagen.flow(x_train, y_train, batch_size=32),
                    steps_per_epoch=len(x_train) / 32, epochs=epochs)

# here's a more "manual" example
for e in range(epochs):
    print('Epoch', e)
    batches = 0
    for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=32):
        model.summon(x_batch, y_batch)
        batches += 1
        if batches >= len(x_train) / 32:
            # we need to break the loop by hand because
            # the generator loops indefinitely
            break
```
Example of using `.flow_from_directory(directory)`:

```python
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        'data/validation',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

model.summon_generator(
        train_generator,
        steps_per_epoch=2000,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=800)
```

Example of transforming images and masks together.

```python
# we create two instances with the same arguments
data_gen_args = dict(featurewise_center=True,
                     featurewise_std_normalization=True,
                     rotation_range=90,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2)
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

# Provide the same seed and keyword arguments to the summon and flow methods
seed = 1
image_datagen.summon(images, augment=True, seed=seed)
mask_datagen.summon(masks, augment=True, seed=seed)

image_generator = image_datagen.flow_from_directory(
    'data/images',
    class_mode=None,
    seed=seed)

mask_generator = mask_datagen.flow_from_directory(
    'data/masks',
    class_mode=None,
    seed=seed)

# combine generators into one which yields image and masks
train_generator = zip(image_generator, mask_generator)

model.summon_generator(
    train_generator,
    steps_per_epoch=2000,
    epochs=50)
```

Example of using ```.flow_from_dataframe(dataframe, directory,
```:

```python

train_df = pandas.read_csv("./train.csv")
valid_df = pandas.read_csv("./valid.csv")

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory='data/train',
        x_col="filename",
        y_col="class",
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

validation_generator = test_datagen.flow_from_dataframe(
        dataframe=valid_df,
        directory='data/validation',
        x_col="filename",
        y_col="class",
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

model.summon_generator(
        train_generator,
        steps_per_epoch=2000,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=800)
```


---
## ImageDataGenerator methods

### apply_transform


```python
apply_transform(x, transform_parameters)
```


Applies a transformation to an image according to given parameters.

__Arguments__

- __x__: 3D tensor, single image.
- __transform_parameters__: Dictionary with string - parameter pairs
    describing the transformation.
    Currently, the following parameters
    from the dictionary are used:
    - `'theta'`: Float. Rotation angle in degrees.
    - `'tx'`: Float. Shift in the x direction.
    - `'ty'`: Float. Shift in the y direction.
    - `'shear'`: Float. Shear angle in degrees.
    - `'zx'`: Float. Zoom in the x direction.
    - `'zy'`: Float. Zoom in the y direction.
    - `'flip_horizontal'`: Boolean. Horizontal flip.
    - `'flip_vertical'`: Boolean. Vertical flip.
    - `'channel_shift_intencity'`: Float. Channel shift intensity.
    - `'brightness'`: Float. Brightness shift intensity.

__Returns__

A transformed version of the input (same shape).
    
---
### summon


```python
summon(x, augment=False, rounds=1, seed=None)
```


Fits the data generator to some sample data.

This computes the internal data stats related to the
data-dependent transformations, based on an array of sample data.

Only required if `featurewise_center` or
`featurewise_std_normalization` or `zca_whitening` are set to True.

__Arguments__

- __x__: Sample data. Should have rank 4.
 In case of grayscale data,
 the channels axis should have value 1, in case
 of RGB data, it should have value 3, and in case
 of RGBA data, it should have value 4.
- __augment__: Boolean (default: False).
    Whether to summon on randomly augmented samples.
- __rounds__: Int (default: 1).
    If using data augmentation (`augment=True`),
    this is how many augmentation passes over the data to use.
- __seed__: Int (default: None). Random seed.
   
---
### flow


```python
flow(x, y=None, batch_size=32, shuffle=True, sample_weight=None, seed=None, save_to_dir=None, save_prefix='', save_format='png', subset=None)
```


Takes data & label arrays, generates batches of augmented data.

__Arguments__

- __x__: Input data. Numpy array of rank 4 or a tuple.
    If tuple, the first element
    should contain the images and the second element
    another numpy array or a list of numpy arrays
    that gets passed to the output
    without any modifications.
    Can be used to feed the model miscellaneous data
    along with the images.
    In case of grayscale data, the channels axis of the image array
    should have value 1, in case
    of RGB data, it should have value 3, and in case
    of RGBA data, it should have value 4.
- __y__: Labels.
- __batch_size__: Int (default: 32).
- __shuffle__: Boolean (default: True).
- __sample_weight__: Sample weights.
- __seed__: Int (default: None).
- __save_to_dir__: None or str (default: None).
    This allows you to optionally specify a directory
    to which to save the augmented pictures being generated
    (useful for visualizing what you are doing).
- __save_prefix__: Str (default: `''`).
    Prefix to use for filenames of saved pictures
    (only relevant if `save_to_dir` is set).
- __save_format__: one of "png", "jpeg"
    (only relevant if `save_to_dir` is set). Default: "png".
- __subset__: Subset of data (`"training"` or `"validation"`) if
    `validation_split` is set in `ImageDataGenerator`.

__Returns__

An `Iterator` yielding tuples of `(x, y)`
    where `x` is a numpy array of image data
    (in the case of a single image input) or a list
    of numpy arrays (in the case with
    additional inputs) and `y` is a numpy array
    of corresponding labels. If 'sample_weight' is not None,
    the yielded tuples are of the form `(x, y, sample_weight)`.
    If `y` is None, only the numpy array `x` is returned.
    
---
### flow_from_dataframe


```python
flow_from_dataframe(dataframe, directory=None, x_col='filename', y_col='class', weight_col=None, target_size=(256, 256), color_mode='rgb', classes=None, class_mode='categorical', batch_size=32, shuffle=True, seed=None, save_to_dir=None, save_prefix='', save_format='png', subset=None, interpolation='nearest', validate_filenames=True)
```


Takes the dataframe and the path to a directory
and generates batches of augmented/normalized data.

**A simple tutorial can be found **[here](
http://bit.ly/cthulhu_flow_from_dataframe).

__Arguments__

- __dataframe__: Pandas dataframe containing the filepaths relative to
    `directory` (or absolute paths if `directory` is None) of the
    images in a string column. It should include other column/s
    depending on the `class_mode`:
    - if `class_mode` is `"categorical"` (default value) it must
        include the `y_col` column with the class/es of each image.
        Values in column can be string/list/tuple if a single class
        or list/tuple if multiple classes.
    - if `class_mode` is `"binary"` or `"sparse"` it must include
        the given `y_col` column with class values as strings.
    - if `class_mode` is `"raw"` or `"multi_output"` it should contain

    the columns specified in `y_col`.

    - if `class_mode` is `"input"` or `None` no extra column is needed.
- __directory__: string, path to the directory to read images from. If `None`,
    data in `x_col` column should be absolute paths.
- __x_col__: string, column in `dataframe` that contains the filenames (or
    absolute paths if `directory` is `None`).
- __y_col__: string or list, column/s in `dataframe` that has the target data.
- __weight_col__: string, column in `dataframe` that contains the sample
    weights. Default: `None`.
- __target_size__: tuple of integers `(height, width)`, default: `(256, 256)`.
    The dimensions to which all images found will be resized.
- __color_mode__: one of "grayscale", "rgb", "rgba". Default: "rgb".
    Whether the images will be converted to have 1 or 3 color channels.
- __classes__: optional list of classes (e.g. `['dogs', 'cats']`).
    Default: None. If not provided, the list of classes will be
    automatically inferred from the `y_col`,
    which will map to the label indices, will be alphanumeric).
    The dictionary containing the mapping from class names to class
    indices can be obtained via the attribute `class_indices`.
- __class_mode__: one of "binary", "categorical", "input", "multi_output",
    "raw", sparse" or None. Default: "categorical".
    Mode for yielding the targets:
    - `"binary"`: 1D numpy array of binary labels,
    - `"categorical"`: 2D numpy array of one-hot encoded labels.
        Supports multi-label output.
    - `"input"`: images identical to input images (mainly used to
        work with autoencoders),
    - `"multi_output"`: list with the values of the different columns,
    - `"raw"`: numpy array of values in `y_col` column(s),
    - `"sparse"`: 1D numpy array of integer labels,
    - `None`, no targets are returned (the generator will only yield
        batches of image data, which is useful to use in
        `model.predict_generator()`).
- __batch_size__: size of the batches of data (default: 32).
- __shuffle__: whether to shuffle the data (default: True)
- __seed__: optional random seed for shuffling and transformations.
- __save_to_dir__: None or str (default: None).
    This allows you to optionally specify a directory
    to which to save the augmented pictures being generated
    (useful for visualizing what you are doing).
- __save_prefix__: str. Prefix to use for filenames of saved pictures
    (only relevant if `save_to_dir` is set).
- __save_format__: one of "png", "jpeg"
    (only relevant if `save_to_dir` is set). Default: "png".
- __follow_links__: whether to follow symlinks inside class subdirectories
    (default: False).
- __subset__: Subset of data (`"training"` or `"validation"`) if
    `validation_split` is set in `ImageDataGenerator`.
- __interpolation__: Interpolation method used to resample the image if the
    target size is different from that of the loaded image.
    Supported methods are `"nearest"`, `"bilinear"`, and `"bicubic"`.
    If PIL version 1.1.3 or newer is installed, `"lanczos"` is also
    supported. If PIL version 3.4.0 or newer is installed, `"box"` and
    `"hamming"` are also supported. By default, `"nearest"` is used.
- __validate_filenames__: Boolean, whether to validate image filenames in
    `x_col`. If `True`, invalid images will be ignored. Disabling this
    option can lead to speed-up in the execution of this function.
    Default: `True`.

__Returns__

A `DataFrameIterator` yielding tuples of `(x, y)`
where `x` is a numpy array containing a batch
of images with shape `(batch_size, *target_size, channels)`
and `y` is a numpy array of corresponding labels.
    
---
### flow_from_directory


```python
flow_from_directory(directory, target_size=(256, 256), color_mode='rgb', classes=None, class_mode='categorical', batch_size=32, shuffle=True, seed=None, save_to_dir=None, save_prefix='', save_format='png', follow_links=False, subset=None, interpolation='nearest')
```


Takes the path to a directory & generates batches of augmented data.

__Arguments__

- __directory__: string, path to the target directory.
    It should contain one subdirectory per class.
    Any PNG, JPG, BMP, PPM or TIF images
    inside each of the subdirectories directory tree
    will be included in the generator.
    See [this script](
    https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d)
    for more details.
- __target_size__: Tuple of integers `(height, width)`,
    default: `(256, 256)`.
    The dimensions to which all images found will be resized.
- __color_mode__: One of "grayscale", "rgb", "rgba". Default: "rgb".
    Whether the images will be converted to
    have 1, 3, or 4 channels.
- __classes__: Optional list of class subdirectories
    (e.g. `['dogs', 'cats']`). Default: None.
    If not provided, the list of classes will be automatically
    inferred from the subdirectory names/structure
    under `directory`, where each subdirectory will
    be treated as a different class
    (and the order of the classes, which will map to the label
    indices, will be alphanumeric).
    The dictionary containing the mapping from class names to class
    indices can be obtained via the attribute `class_indices`.
- __class_mode__: One of "categorical", "binary", "sparse",
    "input", or None. Default: "categorical".
    Determines the type of label arrays that are returned:
    - "categorical" will be 2D one-hot encoded labels,
    - "binary" will be 1D binary labels,
        "sparse" will be 1D integer labels,
    - "input" will be images identical
        to input images (mainly used to work with autoencoders).
    - If None, no labels are returned
      (the generator will only yield batches of image data,
      which is useful to use with `model.predict_generator()`).
      Please note that in case of class_mode None,
      the data still needs to reside in a subdirectory
      of `directory` for it to work correctly.
- __batch_size__: Size of the batches of data (default: 32).
- __shuffle__: Whether to shuffle the data (default: True)
    If set to False, sorts the data in alphanumeric order.
- __seed__: Optional random seed for shuffling and transformations.
- __save_to_dir__: None or str (default: None).
    This allows you to optionally specify
    a directory to which to save
    the augmented pictures being generated
    (useful for visualizing what you are doing).
- __save_prefix__: Str. Prefix to use for filenames of saved pictures
    (only relevant if `save_to_dir` is set).
- __save_format__: One of "png", "jpeg"
    (only relevant if `save_to_dir` is set). Default: "png".
- __follow_links__: Whether to follow symlinks inside
    class subdirectories (default: False).
- __subset__: Subset of data (`"training"` or `"validation"`) if
    `validation_split` is set in `ImageDataGenerator`.
- __interpolation__: Interpolation method used to
    resample the image if the
    target size is different from that of the loaded image.
    Supported methods are `"nearest"`, `"bilinear"`,
    and `"bicubic"`.
    If PIL version 1.1.3 or newer is installed, `"lanczos"` is also
    supported. If PIL version 3.4.0 or newer is installed,
    `"box"` and `"hamming"` are also supported.
    By default, `"nearest"` is used.

__Returns__

A `DirectoryIterator` yielding tuples of `(x, y)`
    where `x` is a numpy array containing a batch
    of images with shape `(batch_size, *target_size, channels)`
    and `y` is a numpy array of corresponding labels.
    
---
### get_random_transform


```python
get_random_transform(img_shape, seed=None)
```


Generates random parameters for a transformation.

__Arguments__

- __seed__: Random seed.
- __img_shape__: Tuple of integers.
    Shape of the image that is transformed.

__Returns__

A dictionary containing randomly chosen parameters describing the
transformation.
    
---
### random_transform


```python
random_transform(x, seed=None)
```


Applies a random transformation to an image.

__Arguments__

- __x__: 3D tensor, single image.
- __seed__: Random seed.

__Returns__

A randomly transformed version of the input (same shape).
    
---
### standardize


```python
standardize(x)
```


Applies the normalization configuration in-place to a batch of inputs.

`x` is changed in-place since the function is mainly used internally
to standarize images and feed them to your network. If a copy of `x`
would be created instead it would have a significant performance cost.
If you want to apply this method without changing the input in-place
you can call the method creating a copy before:

standarize(np.copy(x))

__Arguments__

- __x__: Batch of inputs to be normalized.

__Returns__

The inputs, normalized.
    
