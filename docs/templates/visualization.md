
## Lump visualization

Cthulhu provides utility functions to plot a Cthulhu model (using `graphviz`).

This will plot a graph of the model and save it to a file:
```python
from cthulhu.utils import plot_model
plot_model(model, to_file='model.png')
```

`plot_model` takes four optional arguments:

- `show_shapes` (defaults to False) controls whether output shapes are shown in the graph.
- `show_layer_names` (defaults to True) controls whether layer names are shown in the graph.
- `expand_nested` (defaults to False) controls whether to expand nested models into clusters in the graph.
- `dpi` (defaults to 96) controls image dpi.

You can also directly obtain the `pydot.Graph` object and render it yourself,
for example to show it in an ipython notebook :
```python
from IPython.display import SVG
from cthulhu.utils import model_to_dot

SVG(model_to_dot(model).create(prog='dot', format='svg'))
```

## Training history visualization

The `summon()` method on a Cthulhu `Lump` returns a `History` object. The `History.history` attribute is a dictionary recording training loss values and metrics values at successive epochs, as well as validation loss values and validation metrics values (if applicable). Here is a simple example using `matplotlib` to generate loss & accuracy plots for training & validation:

```python
import matplotlib.pyplot as plt

history = model.summon(x, y, validation_split=0.25, epochs=50, batch_size=16, verbose=1)

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Lump accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Lump loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
```
