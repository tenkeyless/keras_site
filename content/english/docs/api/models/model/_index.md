---
title: The Model class
toc: true
weight: 1
type: docs
---

[\[source\]](https://github.com/keras-team/keras/tree/v3.6.0/keras/src/models/model.py#L32)

### `Model` class

`keras.Model()`

A model grouping layers into an object with training/inference features.

There are three ways to instantiate a `Model`:

## With the "Functional API"

You start from `Input`, you chain layer calls to specify the model's forward pass, and finally, you create your model from inputs and outputs:

`inputs = keras.Input(shape=(37,)) x = keras.layers.Dense(32, activation="relu")(inputs) outputs = keras.layers.Dense(5, activation="softmax")(x) model = keras.Model(inputs=inputs, outputs=outputs)`

Note: Only dicts, lists, and tuples of input tensors are supported. Nested inputs are not supported (e.g. lists of list or dicts of dict).

A new Functional API model can also be created by using the intermediate tensors. This enables you to quickly extract sub-components of the model.

**Example**

`inputs = keras.Input(shape=(None, None, 3)) processed = keras.layers.RandomCrop(width=128, height=128)(inputs) conv = keras.layers.Conv2D(filters=32, kernel_size=3)(processed) pooling = keras.layers.GlobalAveragePooling2D()(conv) feature = keras.layers.Dense(10)(pooling)  full_model = keras.Model(inputs, feature) backbone = keras.Model(processed, conv) activations = keras.Model(conv, feature)`

Note that the `backbone` and `activations` models are not created with [`keras.Input`](/api/layers/core_layers/input#input-function) objects, but with the tensors that originate from [`keras.Input`](/api/layers/core_layers/input#input-function) objects. Under the hood, the layers and weights will be shared across these models, so that user can train the `full_model`, and use `backbone` or `activations` to do feature extraction. The inputs and outputs of the model can be nested structures of tensors as well, and the created models are standard Functional API models that support all the existing APIs.

## By subclassing the `Model` class

In that case, you should define your layers in `__init__()` and you should implement the model's forward pass in `call()`.

`class MyModel(keras.Model):     def __init__(self):         super().__init__()         self.dense1 = keras.layers.Dense(32, activation="relu")         self.dense2 = keras.layers.Dense(5, activation="softmax")      def call(self, inputs):         x = self.dense1(inputs)         return self.dense2(x)  model = MyModel()`

If you subclass `Model`, you can optionally have a `training` argument (boolean) in `call()`, which you can use to specify a different behavior in training and inference:

`class MyModel(keras.Model):     def __init__(self):         super().__init__()         self.dense1 = keras.layers.Dense(32, activation="relu")         self.dense2 = keras.layers.Dense(5, activation="softmax")         self.dropout = keras.layers.Dropout(0.5)      def call(self, inputs, training=False):         x = self.dense1(inputs)         x = self.dropout(x, training=training)         return self.dense2(x)  model = MyModel()`

Once the model is created, you can config the model with losses and metrics with `model.compile()`, train the model with `model.fit()`, or use the model to do prediction with `model.predict()`.

## With the `Sequential` class

In addition, [`keras.Sequential`](/api/models/sequential#sequential-class) is a special case of model where the model is purely a stack of single-input, single-output layers.

`model = keras.Sequential([     keras.Input(shape=(None, None, 3)),     keras.layers.Conv2D(filters=32, kernel_size=3), ])`

---

[\[source\]](https://github.com/keras-team/keras/tree/v3.6.0/keras/src/models/model.py#L216)

### `summary` method

`Model.summary(     line_length=None,     positions=None,     print_fn=None,     expand_nested=False,     show_trainable=False,     layer_range=None, )`

Prints a string summary of the network.

**Arguments**

- **line_length**: Total length of printed lines (e.g. set this to adapt the display to different terminal window sizes).
- **positions**: Relative or absolute positions of log elements in each line. If not provided, becomes `[0.3, 0.6, 0.70, 1.]`. Defaults to `None`.
- **print_fn**: Print function to use. By default, prints to `stdout`. If `stdout` doesn't work in your environment, change to `print`. It will be called on each line of the summary. You can set it to a custom function in order to capture the string summary.
- **expand_nested**: Whether to expand the nested models. Defaults to `False`.
- **show_trainable**: Whether to show if a layer is trainable. Defaults to `False`.
- **layer_range**: a list or tuple of 2 strings, which is the starting layer name and ending layer name (both inclusive) indicating the range of layers to be printed in summary. It also accepts regex patterns instead of exact names. In this case, the start predicate will be the first element that matches `layer_range[0]` and the end predicate will be the last element that matches `layer_range[1]`. By default `None` considers all layers of the model.

**Raises**

- **ValueError**: if `summary()` is called before the model is built.

---

[\[source\]](https://github.com/keras-team/keras/tree/v3.6.0/keras/src/models/model.py#L175)

### `get_layer` method

`Model.get_layer(name=None, index=None)`

Retrieves a layer based on either its name (unique) or index.

If `name` and `index` are both provided, `index` will take precedence. Indices are based on order of horizontal graph traversal (bottom-up).

**Arguments**

- **name**: String, name of layer.
- **index**: Integer, index of layer.

**Returns**

A layer instance.

---
