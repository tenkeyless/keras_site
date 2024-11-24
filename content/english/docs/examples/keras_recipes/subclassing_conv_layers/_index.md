---
title: Customizing the convolution operation of a Conv2D layer
toc: true
weight: 5
type: docs
---

{{< keras/original checkedAt="2024-11-24" >}}

**Author:** [lukewood](https://lukewood.xyz)  
**Date created:** 2021/03/11  
**Last modified:** 2021/03/11  
**Description:** This example shows how to implement custom convolution layers using the `Conv.convolution_op()` API.

{{< keras/version v=3 >}}

{{< cards cols="2" >}}
{{< card link="https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/keras_recipes/ipynb/subclassing_conv_layers.ipynb" title="Colab" tag="Colab" tagType="warning">}}
{{< card link="https://github.com/keras-team/keras-io/blob/master/examples/keras_recipes/subclassing_conv_layers.py" title="GitHub source" tag="GitHub">}}
{{< /cards >}}

## Introduction

You may sometimes need to implement custom versions of convolution layers like `Conv1D` and `Conv2D`. Keras enables you do this without implementing the entire layer from scratch: you can reuse most of the base convolution layer and just customize the convolution op itself via the `convolution_op()` method.

This method was introduced in Keras 2.7. So before using the `convolution_op()` API, ensure that you are running Keras version 2.7.0 or greater.

## A Simple `StandardizedConv2D` implementation

There are two ways to use the `Conv.convolution_op()` API. The first way is to override the `convolution_op()` method on a convolution layer subclass. Using this approach, we can quickly implement a [StandardizedConv2D](https://arxiv.org/abs/1903.10520) as shown below.

```python
import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
import keras
from keras import layers
import numpy as np


class StandardizedConv2DWithOverride(layers.Conv2D):
    def convolution_op(self, inputs, kernel):
        mean, var = tf.nn.moments(kernel, axes=[0, 1, 2], keepdims=True)
        return tf.nn.conv2d(
            inputs,
            (kernel - mean) / tf.sqrt(var + 1e-10),
            padding="VALID",
            strides=list(self.strides),
            name=self.__class__.__name__,
        )
```

The other way to use the `Conv.convolution_op()` API is to directly call the `convolution_op()` method from the `call()` method of a convolution layer subclass. A comparable class implemented using this approach is shown below.

```python
class StandardizedConv2DWithCall(layers.Conv2D):
    def call(self, inputs):
        mean, var = tf.nn.moments(self.kernel, axes=[0, 1, 2], keepdims=True)
        result = self.convolution_op(
            inputs, (self.kernel - mean) / tf.sqrt(var + 1e-10)
        )
        if self.use_bias:
            result = result + self.bias
        return result
```

## Example Usage

Both of these layers work as drop-in replacements for `Conv2D`. The following demonstration performs classification on the MNIST dataset.

```python
# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = keras.Sequential(
    [
        keras.layers.Input(shape=input_shape),
        StandardizedConv2DWithCall(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        StandardizedConv2DWithOverride(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()
```

{{% details title="Result" closed="true" %}}

```plain
x_train shape: (60000, 28, 28, 1)
60000 train samples
10000 test samples
```

```plain
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape              ┃    Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ standardized_conv2d_with_call   │ (None, 26, 26, 32)        │        320 │
│ (StandardizedConv2DWithCall)    │                           │            │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ max_pooling2d (MaxPooling2D)    │ (None, 13, 13, 32)        │          0 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ standardized_conv2d_with_overr… │ (None, 11, 11, 64)        │     18,496 │
│ (StandardizedConv2DWithOverrid… │                           │            │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ max_pooling2d_1 (MaxPooling2D)  │ (None, 5, 5, 64)          │          0 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ flatten (Flatten)               │ (None, 1600)              │          0 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ dropout (Dropout)               │ (None, 1600)              │          0 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ dense (Dense)                   │ (None, 10)                │     16,010 │
└─────────────────────────────────┴───────────────────────────┴────────────┘
 Total params: 34,826 (136.04 KB)
 Trainable params: 34,826 (136.04 KB)
 Non-trainable params: 0 (0.00 B)
```

{{% /details %}}

```python
batch_size = 128
epochs = 5

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=batch_size, epochs=5, validation_split=0.1)
```

{{% details title="Result" closed="true" %}}

```plain
Epoch 1/5
  64/422 ━━━[37m━━━━━━━━━━━━━━━━━  0s 2ms/step - accuracy: 0.4439 - loss: 13.1274

WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1699557098.952525   26800 device_compiler.h:187] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.

 422/422 ━━━━━━━━━━━━━━━━━━━━ 10s 14ms/step - accuracy: 0.7277 - loss: 4.5649 - val_accuracy: 0.9690 - val_loss: 0.1140
Epoch 2/5
 422/422 ━━━━━━━━━━━━━━━━━━━━ 2s 3ms/step - accuracy: 0.9311 - loss: 0.2493 - val_accuracy: 0.9798 - val_loss: 0.0795
Epoch 3/5
 422/422 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.9531 - loss: 0.1655 - val_accuracy: 0.9838 - val_loss: 0.0610
Epoch 4/5
 422/422 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.9652 - loss: 0.1201 - val_accuracy: 0.9847 - val_loss: 0.0577
Epoch 5/5
 422/422 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.9687 - loss: 0.1059 - val_accuracy: 0.9870 - val_loss: 0.0525

<keras.src.callbacks.history.History at 0x7fed258da200>
```

{{% /details %}}

## Conclusion

The `Conv.convolution_op()` API provides an easy and readable way to implement custom convolution layers. A `StandardizedConvolution` implementation using the API is quite terse, consisting of only four lines of code.
