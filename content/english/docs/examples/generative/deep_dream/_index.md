---
title: Deep Dream
toc: true
weight: 13
type: docs
---

> - Original Link : [https://keras.io/examples/generative/deep_dream/](https://keras.io/examples/generative/deep_dream/)
> - Last Checked at : 2024-11-23

**Author:** [fchollet](https://twitter.com/fchollet)  
**Date created:** 2016/01/13  
**Last modified:** 2020/05/02  
**Description:** Generating Deep Dreams with Keras.

{{< hextra/hero-button
    text="ⓘ This example uses Keras 3"
    style="background: rgb(23, 132, 133); margin: 1em 0 0.5em 0; pointer-events: none;" >}}

{{< cards cols="2" >}}
{{< card link="https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/generative/ipynb/deep_dream.ipynb" title="Colab" tag="Colab" tagType="warning">}}
{{< card link="https://github.com/keras-team/keras-io/blob/master/examples/generative/deep_dream.py" title="GitHub source" tag="GitHub">}}
{{< /cards >}}
ⓘ This example uses Keras 3

## Introduction

"Deep dream" is an image-filtering technique which consists of taking an image classification model, and running gradient ascent over an input image to try to maximize the activations of specific layers (and sometimes, specific units in specific layers) for this input. It produces hallucination-like visuals.

It was first introduced by Alexander Mordvintsev from Google in July 2015.

Process:

- Load the original image.
- Define a number of processing scales ("octaves"), from smallest to largest.
- Resize the original image to the smallest scale.
- For every scale, starting with the smallest (i.e. current one): - Run gradient ascent - Upscale image to the next scale - Reinject the detail that was lost at upscaling time
- Stop when we are back to the original size. To obtain the detail lost during upscaling, we simply take the original image, shrink it down, upscale it, and compare the result to the (resized) original image.

## Setup

```python
import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import tensorflow as tf
import keras
from keras.applications import inception_v3

base_image_path = keras.utils.get_file("sky.jpg", "https://i.imgur.com/aGBdQyK.jpg")
result_prefix = "sky_dream"

# These are the names of the layers
# for which we try to maximize activation,
# as well as their weight in the final loss
# we try to maximize.
# You can tweak these setting to obtain new visual effects.
layer_settings = {
    "mixed4": 1.0,
    "mixed5": 1.5,
    "mixed6": 2.0,
    "mixed7": 2.5,
}

# Playing with these hyperparameters will also allow you to achieve new effects
step = 0.01  # Gradient ascent step size
num_octave = 3  # Number of scales at which to run gradient ascent
octave_scale = 1.4  # Size ratio between scales
iterations = 20  # Number of ascent steps per scale
max_loss = 15.0
```

This is our base image:

```python
from IPython.display import Image, display

display(Image(base_image_path))
```

![jpeg](/images/examples/generative/deep_dream/deep_dream_5_0.jpg)

Let's set up some image preprocessing/deprocessing utilities:

```python
def preprocess_image(image_path):
    # Util function to open, resize and format pictures
    # into appropriate arrays.
    img = keras.utils.load_img(image_path)
    img = keras.utils.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = inception_v3.preprocess_input(img)
    return img


def deprocess_image(x):
    # Util function to convert a NumPy array into a valid image.
    x = x.reshape((x.shape[1], x.shape[2], 3))
    # Undo inception v3 preprocessing
    x /= 2.0
    x += 0.5
    x *= 255.0
    # Convert to uint8 and clip to the valid range [0, 255]
    x = np.clip(x, 0, 255).astype("uint8")
    return x
```

## Compute the Deep Dream loss

First, build a feature extraction model to retrieve the activations of our target layers given an input image.

```python
# Build an InceptionV3 model loaded with pre-trained ImageNet weights
model = inception_v3.InceptionV3(weights="imagenet", include_top=False)

# Get the symbolic outputs of each "key" layer (we gave them unique names).
outputs_dict = dict(
    [
        (layer.name, layer.output)
        for layer in [model.get_layer(name) for name in layer_settings.keys()]
    ]
)

# Set up a model that returns the activation values for every target layer
# (as a dict)
feature_extractor = keras.Model(inputs=model.inputs, outputs=outputs_dict)
```

The actual loss computation is very simple:

```python
def compute_loss(input_image):
    features = feature_extractor(input_image)
    # Initialize the loss
    loss = tf.zeros(shape=())
    for name in features.keys():
        coeff = layer_settings[name]
        activation = features[name]
        # We avoid border artifacts by only involving non-border pixels in the loss.
        scaling = tf.reduce_prod(tf.cast(tf.shape(activation), "float32"))
        loss += coeff * tf.reduce_sum(tf.square(activation[:, 2:-2, 2:-2, :])) / scaling
    return loss
```

## Set up the gradient ascent loop for one octave

```python
@tf.function
def gradient_ascent_step(img, learning_rate):
    with tf.GradientTape() as tape:
        tape.watch(img)
        loss = compute_loss(img)
    # Compute gradients.
    grads = tape.gradient(loss, img)
    # Normalize gradients.
    grads /= tf.maximum(tf.reduce_mean(tf.abs(grads)), 1e-6)
    img += learning_rate * grads
    return loss, img


def gradient_ascent_loop(img, iterations, learning_rate, max_loss=None):
    for i in range(iterations):
        loss, img = gradient_ascent_step(img, learning_rate)
        if max_loss is not None and loss > max_loss:
            break
        print("... Loss value at step %d: %.2f" % (i, loss))
    return img
```

## Run the training loop, iterating over different octaves

```python
original_img = preprocess_image(base_image_path)
original_shape = original_img.shape[1:3]

successive_shapes = [original_shape]
for i in range(1, num_octave):
    shape = tuple([int(dim / (octave_scale**i)) for dim in original_shape])
    successive_shapes.append(shape)
successive_shapes = successive_shapes[::-1]
shrunk_original_img = tf.image.resize(original_img, successive_shapes[0])

img = tf.identity(original_img)  # Make a copy
for i, shape in enumerate(successive_shapes):
    print("Processing octave %d with shape %s" % (i, shape))
    img = tf.image.resize(img, shape)
    img = gradient_ascent_loop(
        img, iterations=iterations, learning_rate=step, max_loss=max_loss
    )
    upscaled_shrunk_original_img = tf.image.resize(shrunk_original_img, shape)
    same_size_original = tf.image.resize(original_img, shape)
    lost_detail = same_size_original - upscaled_shrunk_original_img

    img += lost_detail
    shrunk_original_img = tf.image.resize(original_img, shape)

keras.utils.save_img(result_prefix + ".png", deprocess_image(img.numpy()))
```

{{% details title="Result" closed="true" %}}

```plain
Processing octave 0 with shape (326, 489)
... Loss value at step 0: 0.45
... Loss value at step 1: 0.63
... Loss value at step 2: 0.91
... Loss value at step 3: 1.24
... Loss value at step 4: 1.57
... Loss value at step 5: 1.91
... Loss value at step 6: 2.20
... Loss value at step 7: 2.50
... Loss value at step 8: 2.82
... Loss value at step 9: 3.11
... Loss value at step 10: 3.40
... Loss value at step 11: 3.70
... Loss value at step 12: 3.95
... Loss value at step 13: 4.20
... Loss value at step 14: 4.48
... Loss value at step 15: 4.72
... Loss value at step 16: 4.99
... Loss value at step 17: 5.23
... Loss value at step 18: 5.47
... Loss value at step 19: 5.69
Processing octave 1 with shape (457, 685)
... Loss value at step 0: 1.11
... Loss value at step 1: 1.77
... Loss value at step 2: 2.35
... Loss value at step 3: 2.82
... Loss value at step 4: 3.25
... Loss value at step 5: 3.67
... Loss value at step 6: 4.05
... Loss value at step 7: 4.44
... Loss value at step 8: 4.79
... Loss value at step 9: 5.15
... Loss value at step 10: 5.50
... Loss value at step 11: 5.84
... Loss value at step 12: 6.18
... Loss value at step 13: 6.49
... Loss value at step 14: 6.82
... Loss value at step 15: 7.12
... Loss value at step 16: 7.42
... Loss value at step 17: 7.71
... Loss value at step 18: 8.01
... Loss value at step 19: 8.30
Processing octave 2 with shape (640, 960)
... Loss value at step 0: 1.27
... Loss value at step 1: 2.02
... Loss value at step 2: 2.63
... Loss value at step 3: 3.15
... Loss value at step 4: 3.66
... Loss value at step 5: 4.12
... Loss value at step 6: 4.58
... Loss value at step 7: 5.01
... Loss value at step 8: 5.42
... Loss value at step 9: 5.80
... Loss value at step 10: 6.19
... Loss value at step 11: 6.54
... Loss value at step 12: 6.89
... Loss value at step 13: 7.22
... Loss value at step 14: 7.57
... Loss value at step 15: 7.88
... Loss value at step 16: 8.21
... Loss value at step 17: 8.53
... Loss value at step 18: 8.80
... Loss value at step 19: 9.10
```

{{% /details %}}

Display the result.

You can use the trained model hosted on [Hugging Face Hub](https://huggingface.co/keras-io/deep-dream) and try the demo on [Hugging Face Spaces](https://huggingface.co/spaces/keras-io/deep-dream).

```python
display(Image(result_prefix + ".png"))
```

![png](/images/examples/generative/deep_dream/deep_dream_17_0.png)
