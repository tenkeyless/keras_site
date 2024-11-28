---
title: Transfer learning & fine-tuning
linkTitle: Transfer learning & fine-tuning
toc: true
weight: 14
type: docs
---

{{< keras/original checkedAt="2024-11-18" >}}

**{{< t f_author >}}** [fchollet](https://twitter.com/fchollet)  
**{{< t f_date_created >}}** 2020/04/15  
**{{< t f_last_modified >}}** 2023/06/25  
**{{< t f_description >}}** Complete guide to transfer learning & fine-tuning in Keras.

{{< cards cols="2" >}}
{{< card link="https://colab.research.google.com/github/keras-team/keras-io/blob/master/guides/ipynb/transfer_learning.ipynb" title="Colab" tag="Colab" tagType="warning">}}
{{< card link="https://github.com/keras-team/keras-io/blob/master/guides/transfer_learning.py" title="GitHub" tag="GitHub">}}
{{< /cards >}}

## Setup

```python
import numpy as np
import keras
from keras import layers
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
```

## Introduction

**Transfer learning** consists of taking features learned on one problem, and leveraging them on a new, similar problem. For instance, features from a model that has learned to identify racoons may be useful to kick-start a model meant to identify tanukis.

Transfer learning is usually done for tasks where your dataset has too little data to train a full-scale model from scratch.

The most common incarnation of transfer learning in the context of deep learning is the following workflow:

1.  Take layers from a previously trained model.
2.  Freeze them, so as to avoid destroying any of the information they contain during future training rounds.
3.  Add some new, trainable layers on top of the frozen layers. They will learn to turn the old features into predictions on a new dataset.
4.  Train the new layers on your dataset.

A last, optional step, is **fine-tuning**, which consists of unfreezing the entire model you obtained above (or part of it), and re-training it on the new data with a very low learning rate. This can potentially achieve meaningful improvements, by incrementally adapting the pretrained features to the new data.

First, we will go over the Keras `trainable` API in detail, which underlies most transfer learning & fine-tuning workflows.

Then, we'll demonstrate the typical workflow by taking a model pretrained on the ImageNet dataset, and retraining it on the Kaggle "cats vs dogs" classification dataset.

This is adapted from [Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python) and the 2016 blog post ["building powerful image classification models using very little data"](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html).

## Freezing layers: understanding the `trainable` attribute

Layers & models have three weight attributes:

- `weights` is the list of all weights variables of the layer.
- `trainable_weights` is the list of those that are meant to be updated (via gradient descent) to minimize the loss during training.
- `non_trainable_weights` is the list of those that aren't meant to be trained. Typically they are updated by the model during the forward pass.

**Example: the `Dense` layer has 2 trainable weights (kernel & bias)**

```python
layer = keras.layers.Dense(3)
layer.build((None, 4))  # Create the weights

print("weights:", len(layer.weights))
print("trainable_weights:", len(layer.trainable_weights))
print("non_trainable_weights:", len(layer.non_trainable_weights))
```

{{% details title="Result" closed="true" %}}

```plain
weights: 2
trainable_weights: 2
non_trainable_weights: 0
```

{{% /details %}}

## Available guides

In general, all weights are trainable weights. The only built-in layer that has non-trainable weights is the `BatchNormalization` layer. It uses non-trainable weights to keep track of the mean and variance of its inputs during training. To learn how to use non-trainable weights in your own custom layers, see the [guide to writing new layers from scratch]({{< relref "/docs/guides/making_new_layers_and_models_via_subclassing" >}}).

**Example: the `BatchNormalization` layer has 2 trainable weights and 2 non-trainable weights**

```python
layer = keras.layers.BatchNormalization()
layer.build((None, 4))  # Create the weights

print("weights:", len(layer.weights))
print("trainable_weights:", len(layer.trainable_weights))
print("non_trainable_weights:", len(layer.non_trainable_weights))
```

{{% details title="Result" closed="true" %}}

```plain
weights: 4
trainable_weights: 2
non_trainable_weights: 2
```

{{% /details %}}

Layers & models also feature a boolean attribute `trainable`. Its value can be changed. Setting `layer.trainable` to `False` moves all the layer's weights from trainable to non-trainable. This is called "freezing" the layer: the state of a frozen layer won't be updated during training (either when training with `fit()` or when training with any custom loop that relies on `trainable_weights` to apply gradient updates).

**Example: setting `trainable` to `False`**

```python
layer = keras.layers.Dense(3)
layer.build((None, 4))  # Create the weights
layer.trainable = False  # Freeze the layer

print("weights:", len(layer.weights))
print("trainable_weights:", len(layer.trainable_weights))
print("non_trainable_weights:", len(layer.non_trainable_weights))
```

{{% details title="Result" closed="true" %}}

```plain
weights: 2
trainable_weights: 0
non_trainable_weights: 2
```

{{% /details %}}

When a trainable weight becomes non-trainable, its value is no longer updated during training.

```python
# Make a model with 2 layers
layer1 = keras.layers.Dense(3, activation="relu")
layer2 = keras.layers.Dense(3, activation="sigmoid")
model = keras.Sequential([keras.Input(shape=(3,)), layer1, layer2])

# Freeze the first layer
layer1.trainable = False

# Keep a copy of the weights of layer1 for later reference
initial_layer1_weights_values = layer1.get_weights()

# Train the model
model.compile(optimizer="adam", loss="mse")
model.fit(np.random.random((2, 3)), np.random.random((2, 3)))

# Check that the weights of layer1 have not changed during training
final_layer1_weights_values = layer1.get_weights()
np.testing.assert_allclose(
    initial_layer1_weights_values[0], final_layer1_weights_values[0]
)
np.testing.assert_allclose(
    initial_layer1_weights_values[1], final_layer1_weights_values[1]
)
```

{{% details title="Result" closed="true" %}}

```plain
 1/1 ━━━━━━━━━━━━━━━━━━━━ 1s 766ms/step - loss: 0.0615
```

{{% /details %}}

Do not confuse the `layer.trainable` attribute with the argument `training` in `layer.__call__()` (which controls whether the layer should run its forward pass in inference mode or training mode). For more information, see the [Keras FAQ]({{< relref "/docs/getting_started/faq/#whats-the-difference-between-the-training-argument-in-call-and-the-trainable-attribute" >}}).

## Recursive setting of the `trainable` attribute

If you set `trainable = False` on a model or on any layer that has sublayers, all children layers become non-trainable as well.

**Example:**

```python
inner_model = keras.Sequential(
    [
        keras.Input(shape=(3,)),
        keras.layers.Dense(3, activation="relu"),
        keras.layers.Dense(3, activation="relu"),
    ]
)

model = keras.Sequential(
    [
        keras.Input(shape=(3,)),
        inner_model,
        keras.layers.Dense(3, activation="sigmoid"),
    ]
)

model.trainable = False  # Freeze the outer model

assert inner_model.trainable == False  # All layers in `model` are now frozen
assert inner_model.layers[0].trainable == False  # `trainable` is propagated recursively
```

## The typical transfer-learning workflow

This leads us to how a typical transfer learning workflow can be implemented in Keras:

1.  Instantiate a base model and load pre-trained weights into it.
2.  Freeze all layers in the base model by setting `trainable = False`.
3.  Create a new model on top of the output of one (or several) layers from the base model.
4.  Train your new model on your new dataset.

Note that an alternative, more lightweight workflow could also be:

1.  Instantiate a base model and load pre-trained weights into it.
2.  Run your new dataset through it and record the output of one (or several) layers from the base model. This is called **feature extraction**.
3.  Use that output as input data for a new, smaller model.

A key advantage of that second workflow is that you only run the base model once on your data, rather than once per epoch of training. So it's a lot faster & cheaper.

An issue with that second workflow, though, is that it doesn't allow you to dynamically modify the input data of your new model during training, which is required when doing data augmentation, for instance. Transfer learning is typically used for tasks when your new dataset has too little data to train a full-scale model from scratch, and in such scenarios data augmentation is very important. So in what follows, we will focus on the first workflow.

Here's what the first workflow looks like in Keras:

First, instantiate a base model with pre-trained weights.

```python
base_model = keras.applications.Xception(
    weights='imagenet',  # Load weights pre-trained on ImageNet.
    input_shape=(150, 150, 3),
    include_top=False)  # Do not include the ImageNet classifier at the top.
```

Then, freeze the base model.

```python
base_model.trainable = False
```

Create a new model on top.

```python
inputs = keras.Input(shape=(150, 150, 3))
# We make sure that the base_model is running in inference mode here,
# by passing `training=False`. This is important for fine-tuning, as you will
# learn in a few paragraphs.
x = base_model(inputs, training=False)
# Convert features of shape `base_model.output_shape[1:]` to vectors
x = keras.layers.GlobalAveragePooling2D()(x)
# A Dense classifier with a single unit (binary classification)
outputs = keras.layers.Dense(1)(x)
model = keras.Model(inputs, outputs)
```

Train the model on new data.

```python
model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=[keras.metrics.BinaryAccuracy()])
model.fit(new_dataset, epochs=20, callbacks=..., validation_data=...)
```

## Fine-tuning

Once your model has converged on the new data, you can try to unfreeze all or part of the base model and retrain the whole model end-to-end with a very low learning rate.

This is an optional last step that can potentially give you incremental improvements. It could also potentially lead to quick overfitting – keep that in mind.

It is critical to only do this step _after_ the model with frozen layers has been trained to convergence. If you mix randomly-initialized trainable layers with trainable layers that hold pre-trained features, the randomly-initialized layers will cause very large gradient updates during training, which will destroy your pre-trained features.

It's also critical to use a very low learning rate at this stage, because you are training a much larger model than in the first round of training, on a dataset that is typically very small. As a result, you are at risk of overfitting very quickly if you apply large weight updates. Here, you only want to readapt the pretrained weights in an incremental way.

This is how to implement fine-tuning of the whole base model:

```python
# Unfreeze the base model
base_model.trainable = True

# It's important to recompile your model after you make any changes
# to the `trainable` attribute of any inner layer, so that your changes
# are take into account
model.compile(optimizer=keras.optimizers.Adam(1e-5),  # Very low learning rate
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=[keras.metrics.BinaryAccuracy()])

# Train end-to-end. Be careful to stop before you overfit!
model.fit(new_dataset, epochs=10, callbacks=..., validation_data=...)
```

**Important note about `compile()` and `trainable`**

Calling `compile()` on a model is meant to "freeze" the behavior of that model. This implies that the `trainable` attribute values at the time the model is compiled should be preserved throughout the lifetime of that model, until `compile` is called again. Hence, if you change any `trainable` value, make sure to call `compile()` again on your model for your changes to be taken into account.

**Important notes about `BatchNormalization` layer**

Many image models contain `BatchNormalization` layers. That layer is a special case on every imaginable count. Here are a few things to keep in mind.

- `BatchNormalization` contains 2 non-trainable weights that get updated during training. These are the variables tracking the mean and variance of the inputs.
- When you set `bn_layer.trainable = False`, the `BatchNormalization` layer will run in inference mode, and will not update its mean & variance statistics. This is not the case for other layers in general, as [weight trainability & inference/training modes are two orthogonal concepts]({{< relref "/docs/getting_started/faq/#whats-the-difference-between-the-training-argument-in-call-and-the-trainable-attribute" >}}). But the two are tied in the case of the `BatchNormalization` layer.
- When you unfreeze a model that contains `BatchNormalization` layers in order to do fine-tuning, you should keep the `BatchNormalization` layers in inference mode by passing `training=False` when calling the base model. Otherwise the updates applied to the non-trainable weights will suddenly destroy what the model has learned.

You'll see this pattern in action in the end-to-end example at the end of this guide.

## An end-to-end example: fine-tuning an image classification model on a cats vs. dogs dataset

To solidify these concepts, let's walk you through a concrete end-to-end transfer learning & fine-tuning example. We will load the Xception model, pre-trained on ImageNet, and use it on the Kaggle "cats vs. dogs" classification dataset.

### Getting the data

First, let's fetch the cats vs. dogs dataset using TFDS. If you have your own dataset, you'll probably want to use the utility [`keras.utils.image_dataset_from_directory`]({{< relref "/docs/api/data_loading/image#image_dataset_from_directory-function" >}}) to generate similar labeled dataset objects from a set of images on disk filed into class-specific folders.

Transfer learning is most useful when working with very small datasets. To keep our dataset small, we will use 40% of the original training data (25,000 images) for training, 10% for validation, and 10% for testing.

```python
tfds.disable_progress_bar()

train_ds, validation_ds, test_ds = tfds.load(
    "cats_vs_dogs",
    # Reserve 10% for validation and 10% for test
    split=["train[:40%]", "train[40%:50%]", "train[50%:60%]"],
    as_supervised=True,  # Include labels
)

print(f"Number of training samples: {train_ds.cardinality()}")
print(f"Number of validation samples: {validation_ds.cardinality()}")
print(f"Number of test samples: {test_ds.cardinality()}")
```

{{% details title="Result" closed="true" %}}

```plain
 Downloading and preparing dataset 786.68 MiB (download: 786.68 MiB, generated: Unknown size, total: 786.68 MiB) to /home/mattdangerw/tensorflow_datasets/cats_vs_dogs/4.0.0...

WARNING:absl:1738 images were corrupted and were skipped

 Dataset cats_vs_dogs downloaded and prepared to /home/mattdangerw/tensorflow_datasets/cats_vs_dogs/4.0.0. Subsequent calls will reuse this data.
Number of training samples: 9305
Number of validation samples: 2326
Number of test samples: 2326
```

{{% /details %}}

These are the first 9 images in the training dataset – as you can see, they're all different sizes.

```python
plt.figure(figsize=(10, 10))
for i, (image, label) in enumerate(train_ds.take(9)):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image)
    plt.title(int(label))
    plt.axis("off")
```

![png](/images/guides/transfer_learning/transfer_learning_21_0.png)

We can also see that label 1 is "dog" and label 0 is "cat".

### Standardizing the data

Our raw images have a variety of sizes. In addition, each pixel consists of 3 integer values between 0 and 255 (RGB level values). This isn't a great fit for feeding a neural network. We need to do 2 things:

- Standardize to a fixed image size. We pick 150x150.
- Normalize pixel values between -1 and 1. We'll do this using a `Normalization` layer as part of the model itself.

In general, it's a good practice to develop models that take raw data as input, as opposed to models that take already-preprocessed data. The reason being that, if your model expects preprocessed data, any time you export your model to use it elsewhere (in a web browser, in a mobile app), you'll need to reimplement the exact same preprocessing pipeline. This gets very tricky very quickly. So we should do the least possible amount of preprocessing before hitting the model.

Here, we'll do image resizing in the data pipeline (because a deep neural network can only process contiguous batches of data), and we'll do the input value scaling as part of the model, when we create it.

Let's resize images to 150x150:

```python
resize_fn = keras.layers.Resizing(150, 150)

train_ds = train_ds.map(lambda x, y: (resize_fn(x), y))
validation_ds = validation_ds.map(lambda x, y: (resize_fn(x), y))
test_ds = test_ds.map(lambda x, y: (resize_fn(x), y))
```

### Using random data augmentation

When you don't have a large image dataset, it's a good practice to artificially introduce sample diversity by applying random yet realistic transformations to the training images, such as random horizontal flipping or small random rotations. This helps expose the model to different aspects of the training data while slowing down overfitting.

```python
augmentation_layers = [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
]


def data_augmentation(x):
    for layer in augmentation_layers:
        x = layer(x)
    return x


train_ds = train_ds.map(lambda x, y: (data_augmentation(x), y))
```

Let's batch the data and use prefetching to optimize loading speed.

```python
from tensorflow import data as tf_data

batch_size = 64

train_ds = train_ds.batch(batch_size).prefetch(tf_data.AUTOTUNE).cache()
validation_ds = validation_ds.batch(batch_size).prefetch(tf_data.AUTOTUNE).cache()
test_ds = test_ds.batch(batch_size).prefetch(tf_data.AUTOTUNE).cache()
```

Let's visualize what the first image of the first batch looks like after various random transformations:

```python
for images, labels in train_ds.take(1):
    plt.figure(figsize=(10, 10))
    first_image = images[0]
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        augmented_image = data_augmentation(np.expand_dims(first_image, 0))
        plt.imshow(np.array(augmented_image[0]).astype("int32"))
        plt.title(int(labels[0]))
        plt.axis("off")
```

![png](/images/guides/transfer_learning/transfer_learning_30_0.png)

## Build a model

Now let's built a model that follows the blueprint we've explained earlier.

Note that:

- We add a `Rescaling` layer to scale input values (initially in the `[0, 255]` range) to the `[-1, 1]` range.
- We add a `Dropout` layer before the classification layer, for regularization.
- We make sure to pass `training=False` when calling the base model, so that it runs in inference mode, so that batchnorm statistics don't get updated even after we unfreeze the base model for fine-tuning.

```python
base_model = keras.applications.Xception(
    weights="imagenet",  # Load weights pre-trained on ImageNet.
    input_shape=(150, 150, 3),
    include_top=False,
)  # Do not include the ImageNet classifier at the top.

# Freeze the base_model
base_model.trainable = False

# Create new model on top
inputs = keras.Input(shape=(150, 150, 3))

# Pre-trained Xception weights requires that input be scaled
# from (0, 255) to a range of (-1., +1.), the rescaling layer
# outputs: `(inputs * scale) + offset`
scale_layer = keras.layers.Rescaling(scale=1 / 127.5, offset=-1)
x = scale_layer(inputs)

# The base model contains batchnorm layers. We want to keep them in inference mode
# when we unfreeze the base model for fine-tuning, so we make sure that the
# base_model is running in inference mode here.
x = base_model(x, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x)  # Regularize with dropout
outputs = keras.layers.Dense(1)(x)
model = keras.Model(inputs, outputs)

model.summary(show_trainable=True)
```

{{% details title="Result" closed="true" %}}

```plain
Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/xception/xception_weights_tf_dim_ordering_tf_kernels_notop.h5
 83683744/83683744 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step
```

```plain
Model: "functional_4"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━┓
┃ Layer (type)                ┃ Output Shape             ┃ Param # ┃ Trai… ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━┩
│ input_layer_4 (InputLayer)  │ (None, 150, 150, 3)      │       0 │   -   │
├─────────────────────────────┼──────────────────────────┼─────────┼───────┤
│ rescaling (Rescaling)       │ (None, 150, 150, 3)      │       0 │   -   │
├─────────────────────────────┼──────────────────────────┼─────────┼───────┤
│ xception (Functional)       │ (None, 5, 5, 2048)       │ 20,861… │   N   │
├─────────────────────────────┼──────────────────────────┼─────────┼───────┤
│ global_average_pooling2d    │ (None, 2048)             │       0 │   -   │
│ (GlobalAveragePooling2D)    │                          │         │       │
├─────────────────────────────┼──────────────────────────┼─────────┼───────┤
│ dropout (Dropout)           │ (None, 2048)             │       0 │   -   │
├─────────────────────────────┼──────────────────────────┼─────────┼───────┤
│ dense_7 (Dense)             │ (None, 1)                │   2,049 │   Y   │
└─────────────────────────────┴──────────────────────────┴─────────┴───────┘
 Total params: 20,863,529 (79.59 MB)
 Trainable params: 2,049 (8.00 KB)
 Non-trainable params: 20,861,480 (79.58 MB)
```

{{% /details %}}

## Train the top layer

```python
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy()],
)

epochs = 2
print("Fitting the top layer of the model")
model.fit(train_ds, epochs=epochs, validation_data=validation_ds)
```

{{% details title="Result" closed="true" %}}

```plain
Fitting the top layer of the model
Epoch 1/2
  78/146 ━━━━━━━━━━[37m━━━━━━━━━━  15s 226ms/step - binary_accuracy: 0.7995 - loss: 0.4088

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 136/146 ━━━━━━━━━━━━━━━━━━[37m━━  2s 231ms/step - binary_accuracy: 0.8430 - loss: 0.3298

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

 143/146 ━━━━━━━━━━━━━━━━━━━[37m━  0s 231ms/step - binary_accuracy: 0.8464 - loss: 0.3235

Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9

 144/146 ━━━━━━━━━━━━━━━━━━━[37m━  0s 231ms/step - binary_accuracy: 0.8468 - loss: 0.3226

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

 146/146 ━━━━━━━━━━━━━━━━━━━━ 0s 260ms/step - binary_accuracy: 0.8478 - loss: 0.3209

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

 146/146 ━━━━━━━━━━━━━━━━━━━━ 54s 317ms/step - binary_accuracy: 0.8482 - loss: 0.3200 - val_binary_accuracy: 0.9667 - val_loss: 0.0877
Epoch 2/2
 146/146 ━━━━━━━━━━━━━━━━━━━━ 7s 51ms/step - binary_accuracy: 0.9483 - loss: 0.1232 - val_binary_accuracy: 0.9705 - val_loss: 0.0786

<keras.src.callbacks.history.History at 0x7fc8b7f1db70>
```

{{% /details %}}

## Do a round of fine-tuning of the entire model

Finally, let's unfreeze the base model and train the entire model end-to-end with a low learning rate.

Importantly, although the base model becomes trainable, it is still running in inference mode since we passed `training=False` when calling it when we built the model. This means that the batch normalization layers inside won't update their batch statistics. If they did, they would wreck havoc on the representations learned by the model so far.

```python
# Unfreeze the base_model. Note that it keeps running in inference mode
# since we passed `training=False` when calling it. This means that
# the batchnorm layers will not update their batch statistics.
# This prevents the batchnorm layers from undoing all the training
# we've done so far.
base_model.trainable = True
model.summary(show_trainable=True)

model.compile(
    optimizer=keras.optimizers.Adam(1e-5),  # Low learning rate
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy()],
)

epochs = 1
print("Fitting the end-to-end model")
model.fit(train_ds, epochs=epochs, validation_data=validation_ds)
```

{{% details title="Result" closed="true" %}}

```plain
Model: "functional_4"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━┓
┃ Layer (type)                ┃ Output Shape             ┃ Param # ┃ Trai… ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━┩
│ input_layer_4 (InputLayer)  │ (None, 150, 150, 3)      │       0 │   -   │
├─────────────────────────────┼──────────────────────────┼─────────┼───────┤
│ rescaling (Rescaling)       │ (None, 150, 150, 3)      │       0 │   -   │
├─────────────────────────────┼──────────────────────────┼─────────┼───────┤
│ xception (Functional)       │ (None, 5, 5, 2048)       │ 20,861… │   Y   │
├─────────────────────────────┼──────────────────────────┼─────────┼───────┤
│ global_average_pooling2d    │ (None, 2048)             │       0 │   -   │
│ (GlobalAveragePooling2D)    │                          │         │       │
├─────────────────────────────┼──────────────────────────┼─────────┼───────┤
│ dropout (Dropout)           │ (None, 2048)             │       0 │   -   │
├─────────────────────────────┼──────────────────────────┼─────────┼───────┤
│ dense_7 (Dense)             │ (None, 1)                │   2,049 │   Y   │
└─────────────────────────────┴──────────────────────────┴─────────┴───────┘
 Total params: 20,867,629 (79.60 MB)
 Trainable params: 20,809,001 (79.38 MB)
 Non-trainable params: 54,528 (213.00 KB)
 Optimizer params: 4,100 (16.02 KB)
```

```plain
Fitting the end-to-end model
 146/146 ━━━━━━━━━━━━━━━━━━━━ 75s 327ms/step - binary_accuracy: 0.8487 - loss: 0.3760 - val_binary_accuracy: 0.9494 - val_loss: 0.1160

<keras.src.callbacks.history.History at 0x7fcd1c755090>
```

{{% /details %}}

After 10 epochs, fine-tuning gains us a nice improvement here. Let's evaluate the model on the test dataset:

```python
print("Test dataset evaluation")
model.evaluate(test_ds)
```

{{% details title="Result" closed="true" %}}

```plain
Test dataset evaluation
 11/37 ━━━━━[37m━━━━━━━━━━━━━━━  1s 52ms/step - binary_accuracy: 0.9407 - loss: 0.1155

Corrupt JPEG data: 99 extraneous bytes before marker 0xd9

 37/37 ━━━━━━━━━━━━━━━━━━━━ 2s 47ms/step - binary_accuracy: 0.9427 - loss: 0.1259

[0.13755160570144653, 0.941300630569458]
```

{{% /details %}}
