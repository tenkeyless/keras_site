---
title: CutMix, MixUp, and RandAugment image augmentation with KerasCV
linkTitle: CutMix, MixUp, and RandAugment image augmentation with KerasCV
toc: true
weight: 3
type: docs
---

> - Original Link : [https://keras.io/guides/keras_cv/cut_mix_mix_up_and_rand_augment/](https://keras.io/guides/keras_cv/cut_mix_mix_up_and_rand_augment/)
> - Last Checked at : 2024-11-19

**Author:** [lukewood](https://twitter.com/luke_wood_ml)  
**Date created:** 2022/04/08  
**Last modified:** 2022/04/08  
**Description:** Use KerasCV to augment images with CutMix, MixUp, RandAugment, and more.

{{< cards cols="2" >}}
{{< card link="https://colab.research.google.com/github/keras-team/keras-io/blob/master/guides/ipynb/keras_cv/cut_mix_mix_up_and_rand_augment.ipynb" title="Colab" tag="Colab" tagType="warning">}}
{{< card link="https://github.com/keras-team/keras-io/blob/master/guides/keras_cv/cut_mix_mix_up_and_rand_augment.py" title="GitHub source" tag="GitHub">}}
{{< /cards >}}

## Overview

KerasCV makes it easy to assemble state-of-the-art, industry-grade data augmentation pipelines for image classification and object detection tasks. KerasCV offers a wide suite of preprocessing layers implementing common data augmentation techniques.

Perhaps three of the most useful layers are [`keras_cv.layers.CutMix`]({{< relref "/docs/api/keras_cv/layers/augmentation/cut_mix#cutmix-class" >}}), [`keras_cv.layers.MixUp`]({{< relref "/docs/api/keras_cv/layers/augmentation/mix_up#mixup-class" >}}), and [`keras_cv.layers.RandAugment`]({{< relref "/docs/api/keras_cv/layers/augmentation/rand_augment#randaugment-class" >}}). These layers are used in nearly all state-of-the-art image classification pipelines.

This guide will show you how to compose these layers into your own data augmentation pipeline for image classification tasks. This guide will also walk you through the process of customizing a KerasCV data augmentation pipeline.

## Imports & setup

KerasCV uses Keras 3 to work with any of TensorFlow, PyTorch or Jax. In the guide below, we will use the `jax` backend. This guide runs in TensorFlow or PyTorch backends with zero changes, simply update the `KERAS_BACKEND` below.

```python
!pip install -q --upgrade keras-cv
!pip install -q --upgrade keras  # Upgrade to Keras 3.
```

We begin by importing all required packages:

```python
import os

os.environ["KERAS_BACKEND"] = "jax"  # @param ["tensorflow", "jax", "torch"]

import matplotlib.pyplot as plt

# Import tensorflow for [`tf.data`](https://www.tensorflow.org/api_docs/python/tf/data) and its preprocessing map functions
import tensorflow as tf
import tensorflow_datasets as tfds
import keras
import keras_cv
```

## Data loading

This guide uses the [102 Category Flower Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) for demonstration purposes.

To get started, we first load the dataset:

```python
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE
tfds.disable_progress_bar()
data, dataset_info = tfds.load("oxford_flowers102", with_info=True, as_supervised=True)
train_steps_per_epoch = dataset_info.splits["train"].num_examples // BATCH_SIZE
val_steps_per_epoch = dataset_info.splits["test"].num_examples // BATCH_SIZE
```

{{% details title="Result" closed="true" %}}

```plain
 Downloading and preparing dataset 328.90 MiB (download: 328.90 MiB, generated: 331.34 MiB, total: 660.25 MiB) to /usr/local/google/home/rameshsampath/tensorflow_datasets/oxford_flowers102/2.1.1...
 Dataset oxford_flowers102 downloaded and prepared to /usr/local/google/home/rameshsampath/tensorflow_datasets/oxford_flowers102/2.1.1. Subsequent calls will reuse this data.
```

{{% /details %}}

Next, we resize the images to a constant size, `(224, 224)`, and one-hot encode the labels. Please note that [`keras_cv.layers.CutMix`]({{< relref "/docs/api/keras_cv/layers/augmentation/cut_mix#cutmix-class" >}}) and [`keras_cv.layers.MixUp`]({{< relref "/docs/api/keras_cv/layers/augmentation/mix_up#mixup-class" >}}) expect targets to be one-hot encoded. This is because they modify the values of the targets in a way that is not possible with a sparse label representation.

```python
IMAGE_SIZE = (224, 224)
num_classes = dataset_info.features["label"].num_classes


def to_dict(image, label):
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32)
    label = tf.one_hot(label, num_classes)
    return {"images": image, "labels": label}


def prepare_dataset(dataset, split):
    if split == "train":
        return (
            dataset.shuffle(10 * BATCH_SIZE)
            .map(to_dict, num_parallel_calls=AUTOTUNE)
            .batch(BATCH_SIZE)
        )
    if split == "test":
        return dataset.map(to_dict, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE)


def load_dataset(split="train"):
    dataset = data[split]
    return prepare_dataset(dataset, split)


train_dataset = load_dataset()
```

Let's inspect some samples from our dataset:

```python
def visualize_dataset(dataset, title):
    plt.figure(figsize=(6, 6)).suptitle(title, fontsize=18)
    for i, samples in enumerate(iter(dataset.take(9))):
        images = samples["images"]
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[0].numpy().astype("uint8"))
        plt.axis("off")
    plt.show()


visualize_dataset(train_dataset, title="Before Augmentation")
```

![png](/images/guides/keras_cv/cut_mix_mix_up_and_rand_augment/cut_mix_mix_up_and_rand_augment_11_0.png)

Great! Now we can move onto the augmentation step.

## RandAugment

[RandAugment](https://arxiv.org/abs/1909.13719) has been shown to provide improved image classification results across numerous datasets. It performs a standard set of augmentations on an image.

To use RandAugment in KerasCV, you need to provide a few values:

- `value_range` describes the range of values covered in your images
- `magnitude` is a value between 0 and 1, describing the strength of the perturbations applied
- `augmentations_per_image` is an integer telling the layer how many augmentations to apply to each individual image
- (Optional) `magnitude_stddev` allows `magnitude` to be randomly sampled from a distribution with a standard deviation of `magnitude_stddev`
- (Optional) `rate` indicates the probability to apply the augmentation applied at each layer.

You can read more about these parameters in the [`RandAugment` API documentation]({{< relref "/docs/api/keras_cv/layers/augmentation/rand_augment" >}}).

Let's use KerasCV's RandAugment implementation.

```python
rand_augment = keras_cv.layers.RandAugment(
    value_range=(0, 255),
    augmentations_per_image=3,
    magnitude=0.3,
    magnitude_stddev=0.2,
    rate=1.0,
)


def apply_rand_augment(inputs):
    inputs["images"] = rand_augment(inputs["images"])
    return inputs


train_dataset = load_dataset().map(apply_rand_augment, num_parallel_calls=AUTOTUNE)
```

Finally, let's inspect some of the results:

```python
visualize_dataset(train_dataset, title="After RandAugment")
```

![png](/images/guides/keras_cv/cut_mix_mix_up_and_rand_augment/cut_mix_mix_up_and_rand_augment_17_0.png)

Try tweaking the magnitude settings to see a wider variety of results.

## CutMix and MixUp: generate high-quality inter-class examples

`CutMix` and `MixUp` allow us to produce inter-class examples. `CutMix` randomly cuts out portions of one image and places them over another, and `MixUp` interpolates the pixel values between two images. Both of these prevent the model from overfitting the training distribution and improve the likelihood that the model can generalize to out of distribution examples. Additionally, `CutMix` prevents your model from over-relying on any particular feature to perform its classifications. You can read more about these techniques in their respective papers:

- [CutMix: Train Strong Classifiers](https://arxiv.org/abs/1905.04899)
- [MixUp: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412)

In this example, we will use `CutMix` and `MixUp` independently in a manually created preprocessing pipeline. In most state of the art pipelines images are randomly augmented by either `CutMix`, `MixUp`, or neither. The function below implements both.

```python
cut_mix = keras_cv.layers.CutMix()
mix_up = keras_cv.layers.MixUp()


def cut_mix_and_mix_up(samples):
    samples = cut_mix(samples, training=True)
    samples = mix_up(samples, training=True)
    return samples


train_dataset = load_dataset().map(cut_mix_and_mix_up, num_parallel_calls=AUTOTUNE)

visualize_dataset(train_dataset, title="After CutMix and MixUp")
```

![png](/images/guides/keras_cv/cut_mix_mix_up_and_rand_augment/cut_mix_mix_up_and_rand_augment_20_0.png)

Great! Looks like we have successfully added `CutMix` and `MixUp` to our preprocessing pipeline.

## Customizing your augmentation pipeline

Perhaps you want to exclude an augmentation from `RandAugment`, or perhaps you want to include the [`keras_cv.layers.GridMask`]({{< relref "/docs/api/keras_cv/layers/augmentation/grid_mask#gridmask-class" >}}) as an option alongside the default `RandAugment` augmentations.

KerasCV allows you to construct production grade custom data augmentation pipelines using the [`keras_cv.layers.RandomAugmentationPipeline`]({{< relref "/docs/api/keras_cv/layers/augmentation/random_augmentation_pipeline#randomaugmentationpipeline-class" >}}) layer. This class operates similarly to `RandAugment`; selecting a random layer to apply to each image `augmentations_per_image` times. `RandAugment` can be thought of as a specific case of `RandomAugmentationPipeline`. In fact, our `RandAugment` implementation inherits from `RandomAugmentationPipeline` internally.

In this example, we will create a custom `RandomAugmentationPipeline` by removing `RandomRotation` layers from the standard `RandAugment` policy, and substitute a `GridMask` layer in its place.

As a first step, let's use the helper method `RandAugment.get_standard_policy()` to create a base pipeline.

```python
layers = keras_cv.layers.RandAugment.get_standard_policy(
    value_range=(0, 255), magnitude=0.75, magnitude_stddev=0.3
)
```

First, let's filter out `RandomRotation` layers

```python
layers = [
    layer for layer in layers if not isinstance(layer, keras_cv.layers.RandomRotation)
]
```

Next, let's add [`keras_cv.layers.GridMask`]({{< relref "/docs/api/keras_cv/layers/augmentation/grid_mask#gridmask-class" >}}) to our layers:

```python
layers = layers + [keras_cv.layers.GridMask()]
```

Finally, we can put together our pipeline

```python
pipeline = keras_cv.layers.RandomAugmentationPipeline(
    layers=layers, augmentations_per_image=3
)


def apply_pipeline(inputs):
    inputs["images"] = pipeline(inputs["images"])
    return inputs
```

Let's check out the results!

```python
train_dataset = load_dataset().map(apply_pipeline, num_parallel_calls=AUTOTUNE)
visualize_dataset(train_dataset, title="After custom pipeline")
```

![png](/images/guides/keras_cv/cut_mix_mix_up_and_rand_augment/cut_mix_mix_up_and_rand_augment_32_0.png)

Awesome! As you can see, no images were randomly rotated. You can customize the pipeline however you like:

```python
pipeline = keras_cv.layers.RandomAugmentationPipeline(
    layers=[keras_cv.layers.GridMask(), keras_cv.layers.Grayscale(output_channels=3)],
    augmentations_per_image=1,
)
```

This pipeline will either apply `GrayScale` or GridMask:

```python
train_dataset = load_dataset().map(apply_pipeline, num_parallel_calls=AUTOTUNE)
visualize_dataset(train_dataset, title="After custom pipeline")
```

![png](/images/guides/keras_cv/cut_mix_mix_up_and_rand_augment/cut_mix_mix_up_and_rand_augment_36_0.png)

Looks great! You can use `RandomAugmentationPipeline` however you want.

## Training a CNN

As a final exercise, let's take some of these layers for a spin. In this section, we will use `CutMix`, `MixUp`, and `RandAugment` to train a state of the art `ResNet50` image classifier on the Oxford flowers dataset.

```python
def preprocess_for_model(inputs):
    images, labels = inputs["images"], inputs["labels"]
    images = tf.cast(images, tf.float32)
    return images, labels


train_dataset = (
    load_dataset()
    .map(apply_rand_augment, num_parallel_calls=AUTOTUNE)
    .map(cut_mix_and_mix_up, num_parallel_calls=AUTOTUNE)
)

visualize_dataset(train_dataset, "CutMix, MixUp and RandAugment")

train_dataset = train_dataset.map(preprocess_for_model, num_parallel_calls=AUTOTUNE)

test_dataset = load_dataset(split="test")
test_dataset = test_dataset.map(preprocess_for_model, num_parallel_calls=AUTOTUNE)

train_dataset = train_dataset.prefetch(AUTOTUNE)
test_dataset = test_dataset.prefetch(AUTOTUNE)
```

![png](/images/guides/keras_cv/cut_mix_mix_up_and_rand_augment/cut_mix_mix_up_and_rand_augment_39_0.png)

Next we should create a the model itself. Notice that we use `label_smoothing=0.1` in the loss function. When using `MixUp`, label smoothing is _highly_ recommended.

```python
input_shape = IMAGE_SIZE + (3,)


def get_model():
    model = keras_cv.models.ImageClassifier.from_preset(
        "efficientnetv2_s", num_classes=num_classes
    )
    model.compile(
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        optimizer=keras.optimizers.SGD(momentum=0.9),
        metrics=["accuracy"],
    )
    return model
```

Finally we train the model:

```python
model = get_model()
model.fit(
    train_dataset,
    epochs=1,
    validation_data=test_dataset,
)
```

{{% details title="Result" closed="true" %}}

```plain
 32/32 ━━━━━━━━━━━━━━━━━━━━ 103s 2s/step - accuracy: 0.0059 - loss: 4.6941 - val_accuracy: 0.0114 - val_loss: 10.4028

<keras.src.callbacks.history.History at 0x7fd0d00e07c0>
```

{{% /details %}}

## Conclusion & next steps

That's all it takes to assemble state of the art image augmentation pipeliens with KerasCV!

As an additional exercise for readers, you can:

- Perform a hyper parameter search over the RandAugment parameters to improve the classifier accuracy
- Substitute the Oxford Flowers dataset with your own dataset
- Experiment with custom `RandomAugmentationPipeline` objects.

Currently, between Keras core and KerasCV there are [_28 image augmentation layers_]({{< relref "/docs/api/keras_cv/layers/preprocessing" >}})! Each of these can be used independently, or in a pipeline. Check them out, and if you find an augmentation techniques you need is missing please file a [GitHub issue on KerasCV](https://github.com/keras-team/keras-cv/issues).
