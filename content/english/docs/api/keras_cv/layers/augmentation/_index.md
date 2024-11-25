---
title: mix_up
toc: false
---

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/layers/preprocessing/mix_up.py#L24" >}}

### `MixUp` class

`keras_cv.layers.MixUp(alpha=0.2, seed=None, **kwargs)`

MixUp implements the MixUp data augmentation technique.

**Arguments**

- **alpha**: Float between 0 and 1. Inverse scale parameter for the gamma distribution. This controls the shape of the distribution from which the smoothing values are sampled. Defaults to 0.2, which is a recommended value when training an imagenet1k classification model.
- **seed**: Integer. Used to create a random seed.

**References**

- [MixUp paper](https://arxiv.org/abs/1710.09412). - [MixUp for Object Detection paper](https://arxiv.org/pdf/1902.04103).

**Example**

`(images, labels), _ = keras.datasets.cifar10.load_data() images, labels = images[:10], labels[:10] # Labels must be floating-point and one-hot encoded labels = tf.cast(tf.one_hot(labels, 10), tf.float32) mixup = keras_cv.layers.preprocessing.MixUp(10) augmented_images, updated_labels = mixup(     {'images': images, 'labels': labels} ) # output == {'images': updated_images, 'labels': updated_labels}`

---
