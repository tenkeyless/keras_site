---
title: pipeline
toc: false
---

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/layers/preprocessing/pipeline.py#L7" >}}

### `Pipeline` class

`keras.layers.Pipeline(layers, name=None)`

Applies a series of layers to an input.

This class is useful to build a preprocessing pipeline, in particular an image data augmentation pipeline. Compared to a `Sequential` model, `Pipeline` features a few important differences:

- It's not a `Model`, just a plain layer.
- When the layers in the pipeline are compatible with [`tf.data`](https://www.tensorflow.org/api_docs/python/tf/data), the pipeline will also remain [`tf.data`](https://www.tensorflow.org/api_docs/python/tf/data) compatible. That is to say, the pipeline will not attempt to convert its inputs to backend-native tensors when in a tf.data context (unlike a `Sequential` model).

**Example**

`` from keras import layers preprocessing_pipeline = layers.Pipeline([     layers.AutoContrast(),     layers.RandomZoom(0.2),     layers.RandomRotation(0.2), ])  # `ds` is a tf.data.Dataset preprocessed_ds = ds.map(     preprocessing_pipeline,     num_parallel_calls=4, ) ``

---
