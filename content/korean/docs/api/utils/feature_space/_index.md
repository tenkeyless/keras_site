---
title: Structured data preprocessing utilities
toc: true
weight: 3
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/layers/preprocessing/feature_space.py#L72" >}}

### `FeatureSpace` class

```python
keras.utils.FeatureSpace(
    features,
    output_mode="concat",
    crosses=None,
    crossing_dim=32,
    hashing_dim=32,
    num_discretization_bins=32,
    name=None,
)
```

One-stop utility for preprocessing and encoding structured data.

**Arguments**

- **feature_names**: Dict mapping the names of your features to their
  type specification, e.g. `{"my_feature": "integer_categorical"}`
  or `{"my_feature": FeatureSpace.integer_categorical()}`.
  For a complete list of all supported types, see
  "Available feature types" paragraph below.
- **output_mode**: One of `"concat"` or `"dict"`. In concat mode, all
  features get concatenated together into a single vector.
  In dict mode, the FeatureSpace returns a dict of individually
  encoded features (with the same keys as the input dict keys).
- **crosses**: List of features to be crossed together, e.g.
  `crosses=[("feature_1", "feature_2")]`. The features will be
  "crossed" by hashing their combined value into
  a fixed-length vector.
- **crossing_dim**: Default vector size for hashing crossed features.
  Defaults to `32`.
- **hashing_dim**: Default vector size for hashing features of type
  `"integer_hashed"` and `"string_hashed"`. Defaults to `32`.
- **num_discretization_bins**: Default number of bins to be used for
  discretizing features of type `"float_discretized"`.
  Defaults to `32`.

**Available feature types:**

Note that all features can be referred to by their string name,
e.g. `"integer_categorical"`. When using the string name, the default
argument values are used.

```python
# Plain float values.
FeatureSpace.float(name=None)
# Float values to be preprocessed via featurewise standardization
# (i.e. via a [`keras.layers.Normalization`]({{< relref "/docs/api/layers/preprocessing_layers/numerical/normalization#normalization-class" >}}) layer).
FeatureSpace.float_normalized(name=None)
# Float values to be preprocessed via linear rescaling
# (i.e. via a [`keras.layers.Rescaling`]({{< relref "/docs/api/layers/preprocessing_layers/image_preprocessing/rescaling#rescaling-class" >}}) layer).
FeatureSpace.float_rescaled(scale=1., offset=0., name=None)
# Float values to be discretized. By default, the discrete
# representation will then be one-hot encoded.
FeatureSpace.float_discretized(
    num_bins, bin_boundaries=None, output_mode="one_hot", name=None)
# Integer values to be indexed. By default, the discrete
# representation will then be one-hot encoded.
FeatureSpace.integer_categorical(
    max_tokens=None, num_oov_indices=1, output_mode="one_hot", name=None)
# String values to be indexed. By default, the discrete
# representation will then be one-hot encoded.
FeatureSpace.string_categorical(
    max_tokens=None, num_oov_indices=1, output_mode="one_hot", name=None)
# Integer values to be hashed into a fixed number of bins.
# By default, the discrete representation will then be one-hot encoded.
FeatureSpace.integer_hashed(num_bins, output_mode="one_hot", name=None)
# String values to be hashed into a fixed number of bins.
# By default, the discrete representation will then be one-hot encoded.
FeatureSpace.string_hashed(num_bins, output_mode="one_hot", name=None)
```

**Examples**

**Basic usage with a dict of input data:**

```python
raw_data = {
    "float_values": [0.0, 0.1, 0.2, 0.3],
    "string_values": ["zero", "one", "two", "three"],
    "int_values": [0, 1, 2, 3],
}
dataset = tf.data.Dataset.from_tensor_slices(raw_data)
feature_space = FeatureSpace(
    features={
        "float_values": "float_normalized",
        "string_values": "string_categorical",
        "int_values": "integer_categorical",
    },
    crosses=[("string_values", "int_values")],
    output_mode="concat",
)
# Before you start using the FeatureSpace,
# you must `adapt()` it on some data.
feature_space.adapt(dataset)
# You can call the FeatureSpace on a dict of data (batched or unbatched).
output_vector = feature_space(raw_data)
```

**Basic usage with [`tf.data`](https://www.tensorflow.org/api_docs/python/tf/data):**

```python
# Unlabeled data
preprocessed_ds = unlabeled_dataset.map(feature_space)
# Labeled data
preprocessed_ds = labeled_dataset.map(lambda x, y: (feature_space(x), y))
```

**Basic usage with the Keras Functional API:**

```python
# Retrieve a dict Keras Input objects
inputs = feature_space.get_inputs()
# Retrieve the corresponding encoded Keras tensors
encoded_features = feature_space.get_encoded_features()
# Build a Functional model
outputs = keras.layers.Dense(1, activation="sigmoid")(encoded_features)
model = keras.Model(inputs, outputs)
```

**Customizing each feature or feature cross:**

```python
feature_space = FeatureSpace(
    features={
        "float_values": FeatureSpace.float_normalized(),
        "string_values": FeatureSpace.string_categorical(max_tokens=10),
        "int_values": FeatureSpace.integer_categorical(max_tokens=10),
    },
    crosses=[
        FeatureSpace.cross(("string_values", "int_values"), crossing_dim=32)
    ],
    output_mode="concat",
)
```

**Returning a dict of integer-encoded features:**

```python
feature_space = FeatureSpace(
    features={
        "string_values": FeatureSpace.string_categorical(output_mode="int"),
        "int_values": FeatureSpace.integer_categorical(output_mode="int"),
    },
    crosses=[
        FeatureSpace.cross(
            feature_names=("string_values", "int_values"),
            crossing_dim=32,
            output_mode="int",
        )
    ],
    output_mode="dict",
)
```

**Specifying your own Keras preprocessing layer:**

```python
# Let's say that one of the features is a short text paragraph that
# we want to encode as a vector (one vector per paragraph) via TF-IDF.
data = {
    "text": ["1st string", "2nd string", "3rd string"],
}
# There's a Keras layer for this: TextVectorization.
custom_layer = layers.TextVectorization(output_mode="tf_idf")
# We can use FeatureSpace.feature to create a custom feature
# that will use our preprocessing layer.
feature_space = FeatureSpace(
    features={
        "text": FeatureSpace.feature(
            preprocessor=custom_layer, dtype="string", output_mode="float"
        ),
    },
    output_mode="concat",
)
feature_space.adapt(tf.data.Dataset.from_tensor_slices(data))
output_vector = feature_space(data)
```

**Retrieving the underlying Keras preprocessing layers:**

```python
# The preprocessing layer of each feature is available in `.preprocessors`.
preprocessing_layer = feature_space.preprocessors["feature1"]
# The crossing layer of each feature cross is available in `.crossers`.
# It's an instance of keras.layers.HashedCrossing.
crossing_layer = feature_space.crossers["feature1_X_feature2"]
```

**Saving and reloading a FeatureSpace:**

```python
feature_space.save("featurespace.keras")
reloaded_feature_space = keras.models.load_model("featurespace.keras")
```
