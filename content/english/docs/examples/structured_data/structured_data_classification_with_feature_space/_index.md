---
title: Structured data classification with FeatureSpace
toc: true
weight: 1
type: docs
---

{{< keras/original checkedAt="2024-11-22" >}}

**{{< t f_author >}}** [fchollet](https://twitter.com/fchollet)  
**{{< t f_date_created >}}** 2022/11/09  
**{{< t f_last_modified >}}** 2022/11/09  
**{{< t f_description >}}** Classify tabular data in a few lines of code.

{{< keras/version v=3 >}}

{{< cards cols="2" >}}
{{< card link="https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/structured_data/ipynb/structured_data_classification_with_feature_space.ipynb" title="Colab" tag="Colab" tagType="warning">}}
{{< card link="https://github.com/keras-team/keras-io/blob/master/examples/structured_data/structured_data_classification_with_feature_space.py" title="GitHub" tag="GitHub">}}
{{< /cards >}}

## Introduction

This example demonstrates how to do structured data classification (also known as tabular data classification), starting from a raw CSV file. Our data includes numerical features, and integer categorical features, and string categorical features. We will use the utility [`keras.utils.FeatureSpace`]({{< relref "/docs/api/utils/feature_space#featurespace-class" >}}) to index, preprocess, and encode our features.

The code is adapted from the example [Structured data classification from scratch]({{< relref "/docs/examples/structured_data/structured_data_classification_from_scratch" >}}). While the previous example managed its own low-level feature preprocessing and encoding with Keras preprocessing layers, in this example we delegate everything to `FeatureSpace`, making the workflow extremely quick and easy.

### The dataset

[Our dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease) is provided by the Cleveland Clinic Foundation for Heart Disease. It's a CSV file with 303 rows. Each row contains information about a patient (a **sample**), and each column describes an attribute of the patient (a **feature**). We use the features to predict whether a patient has a heart disease (**binary classification**).

Here's the description of each feature:

| Column   | Description                                            | Feature Type                 |
| -------- | ------------------------------------------------------ | ---------------------------- |
| Age      | Age in years                                           | Numerical                    |
| Sex      | (1 = male; 0 = female)                                 | Categorical                  |
| CP       | Chest pain type (0, 1, 2, 3, 4)                        | Categorical                  |
| Trestbpd | Resting blood pressure (in mm Hg on admission)         | Numerical                    |
| Chol     | Serum cholesterol in mg/dl                             | Numerical                    |
| FBS      | fasting blood sugar in 120 mg/dl (1 = true; 0 = false) | Categorical                  |
| RestECG  | Resting electrocardiogram results (0, 1, 2)            | Categorical                  |
| Thalach  | Maximum heart rate achieved                            | Numerical                    |
| Exang    | Exercise induced angina (1 = yes; 0 = no)              | Categorical                  |
| Oldpeak  | ST depression induced by exercise relative to rest     | Numerical                    |
| Slope    | Slope of the peak exercise ST segment                  | Numerical                    |
| CA       | Number of major vessels (0-3) colored by fluoroscopy   | Both numerical & categorical |
| Thal     | 3 = normal; 6 = fixed defect; 7 = reversible defect    | Categorical                  |
| Target   | Diagnosis of heart disease (1 = true; 0 = false)       | Target                       |

## Setup

```python
import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
import pandas as pd
import keras
from keras.utils import FeatureSpace
```

## Preparing the data

Let's download the data and load it into a Pandas dataframe:

```python
file_url = "http://storage.googleapis.com/download.tensorflow.org/data/heart.csv"
dataframe = pd.read_csv(file_url)
```

The dataset includes 303 samples with 14 columns per sample (13 features, plus the target label):

```python
print(dataframe.shape)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
(303, 14)
```

{{% /details %}}

Here's a preview of a few samples:

```python
dataframe.head()
```

|     | age | sex | cp  | trestbps | chol | fbs | restecg | thalach | exang | oldpeak | slope | ca  | thal       | target |
| --- | --- | --- | --- | -------- | ---- | --- | ------- | ------- | ----- | ------- | ----- | --- | ---------- | ------ |
| 0   | 63  | 1   | 1   | 145      | 233  | 1   | 2       | 150     | 0     | 2.3     | 3     | 0   | fixed      | 0      |
| 1   | 67  | 1   | 4   | 160      | 286  | 0   | 2       | 108     | 1     | 1.5     | 2     | 3   | normal     | 1      |
| 2   | 67  | 1   | 4   | 120      | 229  | 0   | 2       | 129     | 1     | 2.6     | 2     | 2   | reversible | 0      |
| 3   | 37  | 1   | 3   | 130      | 250  | 0   | 0       | 187     | 0     | 3.5     | 3     | 0   | normal     | 0      |
| 4   | 41  | 0   | 2   | 130      | 204  | 0   | 2       | 172     | 0     | 1.4     | 1     | 0   | normal     | 0      |

The last column, "target", indicates whether the patient has a heart disease (1) or not (0).

Let's split the data into a training and validation set:

```python
val_dataframe = dataframe.sample(frac=0.2, random_state=1337)
train_dataframe = dataframe.drop(val_dataframe.index)

print(
    "Using %d samples for training and %d for validation"
    % (len(train_dataframe), len(val_dataframe))
)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Using 242 samples for training and 61 for validation
```

{{% /details %}}

Let's generate [`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset) objects for each dataframe:

```python
def dataframe_to_dataset(dataframe):
    dataframe = dataframe.copy()
    labels = dataframe.pop("target")
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    ds = ds.shuffle(buffer_size=len(dataframe))
    return ds


train_ds = dataframe_to_dataset(train_dataframe)
val_ds = dataframe_to_dataset(val_dataframe)
```

Each `Dataset` yields a tuple `(input, target)` where `input` is a dictionary of features and `target` is the value `0` or `1`:

```python
for x, y in train_ds.take(1):
    print("Input:", x)
    print("Target:", y)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Input: {'age': <tf.Tensor: shape=(), dtype=int64, numpy=65>, 'sex': <tf.Tensor: shape=(), dtype=int64, numpy=1>, 'cp': <tf.Tensor: shape=(), dtype=int64, numpy=1>, 'trestbps': <tf.Tensor: shape=(), dtype=int64, numpy=138>, 'chol': <tf.Tensor: shape=(), dtype=int64, numpy=282>, 'fbs': <tf.Tensor: shape=(), dtype=int64, numpy=1>, 'restecg': <tf.Tensor: shape=(), dtype=int64, numpy=2>, 'thalach': <tf.Tensor: shape=(), dtype=int64, numpy=174>, 'exang': <tf.Tensor: shape=(), dtype=int64, numpy=0>, 'oldpeak': <tf.Tensor: shape=(), dtype=float64, numpy=1.4>, 'slope': <tf.Tensor: shape=(), dtype=int64, numpy=2>, 'ca': <tf.Tensor: shape=(), dtype=int64, numpy=1>, 'thal': <tf.Tensor: shape=(), dtype=string, numpy=b'normal'>}
Target: tf.Tensor(0, shape=(), dtype=int64)
```

{{% /details %}}

Let's batch the datasets:

```python
train_ds = train_ds.batch(32)
val_ds = val_ds.batch(32)
```

## Configuring a `FeatureSpace`

To configure how each feature should be preprocessed, we instantiate a [`keras.utils.FeatureSpace`]({{< relref "/docs/api/utils/feature_space#featurespace-class" >}}), and we pass to it a dictionary that maps the name of our features to a string that describes the feature type.

We have a few "integer categorical" features such as `"FBS"`, one "string categorical" feature (`"thal"`), and a few numerical features, which we'd like to normalize – except `"age"`, which we'd like to discretize into a number of bins.

We also use the `crosses` argument to capture _feature interactions_ for some categorical features, that is to say, create additional features that represent value co-occurrences for these categorical features. You can compute feature crosses like this for arbitrary sets of categorical features – not just tuples of two features. Because the resulting co-occurences are hashed into a fixed-sized vector, you don't need to worry about whether the co-occurence space is too large.

```python
feature_space = FeatureSpace(
    features={
        # Categorical features encoded as integers
        "sex": "integer_categorical",
        "cp": "integer_categorical",
        "fbs": "integer_categorical",
        "restecg": "integer_categorical",
        "exang": "integer_categorical",
        "ca": "integer_categorical",
        # Categorical feature encoded as string
        "thal": "string_categorical",
        # Numerical features to discretize
        "age": "float_discretized",
        # Numerical features to normalize
        "trestbps": "float_normalized",
        "chol": "float_normalized",
        "thalach": "float_normalized",
        "oldpeak": "float_normalized",
        "slope": "float_normalized",
    },
    # We create additional features by hashing
    # value co-occurrences for the
    # following groups of categorical features.
    crosses=[("sex", "age"), ("thal", "ca")],
    # The hashing space for these co-occurrences
    # wil be 32-dimensional.
    crossing_dim=32,
    # Our utility will one-hot encode all categorical
    # features and concat all features into a single
    # vector (one vector per sample).
    output_mode="concat",
)
```

## Further customizing a `FeatureSpace`

Specifying the feature type via a string name is quick and easy, but sometimes you may want to further configure the preprocessing of each feature. For instance, in our case, our categorical features don't have a large set of possible values – it's only a handful of values per feature (e.g. `1` and `0` for the feature `"FBS"`), and all possible values are represented in the training set. As a result, we don't need to reserve an index to represent "out of vocabulary" values for these features – which would have been the default behavior. Below, we just specify `num_oov_indices=0` in each of these features to tell the feature preprocessor to skip "out of vocabulary" indexing.

Other customizations you have access to include specifying the number of bins for discretizing features of type `"float_discretized"`, or the dimensionality of the hashing space for feature crossing.

```python
feature_space = FeatureSpace(
    features={
        # Categorical features encoded as integers
        "sex": FeatureSpace.integer_categorical(num_oov_indices=0),
        "cp": FeatureSpace.integer_categorical(num_oov_indices=0),
        "fbs": FeatureSpace.integer_categorical(num_oov_indices=0),
        "restecg": FeatureSpace.integer_categorical(num_oov_indices=0),
        "exang": FeatureSpace.integer_categorical(num_oov_indices=0),
        "ca": FeatureSpace.integer_categorical(num_oov_indices=0),
        # Categorical feature encoded as string
        "thal": FeatureSpace.string_categorical(num_oov_indices=0),
        # Numerical features to discretize
        "age": FeatureSpace.float_discretized(num_bins=30),
        # Numerical features to normalize
        "trestbps": FeatureSpace.float_normalized(),
        "chol": FeatureSpace.float_normalized(),
        "thalach": FeatureSpace.float_normalized(),
        "oldpeak": FeatureSpace.float_normalized(),
        "slope": FeatureSpace.float_normalized(),
    },
    # Specify feature cross with a custom crossing dim.
    crosses=[
        FeatureSpace.cross(feature_names=("sex", "age"), crossing_dim=64),
        FeatureSpace.cross(
            feature_names=("thal", "ca"),
            crossing_dim=16,
        ),
    ],
    output_mode="concat",
)
```

## Adapt the `FeatureSpace` to the training data

Before we start using the `FeatureSpace` to build a model, we have to adapt it to the training data. During `adapt()`, the `FeatureSpace` will:

- Index the set of possible values for categorical features.
- Compute the mean and variance for numerical features to normalize.
- Compute the value boundaries for the different bins for numerical features to discretize.

Note that `adapt()` should be called on a [`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset) which yields dicts of feature values – no labels.

```python
train_ds_with_no_labels = train_ds.map(lambda x, _: x)
feature_space.adapt(train_ds_with_no_labels)
```

At this point, the `FeatureSpace` can be called on a dict of raw feature values, and will return a single concatenate vector for each sample, combining encoded features and feature crosses.

```python
for x, _ in train_ds.take(1):
    preprocessed_x = feature_space(x)
    print("preprocessed_x.shape:", preprocessed_x.shape)
    print("preprocessed_x.dtype:", preprocessed_x.dtype)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
preprocessed_x.shape: (32, 138)
preprocessed_x.dtype: <dtype: 'float32'>
```

{{% /details %}}

## Two ways to manage preprocessing: as part of the [`tf.data`](https://www.tensorflow.org/api_docs/python/tf/data) pipeline, or in the model itself

There are two ways in which you can leverage your `FeatureSpace`:

### Asynchronous preprocessing in [`tf.data`](https://www.tensorflow.org/api_docs/python/tf/data)

You can make it part of your data pipeline, before the model. This enables asynchronous parallel preprocessing of the data on CPU before it hits the model. Do this if you're training on GPU or TPU, or if you want to speed up preprocessing. Usually, this is always the right thing to do during training.

### Synchronous preprocessing in the model

You can make it part of your model. This means that the model will expect dicts of raw feature values, and the preprocessing batch will be done synchronously (in a blocking manner) before the rest of the forward pass. Do this if you want to have an end-to-end model that can process raw feature values – but keep in mind that your model will only be able to run on CPU, since most types of feature preprocessing (e.g. string preprocessing) are not GPU or TPU compatible.

Do not do this on GPU / TPU or in performance-sensitive settings. In general, you want to do in-model preprocessing when you do inference on CPU.

In our case, we will apply the `FeatureSpace` in the tf.data pipeline during training, but we will do inference with an end-to-end model that includes the `FeatureSpace`.

Let's create a training and validation dataset of preprocessed batches:

```python
preprocessed_train_ds = train_ds.map(
    lambda x, y: (feature_space(x), y), num_parallel_calls=tf.data.AUTOTUNE
)
preprocessed_train_ds = preprocessed_train_ds.prefetch(tf.data.AUTOTUNE)

preprocessed_val_ds = val_ds.map(
    lambda x, y: (feature_space(x), y), num_parallel_calls=tf.data.AUTOTUNE
)
preprocessed_val_ds = preprocessed_val_ds.prefetch(tf.data.AUTOTUNE)
```

## Build a model

Time to build a model – or rather two models:

- A training model that expects preprocessed features (one sample = one vector)
- An inference model that expects raw features (one sample = dict of raw feature values)

```python
dict_inputs = feature_space.get_inputs()
encoded_features = feature_space.get_encoded_features()

x = keras.layers.Dense(32, activation="relu")(encoded_features)
x = keras.layers.Dropout(0.5)(x)
predictions = keras.layers.Dense(1, activation="sigmoid")(x)

training_model = keras.Model(inputs=encoded_features, outputs=predictions)
training_model.compile(
    optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
)

inference_model = keras.Model(inputs=dict_inputs, outputs=predictions)
```

## Train the model

Let's train our model for 50 epochs. Note that feature preprocessing is happening as part of the tf.data pipeline, not as part of the model.

```python
training_model.fit(
    preprocessed_train_ds,
    epochs=20,
    validation_data=preprocessed_val_ds,
    verbose=2,
)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Epoch 1/20
8/8 - 3s - 352ms/step - accuracy: 0.5200 - loss: 0.7407 - val_accuracy: 0.6196 - val_loss: 0.6663
Epoch 2/20
8/8 - 0s - 20ms/step - accuracy: 0.5881 - loss: 0.6874 - val_accuracy: 0.7732 - val_loss: 0.6015
Epoch 3/20
8/8 - 0s - 19ms/step - accuracy: 0.6580 - loss: 0.6192 - val_accuracy: 0.7839 - val_loss: 0.5577
Epoch 4/20
8/8 - 0s - 19ms/step - accuracy: 0.7096 - loss: 0.5721 - val_accuracy: 0.7856 - val_loss: 0.5200
Epoch 5/20
8/8 - 0s - 18ms/step - accuracy: 0.7292 - loss: 0.5553 - val_accuracy: 0.7764 - val_loss: 0.4853
Epoch 6/20
8/8 - 0s - 19ms/step - accuracy: 0.7561 - loss: 0.5103 - val_accuracy: 0.7732 - val_loss: 0.4627
Epoch 7/20
8/8 - 0s - 19ms/step - accuracy: 0.7231 - loss: 0.5374 - val_accuracy: 0.7764 - val_loss: 0.4413
Epoch 8/20
8/8 - 0s - 19ms/step - accuracy: 0.7769 - loss: 0.4564 - val_accuracy: 0.7683 - val_loss: 0.4320
Epoch 9/20
8/8 - 0s - 18ms/step - accuracy: 0.7769 - loss: 0.4324 - val_accuracy: 0.7856 - val_loss: 0.4191
Epoch 10/20
8/8 - 0s - 19ms/step - accuracy: 0.7778 - loss: 0.4340 - val_accuracy: 0.7888 - val_loss: 0.4084
Epoch 11/20
8/8 - 0s - 19ms/step - accuracy: 0.7760 - loss: 0.4124 - val_accuracy: 0.7716 - val_loss: 0.3977
Epoch 12/20
8/8 - 0s - 19ms/step - accuracy: 0.7964 - loss: 0.4125 - val_accuracy: 0.7667 - val_loss: 0.3959
Epoch 13/20
8/8 - 0s - 18ms/step - accuracy: 0.8051 - loss: 0.3979 - val_accuracy: 0.7856 - val_loss: 0.3891
Epoch 14/20
8/8 - 0s - 19ms/step - accuracy: 0.8043 - loss: 0.3891 - val_accuracy: 0.7856 - val_loss: 0.3840
Epoch 15/20
8/8 - 0s - 18ms/step - accuracy: 0.8633 - loss: 0.3571 - val_accuracy: 0.7872 - val_loss: 0.3764
Epoch 16/20
8/8 - 0s - 19ms/step - accuracy: 0.8728 - loss: 0.3548 - val_accuracy: 0.7888 - val_loss: 0.3699
Epoch 17/20
8/8 - 0s - 19ms/step - accuracy: 0.8698 - loss: 0.3171 - val_accuracy: 0.7872 - val_loss: 0.3727
Epoch 18/20
8/8 - 0s - 18ms/step - accuracy: 0.8529 - loss: 0.3454 - val_accuracy: 0.7904 - val_loss: 0.3669
Epoch 19/20
8/8 - 0s - 17ms/step - accuracy: 0.8589 - loss: 0.3359 - val_accuracy: 0.7980 - val_loss: 0.3770
Epoch 20/20
8/8 - 0s - 17ms/step - accuracy: 0.8455 - loss: 0.3113 - val_accuracy: 0.8044 - val_loss: 0.3684

<keras.src.callbacks.history.History at 0x7f139bb4ed10>
```

{{% /details %}}

We quickly get to 80% validation accuracy.

## Inference on new data with the end-to-end model

Now, we can use our inference model (which includes the `FeatureSpace`) to make predictions based on dicts of raw features values, as follows:

```python
sample = {
    "age": 60,
    "sex": 1,
    "cp": 1,
    "trestbps": 145,
    "chol": 233,
    "fbs": 1,
    "restecg": 2,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 3,
    "ca": 0,
    "thal": "fixed",
}

input_dict = {name: tf.convert_to_tensor([value]) for name, value in sample.items()}
predictions = inference_model.predict(input_dict)

print(
    f"This particular patient had a {100 * predictions[0][0]:.2f}% probability "
    "of having a heart disease, as evaluated by our model."
)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 273ms/step
This particular patient had a 43.13% probability of having a heart disease, as evaluated by our model.
```

{{% /details %}}
