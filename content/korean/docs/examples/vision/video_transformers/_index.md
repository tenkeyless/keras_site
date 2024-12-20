---
title: 트랜스포머를 사용한 비디오 분류
linkTitle: 트랜스포머 비디오 분류
toc: true
weight: 57
type: docs
---

{{< keras/original checkedAt="2024-11-21" >}}

**{{< t f_author >}}** [Sayak Paul](https://twitter.com/RisingSayak)  
**{{< t f_date_created >}}** 2021/08/06  
**{{< t f_last_modified >}}** 2023/07/22  
**{{< t f_description >}}** Training a video classifier with hybrid transformers.

{{< keras/version v=3 >}}

{{< cards cols="2" >}}
{{< card link="https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/vision/ipynb/video_transformers.ipynb" title="Colab" tag="Colab" tagType="warning">}}
{{< card link="https://github.com/keras-team/keras-io/blob/master/examples/vision/video_transformers.py" title="GitHub" tag="GitHub">}}
{{< /cards >}}

This example is a follow-up to the [Video Classification with a CNN-RNN Architecture](https://keras.io/examples/vision/video_classification/) example. This time, we will be using a Transformer-based model ([Vaswani et al.](https://arxiv.org/abs/1706.03762)) to classify videos. You can follow [this book chapter](https://livebook.manning.com/book/deep-learning-with-python-second-edition/chapter-11) in case you need an introduction to Transformers (with code). After reading this example, you will know how to develop hybrid Transformer-based models for video classification that operate on CNN feature maps.

```python
!pip install -q git+https://github.com/tensorflow/docs
```

## Data collection {#data-collection}

As done in the [predecessor]({{< relref "/docs/examples/vision/video_classification" >}}) to this example, we will be using a subsampled version of the [UCF101 dataset](https://www.crcv.ucf.edu/data/UCF101.php), a well-known benchmark dataset. In case you want to operate on a larger subsample or even the entire dataset, please refer to [this notebook](https://colab.research.google.com/github/sayakpaul/Action-Recognition-in-TensorFlow/blob/main/Data_Preparation_UCF101.ipynb).

```python
!wget -q https://github.com/sayakpaul/Action-Recognition-in-TensorFlow/releases/download/v1.0.0/ucf101_top5.tar.gz
!tar -xf ucf101_top5.tar.gz
```

## Setup {#setup}

```python
import os
import keras
from keras import layers
from keras.applications.densenet import DenseNet121

from tensorflow_docs.vis import embed

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import imageio
import cv2
```

## Define hyperparameters {#define-hyperparameters}

```python
MAX_SEQ_LENGTH = 20
NUM_FEATURES = 1024
IMG_SIZE = 128

EPOCHS = 5
```

## Data preparation {#data-preparation}

We will mostly be following the same data preparation steps in this example, except for the following changes:

- We reduce the image size to 128x128 instead of 224x224 to speed up computation.
- Instead of using a pre-trained [InceptionV3](https://arxiv.org/abs/1512.00567) network, we use a pre-trained [DenseNet121](http://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.pdf) for feature extraction.
- We directly pad shorter videos to length `MAX_SEQ_LENGTH`.

First, let's load up the [DataFrames](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html).

```python
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

print(f"Total videos for training: {len(train_df)}")
print(f"Total videos for testing: {len(test_df)}")

center_crop_layer = layers.CenterCrop(IMG_SIZE, IMG_SIZE)


def crop_center(frame):
    cropped = center_crop_layer(frame[None, ...])
    cropped = keras.ops.convert_to_numpy(cropped)
    cropped = keras.ops.squeeze(cropped)
    return cropped


# Following method is modified from this tutorial:
# https://www.tensorflow.org/hub/tutorials/action_recognition_with_tf_hub
def load_video(path, max_frames=0, offload_to_cpu=False):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = frame[:, :, [2, 1, 0]]
            frame = crop_center(frame)
            if offload_to_cpu and keras.backend.backend() == "torch":
                frame = frame.to("cpu")
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    if offload_to_cpu and keras.backend.backend() == "torch":
        return np.array([frame.to("cpu").numpy() for frame in frames])
    return np.array(frames)


def build_feature_extractor():
    feature_extractor = DenseNet121(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    preprocess_input = keras.applications.densenet.preprocess_input

    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")


feature_extractor = build_feature_extractor()


# Label preprocessing with StringLookup.
label_processor = keras.layers.StringLookup(
    num_oov_indices=0, vocabulary=np.unique(train_df["tag"]), mask_token=None
)
print(label_processor.get_vocabulary())


def prepare_all_videos(df, root_dir):
    num_samples = len(df)
    video_paths = df["video_name"].values.tolist()
    labels = df["tag"].values
    labels = label_processor(labels[..., None]).numpy()

    # `frame_features` are what we will feed to our sequence model.
    frame_features = np.zeros(
        shape=(num_samples, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
    )

    # For each video.
    for idx, path in enumerate(video_paths):
        # Gather all its frames and add a batch dimension.
        frames = load_video(os.path.join(root_dir, path))

        # Pad shorter videos.
        if len(frames) < MAX_SEQ_LENGTH:
            diff = MAX_SEQ_LENGTH - len(frames)
            padding = np.zeros((diff, IMG_SIZE, IMG_SIZE, 3))
            frames = np.concatenate(frames, padding)

        frames = frames[None, ...]

        # Initialize placeholder to store the features of the current video.
        temp_frame_features = np.zeros(
            shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
        )

        # Extract features from the frames of the current video.
        for i, batch in enumerate(frames):
            video_length = batch.shape[0]
            length = min(MAX_SEQ_LENGTH, video_length)
            for j in range(length):
                if np.mean(batch[j, :]) > 0.0:
                    temp_frame_features[i, j, :] = feature_extractor.predict(
                        batch[None, j, :]
                    )

                else:
                    temp_frame_features[i, j, :] = 0.0

        frame_features[idx,] = temp_frame_features.squeeze()

    return frame_features, labels
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Total videos for training: 594
Total videos for testing: 224
['CricketShot', 'PlayingCello', 'Punch', 'ShavingBeard', 'TennisSwing']
```

{{% /details %}}

Calling `prepare_all_videos()` on `train_df` and `test_df` takes ~20 minutes to complete. For this reason, to save time, here we download already preprocessed NumPy arrays:

```python
!!wget -q https://git.io/JZmf4 -O top5_data_prepared.tar.gz
!!tar -xf top5_data_prepared.tar.gz
```

```python
train_data, train_labels = np.load("train_data.npy"), np.load("train_labels.npy")
test_data, test_labels = np.load("test_data.npy"), np.load("test_labels.npy")

print(f"Frame features in train set: {train_data.shape}")
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
[]

Frame features in train set: (594, 20, 1024)
```

{{% /details %}}

## Building the Transformer-based model {#building-the-transformer-based-model}

We will be building on top of the code shared in [this book chapter](https://livebook.manning.com/book/deep-learning-with-python-second-edition/chapter-11) of [Deep Learning with Python (Second ed.)](https://www.manning.com/books/deep-learning-with-python) by François Chollet.

First, self-attention layers that form the basic blocks of a Transformer are order-agnostic. Since videos are ordered sequences of frames, we need our Transformer model to take into account order information. We do this via **positional encoding**. We simply embed the positions of the frames present inside videos with an [`Embedding` layer]({{< relref "/docs/api/layers/core_layers/embedding" >}}). We then add these positional embeddings to the precomputed CNN feature maps.

```python
class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=output_dim
        )
        self.sequence_length = sequence_length
        self.output_dim = output_dim

    def build(self, input_shape):
        self.position_embeddings.build(input_shape)

    def call(self, inputs):
        # The inputs are of shape: `(batch_size, frames, num_features)`
        inputs = keras.ops.cast(inputs, self.compute_dtype)
        length = keras.ops.shape(inputs)[1]
        positions = keras.ops.arange(start=0, stop=length, step=1)
        embedded_positions = self.position_embeddings(positions)
        return inputs + embedded_positions
```

Now, we can create a subclassed layer for the Transformer.

```python
class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.3
        )
        self.dense_proj = keras.Sequential(
            [
                layers.Dense(dense_dim, activation=keras.activations.gelu),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs, mask=None):
        attention_output = self.attention(inputs, inputs, attention_mask=mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)
```

## Utility functions for training {#utility-functions-for-training}

```python
def get_compiled_model(shape):
    sequence_length = MAX_SEQ_LENGTH
    embed_dim = NUM_FEATURES
    dense_dim = 4
    num_heads = 1
    classes = len(label_processor.get_vocabulary())

    inputs = keras.Input(shape=shape)
    x = PositionalEmbedding(
        sequence_length, embed_dim, name="frame_position_embedding"
    )(inputs)
    x = TransformerEncoder(embed_dim, dense_dim, num_heads, name="transformer_layer")(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def run_experiment():
    filepath = "/tmp/video_classifier.weights.h5"
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath, save_weights_only=True, save_best_only=True, verbose=1
    )

    model = get_compiled_model(train_data.shape[1:])
    history = model.fit(
        train_data,
        train_labels,
        validation_split=0.15,
        epochs=EPOCHS,
        callbacks=[checkpoint],
    )

    model.load_weights(filepath)
    _, accuracy = model.evaluate(test_data, test_labels)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

    return model
```

## Model training and inference {#model-training-and-inference}

```python
trained_model = run_experiment()
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Epoch 1/5
 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 160ms/step - accuracy: 0.5286 - loss: 2.6762
Epoch 1: val_loss improved from inf to 7.75026, saving model to /tmp/video_classifier.weights.h5
 16/16 ━━━━━━━━━━━━━━━━━━━━ 7s 272ms/step - accuracy: 0.5387 - loss: 2.6139 - val_accuracy: 0.0000e+00 - val_loss: 7.7503
Epoch 2/5
 15/16 ━━━━━━━━━━━━━━━━━━[37m━━  0s 4ms/step - accuracy: 0.9396 - loss: 0.2264
Epoch 2: val_loss improved from 7.75026 to 1.96635, saving model to /tmp/video_classifier.weights.h5
 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step - accuracy: 0.9406 - loss: 0.2186 - val_accuracy: 0.4000 - val_loss: 1.9664
Epoch 3/5
 14/16 ━━━━━━━━━━━━━━━━━[37m━━━  0s 4ms/step - accuracy: 0.9823 - loss: 0.0384
Epoch 3: val_loss did not improve from 1.96635
 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.9822 - loss: 0.0391 - val_accuracy: 0.3667 - val_loss: 3.7076
Epoch 4/5
 15/16 ━━━━━━━━━━━━━━━━━━[37m━━  0s 4ms/step - accuracy: 0.9825 - loss: 0.0681
Epoch 4: val_loss did not improve from 1.96635
 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.9831 - loss: 0.0674 - val_accuracy: 0.4222 - val_loss: 3.7957
Epoch 5/5
 15/16 ━━━━━━━━━━━━━━━━━━[37m━━  0s 4ms/step - accuracy: 1.0000 - loss: 0.0035
Epoch 5: val_loss improved from 1.96635 to 1.56071, saving model to /tmp/video_classifier.weights.h5
 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step - accuracy: 1.0000 - loss: 0.0033 - val_accuracy: 0.6333 - val_loss: 1.5607
 7/7 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.9286 - loss: 0.4434
Test accuracy: 89.29%
```

{{% /details %}}

**Note**: This model has ~4.23 Million parameters, which is way more than the sequence model (99918 parameters) we used in the prequel of this example. This kind of Transformer model works best with a larger dataset and a longer pre-training schedule.

```python
def prepare_single_video(frames):
    frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

    # Pad shorter videos.
    if len(frames) < MAX_SEQ_LENGTH:
        diff = MAX_SEQ_LENGTH - len(frames)
        padding = np.zeros((diff, IMG_SIZE, IMG_SIZE, 3))
        frames = np.concatenate(frames, padding)

    frames = frames[None, ...]

    # Extract features from the frames of the current video.
    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(MAX_SEQ_LENGTH, video_length)
        for j in range(length):
            if np.mean(batch[j, :]) > 0.0:
                frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])
            else:
                frame_features[i, j, :] = 0.0

    return frame_features


def predict_action(path):
    class_vocab = label_processor.get_vocabulary()

    frames = load_video(os.path.join("test", path), offload_to_cpu=True)
    frame_features = prepare_single_video(frames)
    probabilities = trained_model.predict(frame_features)[0]

    plot_x_axis, plot_y_axis = [], []

    for i in np.argsort(probabilities)[::-1]:
        plot_x_axis.append(class_vocab[i])
        plot_y_axis.append(probabilities[i])
        print(f"  {class_vocab[i]}: {probabilities[i] * 100:5.2f}%")

    plt.bar(plot_x_axis, plot_y_axis, label=plot_x_axis)
    plt.xlabel("class_label")
    plt.xlabel("Probability")
    plt.show()

    return frames


# This utility is for visualization.
# Referenced from:
# https://www.tensorflow.org/hub/tutorials/action_recognition_with_tf_hub
def to_gif(images):
    converted_images = images.astype(np.uint8)
    imageio.mimsave("animation.gif", converted_images, fps=10)
    return embed.embed_file("animation.gif")


test_video = np.random.choice(test_df["video_name"].values.tolist())
print(f"Test video path: {test_video}")
test_frames = predict_action(test_video)
to_gif(test_frames[:MAX_SEQ_LENGTH])
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Test video path: v_ShavingBeard_g03_c02.avi
 1/1 ━━━━━━━━━━━━━━━━━━━━ 20s 20s/step
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 11ms/step
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step
 1/1 ━━━━━━━━━━━━━━━━━━━━ 1s 557ms/step
  ShavingBeard: 100.00%
  Punch:  0.00%
  CricketShot:  0.00%
  TennisSwing:  0.00%
  PlayingCello:  0.00%
```

{{% /details %}}

![png](/images/examples/vision/video_transformers/video_transformers_23_1.png)

![gif](/images/examples/vision/video_transformers/shaving.gif)

The performance of our model is far from optimal, because it was trained on a small dataset.
