---
title: Video Classification with a CNN-RNN Architecture
toc: true
weight: 55
type: docs
---

{{< keras/original checkedAt="2024-11-21" >}}

**{{< t f_author >}}** [Sayak Paul](https://twitter.com/RisingSayak)  
**{{< t f_date_created >}}** 2021/05/28  
**{{< t f_last_modified >}}** 2023/12/08  
**{{< t f_description >}}** Training a video classifier with transfer learning and a recurrent model on the UCF101 dataset.

{{< keras/version v=3 >}}

{{< cards cols="2" >}}
{{< card link="https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/vision/ipynb/video_classification.ipynb" title="Colab" tag="Colab" tagType="warning">}}
{{< card link="https://github.com/keras-team/keras-io/blob/master/examples/vision/video_classification.py" title="GitHub" tag="GitHub">}}
{{< /cards >}}

This example demonstrates video classification, an important use-case with applications in recommendations, security, and so on. We will be using the [UCF101 dataset](https://www.crcv.ucf.edu/data/UCF101.php) to build our video classifier. The dataset consists of videos categorized into different actions, like cricket shot, punching, biking, etc. This dataset is commonly used to build action recognizers, which are an application of video classification.

A video consists of an ordered sequence of frames. Each frame contains _spatial_ information, and the sequence of those frames contains _temporal_ information. To model both of these aspects, we use a hybrid architecture that consists of convolutions (for spatial processing) as well as recurrent layers (for temporal processing). Specifically, we'll use a Convolutional Neural Network (CNN) and a Recurrent Neural Network (RNN) consisting of [GRU layers]({{< relref "/docs/api/layers/recurrent_layers/gru" >}}). This kind of hybrid architecture is popularly known as a **CNN-RNN**.

This example requires TensorFlow 2.5 or higher, as well as TensorFlow Docs, which can be installed using the following command:

```python
!pip install -q git+https://github.com/tensorflow/docs
```

## Data collection

In order to keep the runtime of this example relatively short, we will be using a subsampled version of the original UCF101 dataset. You can refer to [this notebook](https://colab.research.google.com/github/sayakpaul/Action-Recognition-in-TensorFlow/blob/main/Data_Preparation_UCF101.ipynb) to know how the subsampling was done.

```python
!!wget -q https://github.com/sayakpaul/Action-Recognition-in-TensorFlow/releases/download/v1.0.0/ucf101_top5.tar.gz
!tar xf ucf101_top5.tar.gz
```

## Setup

```python
import os

import keras
from imutils import paths

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import imageio
import cv2
from IPython.display import Image
```

## Define hyperparameters

```python
IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 10

MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048
```

## Data preparation

```python
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

print(f"Total videos for training: {len(train_df)}")
print(f"Total videos for testing: {len(test_df)}")

train_df.sample(10)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Total videos for training: 594
Total videos for testing: 224
```

{{% /details %}}

|     | video_name                 | tag          |
| --- | -------------------------- | ------------ |
| 492 | v_TennisSwing_g10_c03.avi  | TennisSwing  |
| 536 | v_TennisSwing_g16_c05.avi  | TennisSwing  |
| 413 | v_ShavingBeard_g16_c05.avi | ShavingBeard |
| 268 | v_Punch_g12_c04.avi        | Punch        |
| 288 | v_Punch_g15_c03.avi        | Punch        |
| 30  | v_CricketShot_g12_c03.avi  | CricketShot  |
| 449 | v_ShavingBeard_g21_c07.avi | ShavingBeard |
| 524 | v_TennisSwing_g14_c07.avi  | TennisSwing  |
| 145 | v_PlayingCello_g12_c01.avi | PlayingCello |
| 566 | v_TennisSwing_g21_c03.avi  | TennisSwing  |

One of the many challenges of training video classifiers is figuring out a way to feed the videos to a network. [This blog post](https://blog.coast.ai/five-video-classification-methods-implemented-in-keras-and-tensorflow-99cad29cc0b5) discusses five such methods. Since a video is an ordered sequence of frames, we could just extract the frames and put them in a 3D tensor. But the number of frames may differ from video to video which would prevent us from stacking them into batches (unless we use padding). As an alternative, we can **save video frames at a fixed interval until a maximum frame count is reached**. In this example we will do the following:

1.  Capture the frames of a video.
2.  Extract frames from the videos until a maximum frame count is reached.
3.  In the case, where a video's frame count is lesser than the maximum frame count we will pad the video with zeros.

Note that this workflow is identical to [problems involving texts sequences](https://developers.google.com/machine-learning/guides/text-classification/). Videos of the UCF101 dataset is [known](https://www.crcv.ucf.edu/papers/UCF101_CRCV-TR-12-01.pdf) to not contain extreme variations in objects and actions across frames. Because of this, it may be okay to only consider a few frames for the learning task. But this approach may not generalize well to other video classification problems. We will be using [OpenCV's `VideoCapture()` method](https://docs.opencv.org/master/dd/d43/tutorial_py_video_display.html) to read frames from videos.

```python
# The following two methods are taken from this tutorial:
# https://www.tensorflow.org/hub/tutorials/action_recognition_with_tf_hub


def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]


def load_video(path, max_frames=0, resize=(IMG_SIZE, IMG_SIZE)):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames)
```

We can use a pre-trained network to extract meaningful features from the extracted frames. The [`Keras Applications`]({{< relref "/docs/api/applications" >}}) module provides a number of state-of-the-art models pre-trained on the [ImageNet-1k dataset](http://image-net.org/). We will be using the [InceptionV3 model](https://arxiv.org/abs/1512.00567) for this purpose.

```python
def build_feature_extractor():
    feature_extractor = keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    preprocess_input = keras.applications.inception_v3.preprocess_input

    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")


feature_extractor = build_feature_extractor()
```

The labels of the videos are strings. Neural networks do not understand string values, so they must be converted to some numerical form before they are fed to the model. Here we will use the [`StringLookup`]({{< relref "/docs/api/layers/preprocessing_layers/categorical/string_lookup" >}}) layer encode the class labels as integers.

```python
label_processor = keras.layers.StringLookup(
    num_oov_indices=0, vocabulary=np.unique(train_df["tag"])
)
print(label_processor.get_vocabulary())
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
['CricketShot', 'PlayingCello', 'Punch', 'ShavingBeard', 'TennisSwing']
```

{{% /details %}}

Finally, we can put all the pieces together to create our data processing utility.

```python
def prepare_all_videos(df, root_dir):
    num_samples = len(df)
    video_paths = df["video_name"].values.tolist()
    labels = df["tag"].values
    labels = keras.ops.convert_to_numpy(label_processor(labels[..., None]))

    # `frame_masks` and `frame_features` are what we will feed to our sequence model.
    # `frame_masks` will contain a bunch of booleans denoting if a timestep is
    # masked with padding or not.
    frame_masks = np.zeros(shape=(num_samples, MAX_SEQ_LENGTH), dtype="bool")
    frame_features = np.zeros(
        shape=(num_samples, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
    )

    # For each video.
    for idx, path in enumerate(video_paths):
        # Gather all its frames and add a batch dimension.
        frames = load_video(os.path.join(root_dir, path))
        frames = frames[None, ...]

        # Initialize placeholders to store the masks and features of the current video.
        temp_frame_mask = np.zeros(
            shape=(
                1,
                MAX_SEQ_LENGTH,
            ),
            dtype="bool",
        )
        temp_frame_features = np.zeros(
            shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
        )

        # Extract features from the frames of the current video.
        for i, batch in enumerate(frames):
            video_length = batch.shape[0]
            length = min(MAX_SEQ_LENGTH, video_length)
            for j in range(length):
                temp_frame_features[i, j, :] = feature_extractor.predict(
                    batch[None, j, :], verbose=0,
                )
            temp_frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

        frame_features[idx,] = temp_frame_features.squeeze()
        frame_masks[idx,] = temp_frame_mask.squeeze()

    return (frame_features, frame_masks), labels


train_data, train_labels = prepare_all_videos(train_df, "train")
test_data, test_labels = prepare_all_videos(test_df, "test")

print(f"Frame features in train set: {train_data[0].shape}")
print(f"Frame masks in train set: {train_data[1].shape}")
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Frame features in train set: (594, 20, 2048)
Frame masks in train set: (594, 20)
```

{{% /details %}}

The above code block will take ~20 minutes to execute depending on the machine it's being executed.

## The sequence model

Now, we can feed this data to a sequence model consisting of recurrent layers like `GRU`.

```python
# Utility for our sequence model.
def get_sequence_model():
    class_vocab = label_processor.get_vocabulary()

    frame_features_input = keras.Input((MAX_SEQ_LENGTH, NUM_FEATURES))
    mask_input = keras.Input((MAX_SEQ_LENGTH,), dtype="bool")

    # Refer to the following tutorial to understand the significance of using `mask`:
    # https://keras.io/api/layers/recurrent_layers/gru/
    x = keras.layers.GRU(16, return_sequences=True)(
        frame_features_input, mask=mask_input
    )
    x = keras.layers.GRU(8)(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(8, activation="relu")(x)
    output = keras.layers.Dense(len(class_vocab), activation="softmax")(x)

    rnn_model = keras.Model([frame_features_input, mask_input], output)

    rnn_model.compile(
        loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return rnn_model


# Utility for running experiments.
def run_experiment():
    filepath = "/tmp/video_classifier/ckpt.weights.h5"
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath, save_weights_only=True, save_best_only=True, verbose=1
    )

    seq_model = get_sequence_model()
    history = seq_model.fit(
        [train_data[0], train_data[1]],
        train_labels,
        validation_split=0.3,
        epochs=EPOCHS,
        callbacks=[checkpoint],
    )

    seq_model.load_weights(filepath)
    _, accuracy = seq_model.evaluate([test_data[0], test_data[1]], test_labels)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

    return history, seq_model


_, sequence_model = run_experiment()
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Epoch 1/10
 13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - accuracy: 0.3058 - loss: 1.5597
Epoch 1: val_loss improved from inf to 1.78077, saving model to /tmp/video_classifier/ckpt.weights.h5
 13/13 ━━━━━━━━━━━━━━━━━━━━ 2s 36ms/step - accuracy: 0.3127 - loss: 1.5531 - val_accuracy: 0.1397 - val_loss: 1.7808
Epoch 2/10
 13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - accuracy: 0.5216 - loss: 1.2704
Epoch 2: val_loss improved from 1.78077 to 1.78026, saving model to /tmp/video_classifier/ckpt.weights.h5
 13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step - accuracy: 0.5226 - loss: 1.2684 - val_accuracy: 0.1788 - val_loss: 1.7803
Epoch 3/10
 13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - accuracy: 0.6189 - loss: 1.1656
Epoch 3: val_loss did not improve from 1.78026
 13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step - accuracy: 0.6174 - loss: 1.1651 - val_accuracy: 0.2849 - val_loss: 1.8322
Epoch 4/10
 13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - accuracy: 0.6518 - loss: 1.0645
Epoch 4: val_loss did not improve from 1.78026
 13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step - accuracy: 0.6515 - loss: 1.0647 - val_accuracy: 0.2793 - val_loss: 2.0419
Epoch 5/10
 13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - accuracy: 0.6833 - loss: 0.9976
Epoch 5: val_loss did not improve from 1.78026
 13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step - accuracy: 0.6843 - loss: 0.9965 - val_accuracy: 0.3073 - val_loss: 1.9077
Epoch 6/10
 13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - accuracy: 0.7229 - loss: 0.9312
Epoch 6: val_loss did not improve from 1.78026
 13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step - accuracy: 0.7241 - loss: 0.9305 - val_accuracy: 0.3017 - val_loss: 2.1513
Epoch 7/10
 13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - accuracy: 0.8023 - loss: 0.9132
Epoch 7: val_loss did not improve from 1.78026
 13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step - accuracy: 0.8035 - loss: 0.9093 - val_accuracy: 0.3184 - val_loss: 2.1705
Epoch 8/10
 13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - accuracy: 0.8127 - loss: 0.8380
Epoch 8: val_loss did not improve from 1.78026
 13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step - accuracy: 0.8128 - loss: 0.8356 - val_accuracy: 0.3296 - val_loss: 2.2043
Epoch 9/10
 13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - accuracy: 0.8494 - loss: 0.7641
Epoch 9: val_loss did not improve from 1.78026
 13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step - accuracy: 0.8494 - loss: 0.7622 - val_accuracy: 0.3017 - val_loss: 2.3734
Epoch 10/10
 13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - accuracy: 0.8634 - loss: 0.6883
Epoch 10: val_loss did not improve from 1.78026
 13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step - accuracy: 0.8649 - loss: 0.6882 - val_accuracy: 0.3240 - val_loss: 2.4410
 7/7 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.7816 - loss: 1.0624
Test accuracy: 56.7%
```

{{% /details %}}

**Note**: To keep the runtime of this example relatively short, we just used a few training examples. This number of training examples is low with respect to the sequence model being used that has 99,909 trainable parameters. You are encouraged to sample more data from the UCF101 dataset using [the notebook](https://colab.research.google.com/github/sayakpaul/Action-Recognition-in-TensorFlow/blob/main/Data_Preparation_UCF101.ipynb) mentioned above and train the same model.

## Inference

```python
def prepare_single_video(frames):
    frames = frames[None, ...]
    frame_mask = np.zeros(
        shape=(
            1,
            MAX_SEQ_LENGTH,
        ),
        dtype="bool",
    )
    frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(MAX_SEQ_LENGTH, video_length)
        for j in range(length):
            frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])
        frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

    return frame_features, frame_mask


def sequence_prediction(path):
    class_vocab = label_processor.get_vocabulary()

    frames = load_video(os.path.join("test", path))
    frame_features, frame_mask = prepare_single_video(frames)
    probabilities = sequence_model.predict([frame_features, frame_mask])[0]

    for i in np.argsort(probabilities)[::-1]:
        print(f"  {class_vocab[i]}: {probabilities[i] * 100:5.2f}%")
    return frames


# This utility is for visualization.
# Referenced from:
# https://www.tensorflow.org/hub/tutorials/action_recognition_with_tf_hub
def to_gif(images):
    converted_images = images.astype(np.uint8)
    imageio.mimsave("animation.gif", converted_images, duration=100)
    return Image("animation.gif")


test_video = np.random.choice(test_df["video_name"].values.tolist())
print(f"Test video path: {test_video}")
test_frames = sequence_prediction(test_video)
to_gif(test_frames[:MAX_SEQ_LENGTH])
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Test video path: v_TennisSwing_g03_c01.avi
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 34ms/step
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 34ms/step
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 34ms/step
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 35ms/step
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 34ms/step
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 34ms/step
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 34ms/step
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 35ms/step
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 34ms/step
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 34ms/step
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 34ms/step
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 34ms/step
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 34ms/step
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 32ms/step
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 166ms/step
  CricketShot: 46.99%
  ShavingBeard: 18.83%
  TennisSwing: 14.65%
  Punch: 12.41%
  PlayingCello:  7.12%

<IPython.core.display.Image object>
```

{{% /details %}}

## Next steps

- In this example, we made use of transfer learning for extracting meaningful features from video frames. You could also fine-tune the pre-trained network to notice how that affects the end results.
- For speed-accuracy trade-offs, you can try out other models present inside `keras.applications`.
- Try different combinations of `MAX_SEQ_LENGTH` to observe how that affects the performance.
- Train on a higher number of classes and see if you are able to get good performance.
- Following [this tutorial](https://www.tensorflow.org/hub/tutorials/action_recognition_with_tf_hub), try a [pre-trained action recognition model](https://arxiv.org/abs/1705.07750) from DeepMind.
- Rolling-averaging can be useful technique for video classification and it can be combined with a standard image classification model to infer on videos. [This tutorial](https://www.pyimagesearch.com/2019/07/15/video-classification-with-keras-and-deep-learning/) will help understand how to use rolling-averaging with an image classifier.
- When there are variations in between the frames of a video not all the frames might be equally important to decide its category. In those situations, putting a [self-attention layer](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Attention) in the sequence model will likely yield better results.
- Following [this book chapter](https://livebook.manning.com/book/deep-learning-with-python-second-edition/chapter-11), you can implement Transformers-based models for processing videos.
