---
title: 간단한 MNIST convnet
linkTitle: 간단한 MNIST convnet
toc: true
weight: 2
type: docs
---

{{< keras/original checkedAt="2024-11-19" >}}

**{{< t f_author >}}** [fchollet](https://twitter.com/fchollet)  
**{{< t f_date_created >}}** 2015/06/19  
**{{< t f_last_modified >}}** 2020/04/21  
**{{< t f_description >}}** MNIST에 대해 최대 99%의 테스트 정확도를 달성하는 간단한 convnet입니다.

{{< keras/version v=3 >}}

{{< cards cols="2" >}}
{{< card link="https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/vision/ipynb/mnist_convnet.ipynb" title="Colab" tag="Colab" tagType="warning">}}
{{< card link="https://github.com/keras-team/keras-io/blob/master/examples/vision/mnist_convnet.py" title="GitHub" tag="GitHub">}}
{{< /cards >}}

## 셋업 {#setup}

```python
import numpy as np
import keras
from keras import layers
```

## 데이터 준비 {#prepare-the-data}

```python
# 모델 / 데이터 파라미터
num_classes = 10
input_shape = (28, 28, 1)

# 데이터를 로드하고 트레이닝 세트와 테스트 세트로 분할합니다.
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 이미지 배율을 [0, 1] 범위로 조정하기
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# 이미지가 (28, 28, 1) 모양인지 확인하기
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")


# 클래스 벡터를 이진 클래스 행렬로 변환하기
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
x_train shape: (60000, 28, 28, 1)
60000 train samples
10000 test samples
```

{{% /details %}}

## 모델 빌드 {#build-the-model}

```python
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape              ┃    Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ conv2d (Conv2D)                 │ (None, 26, 26, 32)        │        320 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ max_pooling2d (MaxPooling2D)    │ (None, 13, 13, 32)        │          0 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ conv2d_1 (Conv2D)               │ (None, 11, 11, 64)        │     18,496 │
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

## 모델 트레이닝 {#train-the-model}

```python
batch_size = 128
epochs = 15

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Epoch 1/15
 422/422 ━━━━━━━━━━━━━━━━━━━━ 7s 9ms/step - accuracy: 0.7668 - loss: 0.7644 - val_accuracy: 0.9803 - val_loss: 0.0815
Epoch 2/15
 422/422 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - accuracy: 0.9627 - loss: 0.1237 - val_accuracy: 0.9833 - val_loss: 0.0623
Epoch 3/15
 422/422 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - accuracy: 0.9732 - loss: 0.0898 - val_accuracy: 0.9850 - val_loss: 0.0539
Epoch 4/15
 422/422 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - accuracy: 0.9761 - loss: 0.0763 - val_accuracy: 0.9880 - val_loss: 0.0421
Epoch 5/15
 422/422 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - accuracy: 0.9795 - loss: 0.0647 - val_accuracy: 0.9887 - val_loss: 0.0389
Epoch 6/15
 422/422 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - accuracy: 0.9824 - loss: 0.0580 - val_accuracy: 0.9903 - val_loss: 0.0345
Epoch 7/15
 422/422 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - accuracy: 0.9828 - loss: 0.0537 - val_accuracy: 0.9895 - val_loss: 0.0371
Epoch 8/15
 422/422 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - accuracy: 0.9838 - loss: 0.0503 - val_accuracy: 0.9907 - val_loss: 0.0340
Epoch 9/15
 422/422 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - accuracy: 0.9861 - loss: 0.0451 - val_accuracy: 0.9907 - val_loss: 0.0330
Epoch 10/15
 422/422 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - accuracy: 0.9866 - loss: 0.0427 - val_accuracy: 0.9917 - val_loss: 0.0298
Epoch 11/15
 422/422 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - accuracy: 0.9871 - loss: 0.0389 - val_accuracy: 0.9920 - val_loss: 0.0297
Epoch 12/15
 422/422 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - accuracy: 0.9885 - loss: 0.0371 - val_accuracy: 0.9912 - val_loss: 0.0285
Epoch 13/15
 422/422 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - accuracy: 0.9901 - loss: 0.0332 - val_accuracy: 0.9922 - val_loss: 0.0290
Epoch 14/15
 422/422 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - accuracy: 0.9885 - loss: 0.0340 - val_accuracy: 0.9923 - val_loss: 0.0283
Epoch 15/15
 422/422 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - accuracy: 0.9891 - loss: 0.0326 - val_accuracy: 0.9925 - val_loss: 0.0273

<keras.src.callbacks.history.History at 0x7f8497818af0>
```

{{% /details %}}

## 트레이닝된 모델 평가하기 {#evaluate-the-trained-model}

```python
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Test loss: 0.02499214932322502
Test accuracy: 0.9919000267982483
```

{{% /details %}}
