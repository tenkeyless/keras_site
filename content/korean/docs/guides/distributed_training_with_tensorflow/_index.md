---
title: TensorFlow로 멀티 GPU 분산 트레이닝하기
linkTitle: TensorFlow 분산 트레이닝
toc: true
weight: 16
type: docs
---

{{< keras/original checkedAt="2024-11-18" >}}

**{{< t f_author >}}** [fchollet](https://twitter.com/fchollet)  
**{{< t f_date_created >}}** 2020/04/28  
**{{< t f_last_modified >}}** 2023/06/29  
**{{< t f_description >}}** TensorFlow로 Keras 모델을 사용하여, 멀티 GPU 트레이닝을 진행하는 가이드.

{{< cards cols="2" >}}
{{< card link="https://colab.research.google.com/github/keras-team/keras-io/blob/master/guides/ipynb/distributed_training_with_tensorflow.ipynb" title="Colab" tag="Colab" tagType="warning">}}
{{< card link="https://github.com/keras-team/keras-io/blob/master/guides/distributed_training_with_tensorflow.py" title="GitHub" tag="GitHub">}}
{{< /cards >}}

## 소개 {#introduction}

일반적으로 여러 디바이스에 계산을 분산시키는 방법에는 두 가지가 있습니다:

- **데이터 병렬 처리**
  - **데이터 병렬 처리**에서는 하나의 모델이 여러 장치나 여러 머신에 복제됩니다.
  - 각 장치는 서로 다른 배치의 데이터를 처리한 후, 결과를 병합합니다.
  - 이 설정에는 다양한 변형이 있으며, 서로 다른 모델 복제본이 결과를 병합하는 방식이나,
    각 배치마다 동기화되는지 여부 등에 차이가 있습니다.
- **모델 병렬 처리**
  - **모델 병렬 처리**에서는 하나의 모델의 다른 부분이 서로 다른 장치에서 실행되어, 하나의 데이터 배치를 함께 처리합니다.
  - 이는 여러 가지 브랜치를 특징으로 하는 자연스럽게 병렬화된 아키텍처를 가진 모델에 가장 적합합니다.

이 가이드는 데이터 병렬 처리, 특히 **동기식 데이터 병렬 처리**에 중점을 둡니다.
여기서 모델의 서로 다른 복제본은 각 배치를 처리한 후 동기화됩니다.
동기화는 모델의 수렴 동작을 단일 장치에서의 트레이닝과 동일하게 유지시킵니다.

특히, 이 가이드는 TensorFlow의
[`tf.distribute`](https://www.tensorflow.org/api_docs/python/tf/distribute) API를 사용하여,
여러 GPU(보통 2~16개)를 사용하는 동기식 데이터 병렬 처리 방식으로 Keras 모델을 트레이닝하는 방법을 다룹니다.
최소한의 코드 수정으로 여러 GPU가 설치된 단일 머신(단일 호스트, 멀티 디바이스 트레이닝)에서 트레이닝할 수 있습니다.
이는 연구자와 소규모 산업 워크플로우에서 가장 일반적으로 사용되는 설정입니다.

## 셋업 {#setup}

```python
import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
import keras
```

## 단일 호스트, 다중 장치 동기 트레이닝 {#single-host-multi-device-synchronous-training}

이 설정에서는, 여러 개의 GPU가 있는 하나의 머신(일반적으로 2~16개의 GPU)에서 트레이닝을 진행합니다.
각 디바이스는 **복제본(replica)**이라고 불리는 모델의 사본을 실행합니다.
간단히 설명하기 위해, 다음 내용에서는 8개의 GPU를 사용하는 것으로 가정하겠습니다. 이는 일반성을 잃지 않습니다.​

**작동 방식**

트레이닝의 각 단계에서:

- 현재 데이터 배치(**글로벌 배치**)는 8개의 서로 다른 하위 배치(**로컬 배치**)로 나뉩니다.
  예를 들어, 글로벌 배치에 512개의 샘플이 있으면, 8개의 로컬 배치 각각에는 64개의 샘플이 포함됩니다.
- 8개의 복제본 각각은 로컬 배치를 독립적으로 처리합니다:
  순전파를 실행한 후 역전파를 수행하여 모델 손실에 대한 가중치의 그래디언트를 출력합니다.
- 로컬 그래디언트로부터 발생한 가중치 업데이트는 8개의 복제본 간에 효율적으로 병합됩니다.
  이 병합은 각 스텝이 끝날 때 이루어지기 때문에, 복제본은 항상 동기화된 상태를 유지합니다.

실제로, 모델 복제본의 가중치를 동기화하는 과정은 각 개별 가중치 변수 레벨에서 처리됩니다.
이는 **미러드 변수(mirrored variable)** 객체를 통해 이루어집니다.

**사용 방법**

Keras 모델로 단일 호스트, 멀티 디바이스 동기 트레이닝을 수행하려면,
[`tf.distribute.MirroredStrategy` API](https://www.tensorflow.org/api_docs/python/tf/distribute/MirroredStrategy)를 사용하면 됩니다.
작동 방식은 다음과 같습니다:

- `MirroredStrategy`를 인스턴스화하고,
  선택적으로 사용할 특정 디바이스를 구성할 수 있습니다.
  (기본적으로는 사용 가능한 모든 GPU를 사용합니다)
- strategy 객체를 사용해 scope를 열고, 이 scope 내에서 변수를 포함하는 모든 Keras 객체를 생성합니다.
  일반적으로, **모델 생성 및 컴파일**은 분산 scope 내에서 이루어져야 합니다.
  일부 경우에는, `fit()` 호출 시에도 변수가 생성될 수 있으므로,
  `fit()` 호출도 scope 내에서 이루어지도록 하는 것이 좋습니다.
- `fit()`을 통해 모델을 트레이닝합니다.

중요한 점으로, 멀티 디바이스 또는 분산 워크플로에서 데이터를 로드하려면,
[`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset) 객체를 사용하는 것을 권장합니다.

대략적인 흐름은 다음과 같습니다:

```python
# MirroredStrategy 생성
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

# strategy scope 열기
with strategy.scope():
    # 변수를 생성하는 모든 작업은 strategy scope 내에서 이루어져야 합니다.
    # 일반적으로 모델 생성 및 `compile()`입니다.
    model = Model(...)
    model.compile(...)

    # 사용 가능한 모든 디바이스에서 모델 트레이닝
    model.fit(train_dataset, validation_data=val_dataset, ...)

    # 사용 가능한 모든 디바이스에서 모델 평가
    model.evaluate(test_dataset)
```

다음은 실행 가능한 간단한 엔드투엔드 예제입니다:

```python
def get_compiled_model():
    # 간단한 2 레이어 Dense 신경망을 만듭니다.
    inputs = keras.Input(shape=(784,))
    x = keras.layers.Dense(256, activation="relu")(inputs)
    x = keras.layers.Dense(256, activation="relu")(x)
    outputs = keras.layers.Dense(10)(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    return model


def get_dataset():
    batch_size = 32
    num_val_samples = 10000

    # [`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset) 형식으로 MNIST 데이터셋을 반환합니다.
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # 데이터를 전처리합니다. (이들은 Numpy 배열입니다)
    x_train = x_train.reshape(-1, 784).astype("float32") / 255
    x_test = x_test.reshape(-1, 784).astype("float32") / 255
    y_train = y_train.astype("float32")
    y_test = y_test.astype("float32")

    # num_val_samples 샘플을 검증용으로 예약합니다.
    x_val = x_train[-num_val_samples:]
    y_val = y_train[-num_val_samples:]
    x_train = x_train[:-num_val_samples]
    y_train = y_train[:-num_val_samples]
    return (
        tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size),
        tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size),
        tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size),
    )


# MirroredStrategy를 생성합니다.
strategy = tf.distribute.MirroredStrategy()
print("Number of devices: {}".format(strategy.num_replicas_in_sync))

# strategy scope를 엽니다.
with strategy.scope():
    # 변수를 생성하는 모든 작업은 strategy scope 내에서 이루어져야 합니다.
    # 일반적으로 모델 생성 및 `compile()`입니다.
    model = get_compiled_model()

    # 사용 가능한 모든 디바이스에서 모델을 트레이닝합니다.
    train_dataset, val_dataset, test_dataset = get_dataset()
    model.fit(train_dataset, epochs=2, validation_data=val_dataset)

    # 사용 가능한 모든 디바이스에서 모델을 테스트합니다.
    model.evaluate(test_dataset)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
INFO:tensorflow:Using MirroredStrategy with devices ('/job:localhost/replica:0/task:0/device:CPU:0',)
Number of devices: 1
Epoch 1/2
 1563/1563 ━━━━━━━━━━━━━━━━━━━━ 7s 4ms/step - loss: 0.3830 - sparse_categorical_accuracy: 0.8884 - val_loss: 0.1361 - val_sparse_categorical_accuracy: 0.9574
Epoch 2/2
 1563/1563 ━━━━━━━━━━━━━━━━━━━━ 9s 3ms/step - loss: 0.1068 - sparse_categorical_accuracy: 0.9671 - val_loss: 0.0894 - val_sparse_categorical_accuracy: 0.9724
 313/313 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - loss: 0.0988 - sparse_categorical_accuracy: 0.9673
```

{{% /details %}}

## 콜백을 사용하여 장애 복원력(fault tolerance) 보장하기 {#using-callbacks-to-ensure-fault-tolerance}

분산 트레이닝을 사용할 때, 항상 장애 복원력을 위한 전략을 세워야 합니다.
가장 간단한 방법은 `ModelCheckpoint` 콜백을 `fit()`에 전달하여,
일정 간격마다 모델을 저장하는 것입니다. (예: 매 100 배치마다 또는 매 에포크마다)
이렇게 하면, 저장된 모델에서 트레이닝을 재시작할 수 있습니다.

다음은 간단한 예시입니다:

```python
# 체크포인트를 저장할 디렉토리를 준비합니다.
checkpoint_dir = "./ckpt"
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)


def make_or_restore_model():
    # 체크포인트가 있으면 최신 모델을 복원하고,
    # 체크포인트가 없으면 새로운 모델을 생성합니다.
    checkpoints = [checkpoint_dir + "/" + name for name in os.listdir(checkpoint_dir)]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print("Restoring from", latest_checkpoint)
        return keras.models.load_model(latest_checkpoint)
    print("Creating a new model")
    return get_compiled_model()


def run_training(epochs=1):
    # MirroredStrategy를 생성합니다.
    strategy = tf.distribute.MirroredStrategy()

    # strategy scope를 열고 모델을 생성하거나 복원합니다.
    with strategy.scope():
        model = make_or_restore_model()

        callbacks = [
            # 이 콜백은 매 에포크마다 SavedModel을 저장합니다.
            # 현재 에포크를 폴더 이름에 포함시킵니다.
            keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_dir + "/ckpt-{epoch}.keras",
                save_freq="epoch",
            )
        ]
        model.fit(
            train_dataset,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=val_dataset,
            verbose=2,
        )


# 처음 실행 시 모델을 생성합니다.
run_training(epochs=1)

# 같은 함수를 다시 호출하면 이전 상태에서 재개합니다.
run_training(epochs=1)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
INFO:tensorflow:Using MirroredStrategy with devices ('/job:localhost/replica:0/task:0/device:CPU:0',)
Creating a new model
1563/1563 - 7s - 4ms/step - loss: 0.2275 - sparse_categorical_accuracy: 0.9320 - val_loss: 0.1373 - val_sparse_categorical_accuracy: 0.9571
INFO:tensorflow:Using MirroredStrategy with devices ('/job:localhost/replica:0/task:0/device:CPU:0',)
Restoring from ./ckpt/ckpt-1.keras
1563/1563 - 6s - 4ms/step - loss: 0.0944 - sparse_categorical_accuracy: 0.9717 - val_loss: 0.0972 - val_sparse_categorical_accuracy: 0.9710
```

{{% /details %}}

## [`tf.data`](https://www.tensorflow.org/api_docs/python/tf/data) 성능 팁 {#tfdata}

분산 트레이닝을 수행할 때, 데이터를 로드하는 효율성이 매우 중요해질 수 있습니다.
[`tf.data`](https://www.tensorflow.org/api_docs/python/tf/data)
파이프라인을 가능한 한 빠르게 실행하는 몇 가지 팁을 소개합니다.

**데이터셋 배치에 대한 주의사항**

데이터셋을 생성할 때, 글로벌 배치 크기로 배치되었는지 확인하세요.
예를 들어, 8개의 GPU 각각이 64개의 샘플로 구성된 배치를 실행할 수 있는 경우,
글로벌 배치 크기는 512로 설정합니다.

**`dataset.cache()` 호출**

데이터셋에서 `.cache()`를 호출하면, 첫 번째 반복 이후 데이터가 캐시됩니다.
이후 모든 반복에서는 캐시된 데이터를 사용하게 됩니다.
캐시는 기본적으로 메모리에 저장되며, 또는 사용자가 지정한 로컬 파일에 저장할 수 있습니다.

이 방법은 다음과 같은 경우, 성능을 향상시킬 수 있습니다:

- 데이터가 반복마다 변경되지 않는 경우
- 데이터를 원격 분산 파일 시스템에서 읽어오는 경우
- 데이터를 로컬 디스크에서 읽어오고, 데이터가 메모리에 fit하며,
  워크플로우가 주로 IO 바운드인 경우 (예: 이미지 파일 읽기 및 디코딩)

**`dataset.prefetch(buffer_size)` 호출**

데이터셋을 생성한 후에는 거의 항상 `.prefetch(buffer_size)`를 호출하는 것이 좋습니다.
이 방법은 데이터 파이프라인이 모델과 비동기적으로 실행되도록 하여,
현재 배치가 트레이닝되는 동안 다음 배치의 샘플을 미리 처리하고 버퍼에 저장합니다.
현재 배치가 완료될 때쯤이면 다음 배치가 GPU 메모리로 미리 로드됩니다.

이것이 전부입니다!
