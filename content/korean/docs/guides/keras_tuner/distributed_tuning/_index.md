---
title: KerasTuner로 분산 하이퍼파라미터 튜닝
linkTitle: 분산 하이퍼파라미터 튜닝
toc: true
weight: 2
type: docs
---

{{< keras/original checkedAt="2024-11-18" >}}

**{{< t f_author >}}** Tom O'Malley, Haifeng Jin  
**{{< t f_date_created >}}** 2019/10/24  
**{{< t f_last_modified >}}** 2021/06/02  
**{{< t f_description >}}** 모델의 하이퍼파라미터를 다중 GPU 및 다중 머신에서 튜닝하기.

{{< cards cols="2" >}}
{{< card link="https://colab.research.google.com/github/keras-team/keras-io/blob/master/guides/ipynb/keras_tuner/distributed_tuning.ipynb" title="Colab" tag="Colab" tagType="warning">}}
{{< card link="https://github.com/keras-team/keras-io/blob/master/guides/keras_tuner/distributed_tuning.py" title="GitHub" tag="GitHub">}}
{{< /cards >}}

```python
!pip install keras-tuner -q
```

## 소개 {#introduction}

KerasTuner는 분산된 하이퍼파라미터 검색을 쉽게 수행할 수 있도록 해줍니다.
코드를 변경하지 않고도 로컬에서 단일 스레드로 실행하는 것에서,
수십 또는 수백 개의 작업자(worker)에서 병렬로 실행하는 것으로 확장할 수 있습니다.
분산된 KerasTuner는 chief-worker 모델을 사용합니다.
Chief는 서비스 역할을 하며,
worker들은 결과를 보고하고 다음으로 시도할 하이퍼파라미터를 요청합니다.
Chief는 단일 스레드 CPU 인스턴스에서 실행해야 하며,
대안적으로는 worker 중 하나에서 별도의 프로세스로 실행할 수 있습니다.

### 분산 모드 구성 {#configuring-distributed-mode}

KerasTuner의 분산 모드를 구성하려면, 세 가지 환경 변수를 설정하기만 하면 됩니다:

- **KERASTUNER_TUNER_ID**:
  - Chief 프로세스에는 "chief"로 설정해야 합니다.
  - 다른 worker에는 고유한 ID가 부여되어야 합니다. (관례적으로 "tuner0", "tuner1" 등)
- **KERASTUNER_ORACLE_IP**:
  - Chief 서비스가 실행될 IP 주소 또는 호스트명을 설정합니다.
  - 모든 worker는 이 주소를 resolve하고 접근할 수 있어야 합니다.
- **KERASTUNER_ORACLE_PORT**:

  - Chief 서비스가 실행될 포트를 설정합니다.
  - 이 포트는 다른 worker들이 접근할 수 있는 포트여야 하며, 자유롭게 선택할 수 있습니다.
  - 인스턴스들은 [gRPC](https://www.grpc.io) 프로토콜을 통해 통신합니다.

모든 worker에서 동일한 코드를 실행할 수 있습니다. 분산 모드와 관련된 추가 고려 사항은 다음과 같습니다:

- 모든 worker는 결과를 기록할 수 있는 중앙화된 파일 시스템에 접근할 수 있어야 합니다.
- 모든 worker는 튜닝에 필요한 트레이닝 및 검증 데이터를 사용할 수 있어야 합니다.
- 내결함성을 지원하기 위해, `overwrite`는 `Tuner.__init__`에서 `False`로 유지해야 합니다. (기본값은 `False`입니다)

Chief 서비스에 대한 예시 bash 스크립트 (`run_tuning.py`의 샘플 코드 하단 참고):

```shell
export KERASTUNER_TUNER_ID="chief"
export KERASTUNER_ORACLE_IP="127.0.0.1"
export KERASTUNER_ORACLE_PORT="8000"
python run_tuning.py
```

worker들에 대한 예시 bash 스크립트:

```shell
export KERASTUNER_TUNER_ID="tuner0"
export KERASTUNER_ORACLE_IP="127.0.0.1"
export KERASTUNER_ORACLE_PORT="8000"
python run_tuning.py
```

### [`tf.distribute`](https://www.tensorflow.org/api_docs/python/tf/distribute)로 데이터 병렬 처리 {#tfdistribute}

KerasTuner는 [tf.distribute](https://www.tensorflow.org/tutorials/distribute/keras)를 통해,
데이터 병렬 처리도 지원합니다.
데이터 병렬 처리와 분산 튜닝을 결합할 수 있습니다.
예를 들어, 4개의 GPU가 있는 10개의 worker가 있을 때,
각 worker가 4개의 GPU에서 학습하는 10개의 병렬 실험을 실행할 수 있으며,
이때 [tf.distribute.MirroredStrategy](https://www.tensorflow.org/api_docs/python/tf/distribute/MirroredStrategy)를 사용할 수 있습니다.
또한 [tf.distribute.TPUStrategy](https://www.tensorflow.org/api_docs/python/tf/distribute/experimental/TPUStrategy)를 사용하여,
각 실험을 TPU에서 실행할 수 있습니다.
현재 [tf.distribute.MultiWorkerMirroredStrategy](https://www.tensorflow.org/api_docs/python/tf/distribute/experimental/MultiWorkerMirroredStrategy)는 지원되지 않지만,
향후 지원할 예정입니다.

### 예제 코드 {#example-code}

아래 환경 변수가 설정된 상태에서, 아래 예시는 분산 튜닝을 실행하고,
각 실험에서 [`tf.distribute`](https://www.tensorflow.org/api_docs/python/tf/distribute)를 사용하여,
데이터 병렬 처리를 진행합니다.
이 예시는 `tensorflow_datasets`에서 MNIST를 로드하고,
[Hyperband](https://arxiv.org/abs/1603.06560)를 사용하여 하이퍼파라미터 검색을 수행합니다.

```python
import keras
import keras_tuner
import tensorflow as tf
import numpy as np


def build_model(hp):
    """컨볼루션 모델을 빌드합니다."""
    inputs = keras.Input(shape=(28, 28, 1))
    x = inputs
    for i in range(hp.Int("conv_layers", 1, 3, default=3)):
        x = keras.layers.Conv2D(
            filters=hp.Int("filters_" + str(i), 4, 32, step=4, default=8),
            kernel_size=hp.Int("kernel_size_" + str(i), 3, 5),
            activation="relu",
            padding="same",
        )(x)

        if hp.Choice("pooling" + str(i), ["max", "avg"]) == "max":
            x = keras.layers.MaxPooling2D()(x)
        else:
            x = keras.layers.AveragePooling2D()(x)

        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)

    if hp.Choice("global_pooling", ["max", "avg"]) == "max":
        x = keras.layers.GlobalMaxPooling2D()(x)
    else:
        x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(10, activation="softmax")(x)

    model = keras.Model(inputs, outputs)

    optimizer = hp.Choice("optimizer", ["adam", "sgd"])
    model.compile(
        optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


tuner = keras_tuner.Hyperband(
    hypermodel=build_model,
    objective="val_accuracy",
    max_epochs=2,
    factor=3,
    hyperband_iterations=1,
    distribution_strategy=tf.distribute.MirroredStrategy(),
    directory="results_dir",
    project_name="mnist",
    overwrite=True,
)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 이미지의 채널 차원을 가지도록, 이미지를 reshape 합니다.
x_train = (x_train.reshape(x_train.shape + (1,)) / 255.0)[:1000]
y_train = y_train.astype(np.int64)[:1000]
x_test = (x_test.reshape(x_test.shape + (1,)) / 255.0)[:100]
y_test = y_test.astype(np.int64)[:100]

tuner.search(
    x_train,
    y_train,
    steps_per_epoch=600,
    validation_data=(x_test, y_test),
    validation_steps=100,
    callbacks=[keras.callbacks.EarlyStopping("val_accuracy")],
)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Trial 2 Complete [00h 00m 18s]
val_accuracy: 0.07000000029802322
```

```plain
Best val_accuracy So Far: 0.07000000029802322
Total elapsed time: 00h 00m 26s
```

{{% /details %}}
