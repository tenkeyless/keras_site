---
title: 당신의 커스텀 트레이닝 루프에서 하이퍼파라미터 튜닝
linkTitle: 커스텀 트레이닝 하이퍼파라미터 튜닝
toc: true
weight: 3
type: docs
---

{{< keras/original checkedAt="2024-11-18" >}}

**{{< t f_author >}}** Tom O'Malley, Haifeng Jin  
**{{< t f_date_created >}}** 2019/10/28  
**{{< t f_last_modified >}}** 2022/01/12  
**{{< t f_description >}}** `HyperModel.fit()`을 사용하여 트레이닝 하이퍼파라미터(예: 배치 크기)를 튜닝합니다.

{{< cards cols="2" >}}
{{< card link="https://colab.research.google.com/github/keras-team/keras-io/blob/master/guides/ipynb/keras_tuner/custom_tuner.ipynb" title="Colab" tag="Colab" tagType="warning">}}
{{< card link="https://github.com/keras-team/keras-io/blob/master/guides/keras_tuner/custom_tuner.py" title="GitHub" tag="GitHub">}}
{{< /cards >}}

```python
!pip install keras-tuner -q
```

## 소개 {#introduction}

KerasTuner의 `HyperModel` 클래스는 재사용 가능한 객체에서 검색 공간을 정의하는 편리한 방법을 제공합니다.
`HyperModel.build()`를 재정의하여 모델 자체를 정의하고 하이퍼튜닝할 수 있습니다.
트레이닝 과정에서 하이퍼튜닝을 하려면(예: 적절한 배치 크기, 트레이닝 에포크 수 또는 데이터 보강 설정을 선택하여),
`HyperModel.fit()`을 재정의하여 다음에 접근할 수 있습니다.

- `hp` 객체: [`keras_tuner.HyperParameters`]({{< relref "/docs/api/keras_tuner/hyperparameters#hyperparameters-class" >}})의 인스턴스
- `HyperModel.build()`에 의해 빌드된 모델

기본 예시는 [KerasTuner 시작하기의 "모델 트레이닝 튜닝하기" 섹션]({{< relref "/docs/guides/keras_tuner/getting_started/#tune-model-training" >}})에서 확인할 수 있습니다.

## 커스텀 트레이닝 루프 튜닝 {#tuning-the-custom-training-loop}

이 가이드에서는, `HyperModel` 클래스를 서브클래싱하고 `HyperModel.fit()`을 재정의하여,
커스텀 트레이닝 루프를 작성합니다.
Keras에서 커스텀 트레이닝 루프를 작성하는 방법은,
{{< titledRelref "/docs/guides/writing_a_custom_training_loop_in_tensorflow" >}} 가이드를 참조하십시오.

먼저 필요한 라이브러리를 import 하고, 트레이닝 및 검증용 데이터셋을 생성합니다.
여기서는 시연 목적으로 랜덤 데이터를 사용합니다.

```python
import keras_tuner
import tensorflow as tf
import keras
import numpy as np


x_train = np.random.rand(1000, 28, 28, 1)
y_train = np.random.randint(0, 10, (1000, 1))
x_val = np.random.rand(1000, 28, 28, 1)
y_val = np.random.randint(0, 10, (1000, 1))
```

그런다음, `MyHyperModel`로 `HyperModel` 클래스를 서브클래싱합니다.
`MyHyperModel.build()`에서는 10개의 다른 클래스를 분류하기 위한 간단한 Keras 모델을 빌드합니다.
`MyHyperModel.fit()`은 여러 인수를 수락합니다. 그 시그니처는 아래와 같습니다.

```python
def fit(self, hp, model, x, y, validation_data, callbacks=None, **kwargs):
```

- `hp` 인수는 하이퍼파라미터를 정의하는 데 사용됩니다.
- `model` 인수는 `MyHyperModel.build()`에서 반환된 모델입니다.
- `x`, `y`, 및 `validation_data`는 모두 커스텀 정의된 인수입니다.
  나중에 `tuner.search(x=x, y=y, validation_data=(x_val, y_val))`를 호출하여,
  데이터를 이들에게 전달할 것입니다.
  원하는 만큼 인수를 정의하고, 커스텀 이름을 부여할 수 있습니다.
- `callbacks` 인수는 `model.fit()`과 함께 사용되도록 의도되었습니다.
  KerasTuner는 체크포인팅(모델의 최상의 에포크에서 모델 저장)과 같은 유용한 Keras 콜백을 제공합니다.

우리는 커스텀 트레이닝 루프에서 콜백을 수동으로 호출할 것입니다.
콜백을 호출하기 전에, 다음 코드를 통해 모델을 할당해야 체크포인팅을 위해 콜백에서 모델에 액세스할 수 있습니다.

```python
for callback in callbacks:
    callback.model = model
```

이 예에서는, 모델을 체크포인팅하기 위해 콜백의 `on_epoch_end()` 메서드만 호출했습니다.
필요에 따라, 다른 콜백 메서드도 호출할 수 있습니다.
모델을 저장할 필요가 없다면, 콜백을 사용할 필요는 없습니다.

커스텀 트레이닝 루프에서는, `tf.data.Dataset`으로 NumPy 데이터를 래핑할 때 데이터셋의 배치 크기를 튜닝합니다.
이 단계에서 다른 전처리 단계를 튜닝할 수도 있습니다.
또한 옵티마이저의 학습률도 튜닝합니다.

우리는 검증 손실을 모델 평가 지표로 사용할 것입니다.
배치마다 검증 손실을 평균화하기 위해, `keras.metrics.Mean()`을 사용합니다.
튜너가 기록을 남기기 위해 검증 손실 값을 반환해야 합니다.

```python
class MyHyperModel(keras_tuner.HyperModel):
    def build(self, hp):
        """컨볼루션 모델을 빌드합니다."""
        inputs = keras.Input(shape=(28, 28, 1))
        x = keras.layers.Flatten()(inputs)
        x = keras.layers.Dense(
            units=hp.Choice("units", [32, 64, 128]), activation="relu"
        )(x)
        outputs = keras.layers.Dense(10)(x)
        return keras.Model(inputs=inputs, outputs=outputs)

    def fit(self, hp, model, x, y, validation_data, callbacks=None, **kwargs):
        # 데이터셋을 tf.data.Dataset으로 변환합니다.
        batch_size = hp.Int("batch_size", 32, 128, step=32, default=64)
        train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(
            batch_size
        )
        validation_data = tf.data.Dataset.from_tensor_slices(validation_data).batch(
            batch_size
        )

        # 옵티마이저 정의.
        optimizer = keras.optimizers.Adam(
            hp.Float("learning_rate", 1e-4, 1e-2, sampling="log", default=1e-3)
        )
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        # 검증 손실을 추적할 메트릭.
        epoch_loss_metric = keras.metrics.Mean()

        # 트레이닝 단계를 실행하는 함수.
        @tf.function
        def run_train_step(images, labels):
            with tf.GradientTape() as tape:
                logits = model(images)
                loss = loss_fn(labels, logits)
                # 정규화 손실 추가.
                if model.losses:
                    loss += tf.math.add_n(model.losses)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # 검증 단계를 실행하는 함수.
        @tf.function
        def run_val_step(images, labels):
            logits = model(images)
            loss = loss_fn(labels, logits)
            # 메트릭 업데이트.
            epoch_loss_metric.update_state(loss)

        # 모델을 콜백에 할당합니다.
        for callback in callbacks:
            callback.set_model(model)

        # 최상의 검증 손실 값을 기록.
        best_epoch_loss = float("inf")

        # 커스텀 트레이닝 루프.
        for epoch in range(2):
            print(f"Epoch: {epoch}")

            # 트레이닝 데이터를 반복하면서, 트레이닝 단계를 실행.
            for images, labels in train_ds:
                run_train_step(images, labels)

            # 검증 데이터를 반복하면서, 검증 단계를 실행.
            for images, labels in validation_data:
                run_val_step(images, labels)

            # 에포크가 끝난 후 콜백 호출.
            epoch_loss = float(epoch_loss_metric.result().numpy())
            for callback in callbacks:
                # "my_metric"은 튜너에 전달된 objective 입니다.
                callback.on_epoch_end(epoch, logs={"my_metric": epoch_loss})
            epoch_loss_metric.reset_state()

            print(f"Epoch loss: {epoch_loss}")
            best_epoch_loss = min(best_epoch_loss, epoch_loss)

        # 평가 메트릭 값을 반환.
        return best_epoch_loss
```

이제 튜너를 초기화할 수 있습니다.
여기에서는, 최소화할 메트릭으로 `Objective("my_metric", "min")`을 사용합니다.
목표 이름은 콜백의 `on_epoch_end()` 메서드에 전달된 `logs`에서 사용하는 키와 일치해야 합니다.
콜백은 `logs`의 이 값을 사용하여 최상의 에포크를 찾아 모델을 체크포인팅합니다.

```python
tuner = keras_tuner.RandomSearch(
    objective=keras_tuner.Objective("my_metric", "min"),
    max_trials=2,
    hypermodel=MyHyperModel(),
    directory="results",
    project_name="custom_training",
    overwrite=True,
)
```

`MyHyperModel.fit()`의 서명에서 정의한 인수를 `tuner.search()`에 전달하여 검색을 시작합니다.

```python
tuner.search(x=x_train, y=y_train, validation_data=(x_val, y_val))
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Trial 2 Complete [00h 00m 02s]
my_metric: 2.3025283813476562
```

```plain
Best my_metric So Far: 2.3025283813476562
Total elapsed time: 00h 00m 04s
```

{{% /details %}}

마지막으로, 결과를 검색할 수 있습니다.

```python
best_hps = tuner.get_best_hyperparameters()[0]
print(best_hps.values)

best_model = tuner.get_best_models()[0]
best_model.summary()
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
{'units': 128, 'batch_size': 32, 'learning_rate': 0.0034272591820215972}
```

```plain
Model: "functional_1"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape              ┃    Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ input_layer (InputLayer)        │ (None, 28, 28, 1)         │          0 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ flatten (Flatten)               │ (None, 784)               │          0 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ dense (Dense)                   │ (None, 128)               │    100,480 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ dense_1 (Dense)                 │ (None, 10)                │      1,290 │
└─────────────────────────────────┴───────────────────────────┴────────────┘
 Total params: 101,770 (397.54 KB)
 Trainable params: 101,770 (397.54 KB)
 Non-trainable params: 0 (0.00 B)
```

{{% /details %}}

요약하자면, 커스텀 트레이닝 루프에서 하이퍼파라미터를 튜닝하려면,
`HyperModel.fit()`을 오버라이드하여 모델을 트레이닝하고 평가 결과를 반환하면 됩니다.
제공된 콜백을 사용하면, 최상의 에포크에서 트레이닝된 모델을 쉽게 저장하고,
나중에 최상의 모델을 로드할 수 있습니다.

KerasTuner의 기본 사항에 대해 자세히 알아보려면,
[KerasTuner로 시작하기]({{< relref "/docs/guides/keras_tuner/getting_started" >}})를 참조하세요.
