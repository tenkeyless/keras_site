---
title: 당신만의 커스텀 콜백 작성하기
linkTitle: 커스텀 콜백 작성
toc: true
weight: 13
type: docs
---

{{< keras/original checkedAt="2024-11-18" >}}

**{{< t f_author >}}** Rick Chao, Francois Chollet  
**{{< t f_date_created >}}** 2019/03/20  
**{{< t f_last_modified >}}** 2023/06/25  
**{{< t f_description >}}** Keras에서 새로운 콜백을 작성하는 완벽 가이드.

{{< cards cols="2" >}}
{{< card link="https://colab.research.google.com/github/keras-team/keras-io/blob/master/guides/ipynb/writing_your_own_callbacks.ipynb" title="Colab" tag="Colab" tagType="warning">}}
{{< card link="https://github.com/keras-team/keras-io/blob/master/guides/writing_your_own_callbacks.py" title="GitHub" tag="GitHub">}}
{{< /cards >}}

## 소개 {#introduction}

콜백은 Keras 모델의 트레이닝, 평가, 또는 추론 중에 동작을 커스터마이즈할 수 있는 강력한 도구입니다.
예를 들어, [`keras.callbacks.TensorBoard`]({{< relref "/docs/api/callbacks/tensorboard#tensorboard-class" >}})는 TensorBoard로 트레이닝 진행 상황과 결과를 시각화하고,
[`keras.callbacks.ModelCheckpoint`]({{< relref "/docs/api/callbacks/model_checkpoint#modelcheckpoint-class" >}})는 트레이닝 중 주기적으로 모델을 저장합니다.

이 가이드에서는 Keras 콜백이 무엇인지, 무엇을 할 수 있는지,
그리고 어떻게 직접 콜백을 작성할 수 있는지 배우게 됩니다.
간단한 콜백 애플리케이션의 예시를 제공하여 시작할 수 있도록 도와드립니다.

## 셋업 {#setup}

```python
import numpy as np
import keras
```

## Keras 콜백 개요 {#keras-callbacks-overview}

모든 콜백은 [`keras.callbacks.Callback`]({{< relref "/docs/api/callbacks/base_callback#callback-class" >}}) 클래스를 서브클래싱하며,
트레이닝, 테스트, 예측의 다양한 단계에서 호출되는 일련의 메서드를 오버라이드합니다.
콜백은 트레이닝 중에 모델의 내부 상태와 통계를 확인하는데 유용합니다.

다음 모델 메서드에 `callbacks`라는 키워드 인자로 콜백 리스트를 전달할 수 있습니다:

- `keras.Model.fit()`
- `keras.Model.evaluate()`
- `keras.Model.predict()`

## 콜백 메서드 개요 {#an-overview-of-callback-methods}

### Global 메서드 {#global-methods}

#### `on_(train|test|predict)_begin(self, logs=None)` {#on_traintestpredict_beginself-logsnone}

`fit`/`evaluate`/`predict` 시작 시 호출됩니다.

#### `on_(train|test|predict)_end(self, logs=None)` {#on_traintestpredict_endself-logsnone}

`fit`/`evaluate`/`predict` 종료 시 호출됩니다.

### 트레이닝/테스트/예측을 위한 배치 레벨 메서드 {#batch-level-methods-for-trainingtestingpredicting}

#### `on_(train|test|predict)_batch_begin(self, batch, logs=None)` {#on_traintestpredict_batch_beginself-batch-logsnone}

트레이닝/테스트/예측 중 배치 처리를 시작하기 직전에 호출됩니다.

#### `on_(train|test|predict)_batch_end(self, batch, logs=None)` {#on_traintestpredict_batch_endself-batch-logsnone}

배치 트레이닝/테스트/예측이 완료된 후 호출됩니다.
이 메서드 내에서 `logs`는 메트릭 결과를 포함하는 딕셔너리입니다.

### 에포크 레벨 메서드 (트레이닝 전용) {#epoch-level-methods-training-only}

#### `on_epoch_begin(self, epoch, logs=None)` {#on_epoch_beginself-epoch-logsnone}

트레이닝 중 에포크가 시작될 때 호출됩니다.

#### `on_epoch_end(self, epoch, logs=None)` {#on_epoch_endself-epoch-logsnone}

트레이닝 중 에포크가 끝날 때 호출됩니다.

## 기본 예제 {#a-basic-example}

구체적인 예시를 살펴보겠습니다.
먼저, TensorFlow를 임포트하고 간단한 Sequential Keras 모델을 정의해봅시다:

```python
# 콜백을 추가할 Keras 모델 정의
def get_model():
    model = keras.Sequential()
    model.add(keras.layers.Dense(1))
    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=0.1),
        loss="mean_squared_error",
        metrics=["mean_absolute_error"],
    )
    return model
```

그런 다음, Keras 데이터셋 API를 사용하여 트레이닝 및 테스트용 MNIST 데이터를 로드합니다:

```python
# 예시 MNIST 데이터를 로드하고 전처리합니다.
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
x_test = x_test.reshape(-1, 784).astype("float32") / 255.0

# 데이터를 1000개의 샘플로 제한합니다.
x_train = x_train[:1000]
y_train = y_train[:1000]
x_test = x_test[:1000]
y_test = y_test[:1000]
```

이제, 다음을 로그하는 간단한 커스텀 콜백을 정의해봅시다:

- `fit`/`evaluate`/`predict`가 시작하고 끝날 때
- 각 에포크가 시작하고 끝날 때
- 각 트레이닝 배치가 시작하고 끝날 때
- 각 평가(테스트) 배치가 시작하고 끝날 때
- 각 추론(예측) 배치가 시작하고 끝날 때

```python
class CustomCallback(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        keys = list(logs.keys())
        print("Starting training; got log keys: {}".format(keys))

    def on_train_end(self, logs=None):
        keys = list(logs.keys())
        print("Stop training; got log keys: {}".format(keys))

    def on_epoch_begin(self, epoch, logs=None):
        keys = list(logs.keys())
        print("Start epoch {} of training; got log keys: {}".format(epoch, keys))

    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        print("End epoch {} of training; got log keys: {}".format(epoch, keys))

    def on_test_begin(self, logs=None):
        keys = list(logs.keys())
        print("Start testing; got log keys: {}".format(keys))

    def on_test_end(self, logs=None):
        keys = list(logs.keys())
        print("Stop testing; got log keys: {}".format(keys))

    def on_predict_begin(self, logs=None):
        keys = list(logs.keys())
        print("Start predicting; got log keys: {}".format(keys))

    def on_predict_end(self, logs=None):
        keys = list(logs.keys())
        print("Stop predicting; got log keys: {}".format(keys))

    def on_train_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Training: start of batch {}; got log keys: {}".format(batch, keys))

    def on_train_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Training: end of batch {}; got log keys: {}".format(batch, keys))

    def on_test_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Evaluating: start of batch {}; got log keys: {}".format(batch, keys))

    def on_test_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Evaluating: end of batch {}; got log keys: {}".format(batch, keys))

    def on_predict_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Predicting: start of batch {}; got log keys: {}".format(batch, keys))

    def on_predict_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Predicting: end of batch {}; got log keys: {}".format(batch, keys))
```

한번 실행해봅시다:

```python
model = get_model()
model.fit(
    x_train,
    y_train,
    batch_size=128,
    epochs=1,
    verbose=0,
    validation_split=0.5,
    callbacks=[CustomCallback()],
)

res = model.evaluate(
    x_test, y_test, batch_size=128, verbose=0, callbacks=[CustomCallback()]
)

res = model.predict(x_test, batch_size=128, callbacks=[CustomCallback()])
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Starting training; got log keys: []
Start epoch 0 of training; got log keys: []
...Training: start of batch 0; got log keys: []
...Training: end of batch 0; got log keys: ['loss', 'mean_absolute_error']
...Training: start of batch 1; got log keys: []
...Training: end of batch 1; got log keys: ['loss', 'mean_absolute_error']
...Training: start of batch 2; got log keys: []
...Training: end of batch 2; got log keys: ['loss', 'mean_absolute_error']
...Training: start of batch 3; got log keys: []
...Training: end of batch 3; got log keys: ['loss', 'mean_absolute_error']
Start testing; got log keys: []
...Evaluating: start of batch 0; got log keys: []
...Evaluating: end of batch 0; got log keys: ['loss', 'mean_absolute_error']
...Evaluating: start of batch 1; got log keys: []
...Evaluating: end of batch 1; got log keys: ['loss', 'mean_absolute_error']
...Evaluating: start of batch 2; got log keys: []
...Evaluating: end of batch 2; got log keys: ['loss', 'mean_absolute_error']
...Evaluating: start of batch 3; got log keys: []
...Evaluating: end of batch 3; got log keys: ['loss', 'mean_absolute_error']
Stop testing; got log keys: ['loss', 'mean_absolute_error']
End epoch 0 of training; got log keys: ['loss', 'mean_absolute_error', 'val_loss', 'val_mean_absolute_error']
Stop training; got log keys: ['loss', 'mean_absolute_error', 'val_loss', 'val_mean_absolute_error']
Start testing; got log keys: []
...Evaluating: start of batch 0; got log keys: []
...Evaluating: end of batch 0; got log keys: ['loss', 'mean_absolute_error']
...Evaluating: start of batch 1; got log keys: []
...Evaluating: end of batch 1; got log keys: ['loss', 'mean_absolute_error']
...Evaluating: start of batch 2; got log keys: []
...Evaluating: end of batch 2; got log keys: ['loss', 'mean_absolute_error']
...Evaluating: start of batch 3; got log keys: []
...Evaluating: end of batch 3; got log keys: ['loss', 'mean_absolute_error']
...Evaluating: start of batch 4; got log keys: []
...Evaluating: end of batch 4; got log keys: ['loss', 'mean_absolute_error']
...Evaluating: start of batch 5; got log keys: []
...Evaluating: end of batch 5; got log keys: ['loss', 'mean_absolute_error']
...Evaluating: start of batch 6; got log keys: []
...Evaluating: end of batch 6; got log keys: ['loss', 'mean_absolute_error']
...Evaluating: start of batch 7; got log keys: []
...Evaluating: end of batch 7; got log keys: ['loss', 'mean_absolute_error']
Stop testing; got log keys: ['loss', 'mean_absolute_error']
Start predicting; got log keys: []
...Predicting: start of batch 0; got log keys: []
...Predicting: end of batch 0; got log keys: ['outputs']
 1/8 ━━[37m━━━━━━━━━━━━━━━━━━  0s 13ms/step...Predicting: start of batch 1; got log keys: []
...Predicting: end of batch 1; got log keys: ['outputs']
...Predicting: start of batch 2; got log keys: []
...Predicting: end of batch 2; got log keys: ['outputs']
...Predicting: start of batch 3; got log keys: []
...Predicting: end of batch 3; got log keys: ['outputs']
...Predicting: start of batch 4; got log keys: []
...Predicting: end of batch 4; got log keys: ['outputs']
...Predicting: start of batch 5; got log keys: []
...Predicting: end of batch 5; got log keys: ['outputs']
...Predicting: start of batch 6; got log keys: []
...Predicting: end of batch 6; got log keys: ['outputs']
...Predicting: start of batch 7; got log keys: []
...Predicting: end of batch 7; got log keys: ['outputs']
Stop predicting; got log keys: []
 8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step
```

{{% /details %}}

### `logs` 딕셔너리의 사용 {#usage-of-logs-dict}

`logs` 딕셔너리는 배치나 에포크가 끝날 때 손실 값과 모든 메트릭을 포함합니다.
예시로는 손실 값과 평균 절대 오차(mean absolute error)가 있습니다.

```python
class LossAndErrorPrintingCallback(keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        print(
            "Up to batch {}, the average loss is {:7.2f}.".format(batch, logs["loss"])
        )

    def on_test_batch_end(self, batch, logs=None):
        print(
            "Up to batch {}, the average loss is {:7.2f}.".format(batch, logs["loss"])
        )

    def on_epoch_end(self, epoch, logs=None):
        print(
            "The average loss for epoch {} is {:7.2f} "
            "and mean absolute error is {:7.2f}.".format(
                epoch, logs["loss"], logs["mean_absolute_error"]
            )
        )


model = get_model()
model.fit(
    x_train,
    y_train,
    batch_size=128,
    epochs=2,
    verbose=0,
    callbacks=[LossAndErrorPrintingCallback()],
)

res = model.evaluate(
    x_test,
    y_test,
    batch_size=128,
    verbose=0,
    callbacks=[LossAndErrorPrintingCallback()],
)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Up to batch 0, the average loss is   29.25.
Up to batch 1, the average loss is  485.36.
Up to batch 2, the average loss is  330.94.
Up to batch 3, the average loss is  250.62.
Up to batch 4, the average loss is  202.20.
Up to batch 5, the average loss is  169.51.
Up to batch 6, the average loss is  145.98.
Up to batch 7, the average loss is  128.48.
The average loss for epoch 0 is  128.48 and mean absolute error is    6.01.
Up to batch 0, the average loss is    5.10.
Up to batch 1, the average loss is    4.80.
Up to batch 2, the average loss is    4.96.
Up to batch 3, the average loss is    4.96.
Up to batch 4, the average loss is    4.82.
Up to batch 5, the average loss is    4.69.
Up to batch 6, the average loss is    4.51.
Up to batch 7, the average loss is    4.53.
The average loss for epoch 1 is    4.53 and mean absolute error is    1.72.
Up to batch 0, the average loss is    5.08.
Up to batch 1, the average loss is    4.66.
Up to batch 2, the average loss is    4.64.
Up to batch 3, the average loss is    4.72.
Up to batch 4, the average loss is    4.82.
Up to batch 5, the average loss is    4.83.
Up to batch 6, the average loss is    4.77.
Up to batch 7, the average loss is    4.72.
```

{{% /details %}}

## `self.model` 속성의 사용 {#usage-of-selfmodel-attribute}

메서드가 호출될 때 로그 정보를 받는 것 외에도,
콜백은 현재 트레이닝/평가/추론 라운드와 연결된 모델인 `self.model`에 접근할 수 있습니다.

콜백에서 `self.model`을 사용하여 할 수 있는 몇 가지 예는 다음과 같습니다:

- `self.model.stop_training = True`를 설정하여, 트레이닝을 즉시 중단할 수 있습니다.
- 옵티마이저(`self.model.optimizer`로 사용가능)의 하이퍼파라미터를 변경할 수 있습니다. (예: `self.model.optimizer.learning_rate`)
- 주기적으로 모델을 저장할 수 있습니다.
- 에포크가 끝날 때 몇 가지 테스트 샘플에 대해 `model.predict()`의 출력을 기록하여, 트레이닝 중 검증(sanity check)할 수 있습니다.
- 에포크가 끝날 때 중간 특성의 시각화를 추출하여, 시간에 걸쳐 모델이 학습하는 내용을 모니터링할 수 있습니다.
- 기타 등등.

몇 가지 예시를 통해 이를 실제로 확인해보겠습니다.

## Keras 콜백 애플리케이션 예시 {#examples-of-keras-callback-applications}

### 최소 손실에서의 조기 종료 {#early-stopping-at-minimum-loss}

첫 번째 예시는 손실의 최소값에 도달하면 트레이닝을 중지하는 `Callback`을 생성하는 방법을 보여줍니다.
이때 `self.model.stop_training` (boolean) 속성을 설정합니다.
선택적으로, `patience`라는 인자를 제공하여 로컬 최소값에 도달한 후,
몇 에포크 동안 대기한 뒤 트레이닝을 중지할지 지정할 수 있습니다.

[`keras.callbacks.EarlyStopping`]({{< relref "/docs/api/callbacks/early_stopping#earlystopping-class" >}})는 더 완전하고 일반적인 구현을 제공합니다.

```python
class EarlyStoppingAtMinLoss(keras.callbacks.Callback):
    """손실이 최소값에 도달하면 트레이닝을 중지합니다, 즉 손실이 더 이상 감소하지 않을 때.

    Arguments:
        patience: 최소값에 도달한 후 대기할 에포크 수.
            개선되지 않은 상태에서 지정된 에포크 수가 지나면, 트레이닝이 중지됩니다.
    """

    def __init__(self, patience=0):
        super().__init__()
        self.patience = patience
        # 최소 손실이 발생한 지점에서의 가중치를 저장하기 위한 best_weights.
        self.best_weights = None

    def on_train_begin(self, logs=None):
        # 손실이 더 이상 최소가 아닐 때 기다린 에포크 수.
        self.wait = 0
        # 트레이닝이 중지되는 에포크.
        self.stopped_epoch = 0
        # 초기값을 무한대로 설정.
        self.best = np.inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get("loss")
        if np.less(current, self.best):
            self.best = current
            self.wait = 0
            # 현재 결과가 더 좋으면(작으면) 최상의 가중치를 기록합니다.
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print(f"Epoch {self.stopped_epoch + 1}: early stopping")


model = get_model()
model.fit(
    x_train,
    y_train,
    batch_size=64,
    epochs=30,
    verbose=0,
    callbacks=[LossAndErrorPrintingCallback(), EarlyStoppingAtMinLoss()],
)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Up to batch 0, the average loss is   25.57.
Up to batch 1, the average loss is  471.66.
Up to batch 2, the average loss is  322.55.
Up to batch 3, the average loss is  243.88.
Up to batch 4, the average loss is  196.53.
Up to batch 5, the average loss is  165.02.
Up to batch 6, the average loss is  142.34.
Up to batch 7, the average loss is  125.17.
Up to batch 8, the average loss is  111.83.
Up to batch 9, the average loss is  101.35.
Up to batch 10, the average loss is   92.60.
Up to batch 11, the average loss is   85.16.
Up to batch 12, the average loss is   79.02.
Up to batch 13, the average loss is   73.71.
Up to batch 14, the average loss is   69.23.
Up to batch 15, the average loss is   65.26.
The average loss for epoch 0 is   65.26 and mean absolute error is    3.89.
Up to batch 0, the average loss is    3.92.
Up to batch 1, the average loss is    4.34.
Up to batch 2, the average loss is    5.39.
Up to batch 3, the average loss is    6.58.
Up to batch 4, the average loss is   10.55.
Up to batch 5, the average loss is   19.29.
Up to batch 6, the average loss is   31.58.
Up to batch 7, the average loss is   38.20.
Up to batch 8, the average loss is   41.96.
Up to batch 9, the average loss is   41.30.
Up to batch 10, the average loss is   39.31.
Up to batch 11, the average loss is   37.09.
Up to batch 12, the average loss is   35.08.
Up to batch 13, the average loss is   33.27.
Up to batch 14, the average loss is   31.54.
Up to batch 15, the average loss is   30.00.
The average loss for epoch 1 is   30.00 and mean absolute error is    4.23.
Up to batch 0, the average loss is    5.70.
Up to batch 1, the average loss is    6.90.
Up to batch 2, the average loss is    7.74.
Up to batch 3, the average loss is    8.85.
Up to batch 4, the average loss is   12.53.
Up to batch 5, the average loss is   21.55.
Up to batch 6, the average loss is   35.70.
Up to batch 7, the average loss is   44.16.
Up to batch 8, the average loss is   44.82.
Up to batch 9, the average loss is   43.07.
Up to batch 10, the average loss is   40.51.
Up to batch 11, the average loss is   38.44.
Up to batch 12, the average loss is   36.69.
Up to batch 13, the average loss is   34.77.
Up to batch 14, the average loss is   32.97.
Up to batch 15, the average loss is   31.32.
The average loss for epoch 2 is   31.32 and mean absolute error is    4.39.
Restoring model weights from the end of the best epoch.
Epoch 3: early stopping

<keras.src.callbacks.history.History at 0x1187b7430>
```

{{% /details %}}

### 학습률 스케줄링 {#learning-rate-scheduling}

이 예시에서는, 커스텀 콜백을 사용하여 트레이닝 과정에서 옵티마이저의 학습률을 동적으로 변경하는 방법을 보여줍니다.

일반적인 구현을 위해서는 `callbacks.LearningRateScheduler`를 참고하세요.

```python
class CustomLearningRateScheduler(keras.callbacks.Callback):
    """스케줄에 따라 학습률을 설정하는 학습률 스케줄러.

    Arguments:
        schedule: 에포크 인덱스(정수, 0부터 시작)와 현재 학습률을 입력으로 받아 새로운 학습률(float)을 출력으로 반환하는 함수.
    """

    def __init__(self, schedule):
        super().__init__()
        self.schedule = schedule

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, "learning_rate"):
            raise ValueError('Optimizer must have a "learning_rate" attribute.')
        # 모델의 옵티마이저에서 현재 학습률을 가져옵니다.
        lr = self.model.optimizer.learning_rate
        # 스케줄 함수 호출하여 스케줄된 학습률을 가져옵니다.
        scheduled_lr = self.schedule(epoch, lr)
        # 이 에포크가 시작되기 전에 옵티마이저에 값 설정.
        self.model.optimizer.learning_rate = scheduled_lr
        print(f"\nEpoch {epoch}: Learning rate is {float(np.array(scheduled_lr))}.")


LR_SCHEDULE = [
    # (시작 할 에포크, 학습률) 튜플
    (3, 0.05),
    (6, 0.01),
    (9, 0.005),
    (12, 0.001),
]


def lr_schedule(epoch, lr):
    """에포크에 기반하여 스케줄된 학습률을 가져오는 헬퍼 함수."""
    if epoch < LR_SCHEDULE[0][0] or epoch > LR_SCHEDULE[-1][0]:
        return lr
    for i in range(len(LR_SCHEDULE)):
        if epoch == LR_SCHEDULE[i][0]:
            return LR_SCHEDULE[i][1]
    return lr


model = get_model()
model.fit(
    x_train,
    y_train,
    batch_size=64,
    epochs=15,
    verbose=0,
    callbacks=[
        LossAndErrorPrintingCallback(),
        CustomLearningRateScheduler(lr_schedule),
    ],
)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Epoch 0: Learning rate is 0.10000000149011612.
Up to batch 0, the average loss is   27.90.
Up to batch 1, the average loss is  439.49.
Up to batch 2, the average loss is  302.08.
Up to batch 3, the average loss is  228.83.
Up to batch 4, the average loss is  184.97.
Up to batch 5, the average loss is  155.25.
Up to batch 6, the average loss is  134.03.
Up to batch 7, the average loss is  118.29.
Up to batch 8, the average loss is  105.65.
Up to batch 9, the average loss is   95.53.
Up to batch 10, the average loss is   87.25.
Up to batch 11, the average loss is   80.33.
Up to batch 12, the average loss is   74.48.
Up to batch 13, the average loss is   69.46.
Up to batch 14, the average loss is   65.05.
Up to batch 15, the average loss is   61.31.
The average loss for epoch 0 is   61.31 and mean absolute error is    3.85.
```

```plain
Epoch 1: Learning rate is 0.10000000149011612.
Up to batch 0, the average loss is   57.96.
Up to batch 1, the average loss is   55.11.
Up to batch 2, the average loss is   52.81.
Up to batch 3, the average loss is   51.06.
Up to batch 4, the average loss is   50.58.
Up to batch 5, the average loss is   51.49.
Up to batch 6, the average loss is   53.24.
Up to batch 7, the average loss is   54.20.
Up to batch 8, the average loss is   54.39.
Up to batch 9, the average loss is   54.31.
Up to batch 10, the average loss is   53.83.
Up to batch 11, the average loss is   52.93.
Up to batch 12, the average loss is   51.73.
Up to batch 13, the average loss is   50.34.
Up to batch 14, the average loss is   48.94.
Up to batch 15, the average loss is   47.65.
The average loss for epoch 1 is   47.65 and mean absolute error is    4.30.
```

```plain
Epoch 2: Learning rate is 0.10000000149011612.
Up to batch 0, the average loss is   46.38.
Up to batch 1, the average loss is   45.16.
Up to batch 2, the average loss is   44.03.
Up to batch 3, the average loss is   43.11.
Up to batch 4, the average loss is   42.52.
Up to batch 5, the average loss is   42.32.
Up to batch 6, the average loss is   43.06.
Up to batch 7, the average loss is   44.58.
Up to batch 8, the average loss is   45.33.
Up to batch 9, the average loss is   45.15.
Up to batch 10, the average loss is   44.59.
Up to batch 11, the average loss is   43.88.
Up to batch 12, the average loss is   43.17.
Up to batch 13, the average loss is   42.40.
Up to batch 14, the average loss is   41.74.
Up to batch 15, the average loss is   41.19.
The average loss for epoch 2 is   41.19 and mean absolute error is    4.27.
```

```plain
Epoch 3: Learning rate is 0.05.
Up to batch 0, the average loss is   40.85.
Up to batch 1, the average loss is   40.11.
Up to batch 2, the average loss is   39.38.
Up to batch 3, the average loss is   38.69.
Up to batch 4, the average loss is   38.01.
Up to batch 5, the average loss is   37.38.
Up to batch 6, the average loss is   36.77.
Up to batch 7, the average loss is   36.18.
Up to batch 8, the average loss is   35.61.
Up to batch 9, the average loss is   35.08.
Up to batch 10, the average loss is   34.54.
Up to batch 11, the average loss is   34.04.
Up to batch 12, the average loss is   33.56.
Up to batch 13, the average loss is   33.08.
Up to batch 14, the average loss is   32.64.
Up to batch 15, the average loss is   32.25.
The average loss for epoch 3 is   32.25 and mean absolute error is    3.64.
```

```plain
Epoch 4: Learning rate is 0.05000000074505806.
Up to batch 0, the average loss is   31.83.
Up to batch 1, the average loss is   31.42.
Up to batch 2, the average loss is   31.05.
Up to batch 3, the average loss is   30.72.
Up to batch 4, the average loss is   30.49.
Up to batch 5, the average loss is   30.37.
Up to batch 6, the average loss is   30.15.
Up to batch 7, the average loss is   29.94.
Up to batch 8, the average loss is   29.75.
Up to batch 9, the average loss is   29.56.
Up to batch 10, the average loss is   29.27.
Up to batch 11, the average loss is   28.96.
Up to batch 12, the average loss is   28.67.
Up to batch 13, the average loss is   28.39.
Up to batch 14, the average loss is   28.11.
Up to batch 15, the average loss is   27.80.
The average loss for epoch 4 is   27.80 and mean absolute error is    3.43.
```

```plain
Epoch 5: Learning rate is 0.05000000074505806.
Up to batch 0, the average loss is   27.51.
Up to batch 1, the average loss is   27.25.
Up to batch 2, the average loss is   27.05.
Up to batch 3, the average loss is   26.88.
Up to batch 4, the average loss is   26.76.
Up to batch 5, the average loss is   26.60.
Up to batch 6, the average loss is   26.44.
Up to batch 7, the average loss is   26.25.
Up to batch 8, the average loss is   26.08.
Up to batch 9, the average loss is   25.89.
Up to batch 10, the average loss is   25.71.
Up to batch 11, the average loss is   25.48.
Up to batch 12, the average loss is   25.26.
Up to batch 13, the average loss is   25.03.
Up to batch 14, the average loss is   24.81.
Up to batch 15, the average loss is   24.58.
The average loss for epoch 5 is   24.58 and mean absolute error is    3.25.
```

```plain
Epoch 6: Learning rate is 0.01.
Up to batch 0, the average loss is   24.36.
Up to batch 1, the average loss is   24.14.
Up to batch 2, the average loss is   23.93.
Up to batch 3, the average loss is   23.71.
Up to batch 4, the average loss is   23.52.
Up to batch 5, the average loss is   23.32.
Up to batch 6, the average loss is   23.12.
Up to batch 7, the average loss is   22.93.
Up to batch 8, the average loss is   22.74.
Up to batch 9, the average loss is   22.55.
Up to batch 10, the average loss is   22.37.
Up to batch 11, the average loss is   22.19.
Up to batch 12, the average loss is   22.01.
Up to batch 13, the average loss is   21.83.
Up to batch 14, the average loss is   21.67.
Up to batch 15, the average loss is   21.50.
The average loss for epoch 6 is   21.50 and mean absolute error is    2.98.
```

```plain
Epoch 7: Learning rate is 0.009999999776482582.
Up to batch 0, the average loss is   21.33.
Up to batch 1, the average loss is   21.17.
Up to batch 2, the average loss is   21.01.
Up to batch 3, the average loss is   20.85.
Up to batch 4, the average loss is   20.71.
Up to batch 5, the average loss is   20.57.
Up to batch 6, the average loss is   20.41.
Up to batch 7, the average loss is   20.27.
Up to batch 8, the average loss is   20.13.
Up to batch 9, the average loss is   19.98.
Up to batch 10, the average loss is   19.83.
Up to batch 11, the average loss is   19.69.
Up to batch 12, the average loss is   19.57.
Up to batch 13, the average loss is   19.44.
Up to batch 14, the average loss is   19.32.
Up to batch 15, the average loss is   19.19.
The average loss for epoch 7 is   19.19 and mean absolute error is    2.77.
```

```plain
Epoch 8: Learning rate is 0.009999999776482582.
Up to batch 0, the average loss is   19.07.
Up to batch 1, the average loss is   18.95.
Up to batch 2, the average loss is   18.83.
Up to batch 3, the average loss is   18.70.
Up to batch 4, the average loss is   18.58.
Up to batch 5, the average loss is   18.46.
Up to batch 6, the average loss is   18.35.
Up to batch 7, the average loss is   18.24.
Up to batch 8, the average loss is   18.12.
Up to batch 9, the average loss is   18.01.
Up to batch 10, the average loss is   17.90.
Up to batch 11, the average loss is   17.79.
Up to batch 12, the average loss is   17.68.
Up to batch 13, the average loss is   17.58.
Up to batch 14, the average loss is   17.48.
Up to batch 15, the average loss is   17.38.
The average loss for epoch 8 is   17.38 and mean absolute error is    2.61.
```

```plain
Epoch 9: Learning rate is 0.005.
Up to batch 0, the average loss is   17.28.
Up to batch 1, the average loss is   17.18.
Up to batch 2, the average loss is   17.08.
Up to batch 3, the average loss is   16.99.
Up to batch 4, the average loss is   16.90.
Up to batch 5, the average loss is   16.80.
Up to batch 6, the average loss is   16.71.
Up to batch 7, the average loss is   16.62.
Up to batch 8, the average loss is   16.53.
Up to batch 9, the average loss is   16.44.
Up to batch 10, the average loss is   16.35.
Up to batch 11, the average loss is   16.26.
Up to batch 12, the average loss is   16.17.
Up to batch 13, the average loss is   16.09.
Up to batch 14, the average loss is   16.00.
Up to batch 15, the average loss is   15.92.
The average loss for epoch 9 is   15.92 and mean absolute error is    2.48.
```

```plain
Epoch 10: Learning rate is 0.004999999888241291.
Up to batch 0, the average loss is   15.84.
Up to batch 1, the average loss is   15.76.
Up to batch 2, the average loss is   15.68.
Up to batch 3, the average loss is   15.61.
Up to batch 4, the average loss is   15.53.
Up to batch 5, the average loss is   15.45.
Up to batch 6, the average loss is   15.37.
Up to batch 7, the average loss is   15.29.
Up to batch 8, the average loss is   15.23.
Up to batch 9, the average loss is   15.15.
Up to batch 10, the average loss is   15.08.
Up to batch 11, the average loss is   15.00.
Up to batch 12, the average loss is   14.93.
Up to batch 13, the average loss is   14.86.
Up to batch 14, the average loss is   14.79.
Up to batch 15, the average loss is   14.72.
The average loss for epoch 10 is   14.72 and mean absolute error is    2.37.
```

```plain
Epoch 11: Learning rate is 0.004999999888241291.
Up to batch 0, the average loss is   14.65.
Up to batch 1, the average loss is   14.58.
Up to batch 2, the average loss is   14.52.
Up to batch 3, the average loss is   14.45.
Up to batch 4, the average loss is   14.39.
Up to batch 5, the average loss is   14.33.
Up to batch 6, the average loss is   14.26.
Up to batch 7, the average loss is   14.20.
Up to batch 8, the average loss is   14.14.
Up to batch 9, the average loss is   14.08.
Up to batch 10, the average loss is   14.02.
Up to batch 11, the average loss is   13.96.
Up to batch 12, the average loss is   13.90.
Up to batch 13, the average loss is   13.84.
Up to batch 14, the average loss is   13.78.
Up to batch 15, the average loss is   13.72.
The average loss for epoch 11 is   13.72 and mean absolute error is    2.27.
```

```plain
Epoch 12: Learning rate is 0.001.
Up to batch 0, the average loss is   13.67.
Up to batch 1, the average loss is   13.60.
Up to batch 2, the average loss is   13.55.
Up to batch 3, the average loss is   13.49.
Up to batch 4, the average loss is   13.44.
Up to batch 5, the average loss is   13.38.
Up to batch 6, the average loss is   13.33.
Up to batch 7, the average loss is   13.28.
Up to batch 8, the average loss is   13.22.
Up to batch 9, the average loss is   13.17.
Up to batch 10, the average loss is   13.12.
Up to batch 11, the average loss is   13.07.
Up to batch 12, the average loss is   13.02.
Up to batch 13, the average loss is   12.97.
Up to batch 14, the average loss is   12.92.
Up to batch 15, the average loss is   12.87.
The average loss for epoch 12 is   12.87 and mean absolute error is    2.19.
```

```plain
Epoch 13: Learning rate is 0.0010000000474974513.
Up to batch 0, the average loss is   12.82.
Up to batch 1, the average loss is   12.77.
Up to batch 2, the average loss is   12.72.
Up to batch 3, the average loss is   12.68.
Up to batch 4, the average loss is   12.63.
Up to batch 5, the average loss is   12.58.
Up to batch 6, the average loss is   12.53.
Up to batch 7, the average loss is   12.49.
Up to batch 8, the average loss is   12.45.
Up to batch 9, the average loss is   12.40.
Up to batch 10, the average loss is   12.35.
Up to batch 11, the average loss is   12.30.
Up to batch 12, the average loss is   12.26.
Up to batch 13, the average loss is   12.22.
Up to batch 14, the average loss is   12.17.
Up to batch 15, the average loss is   12.13.
The average loss for epoch 13 is   12.13 and mean absolute error is    2.12.
```

```plain
Epoch 14: Learning rate is 0.0010000000474974513.
Up to batch 0, the average loss is   12.09.
Up to batch 1, the average loss is   12.05.
Up to batch 2, the average loss is   12.01.
Up to batch 3, the average loss is   11.97.
Up to batch 4, the average loss is   11.92.
Up to batch 5, the average loss is   11.88.
Up to batch 6, the average loss is   11.84.
Up to batch 7, the average loss is   11.80.
Up to batch 8, the average loss is   11.76.
Up to batch 9, the average loss is   11.72.
Up to batch 10, the average loss is   11.68.
Up to batch 11, the average loss is   11.64.
Up to batch 12, the average loss is   11.60.
Up to batch 13, the average loss is   11.57.
Up to batch 14, the average loss is   11.54.
Up to batch 15, the average loss is   11.50.
The average loss for epoch 14 is   11.50 and mean absolute error is    2.06.

<keras.src.callbacks.history.History at 0x168619c60>
```

{{% /details %}}

### Keras 내장 콜백 {#built-in-keras-callbacks}

현재 제공하는 Keras 콜백들을 확인하려면 [API 문서]({{< relref "/docs/api/callbacks" >}})를 참고하세요.
애플리케이션에는 CSV에 로그 기록, 모델 저장, TensorBoard에서 메트릭 시각화 등 다양한 기능이 포함됩니다!
