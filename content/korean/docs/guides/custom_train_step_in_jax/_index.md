---
title: JAX에서의 `fit()` 동작을 커스터마이즈
linkTitle: JAX fit() 커스터마이즈
toc: true
weight: 5
type: docs
---

{{< keras/original checkedAt="2024-11-18" >}}

**{{< t f_author >}}** [fchollet](https://twitter.com/fchollet)  
**{{< t f_date_created >}}** 2023/06/27  
**{{< t f_last_modified >}}** 2023/06/27  
**{{< t f_description >}}** JAX를 사용하여 모델 클래스의 트레이닝 단계를 재정의.

{{< cards cols="2" >}}
{{< card link="https://colab.research.google.com/github/keras-team/keras-io/blob/master/guides/ipynb/custom_train_step_in_jax.ipynb" title="Colab" tag="Colab" tagType="warning">}}
{{< card link="https://github.com/keras-team/keras-io/blob/master/guides/custom_train_step_in_jax.py" title="GitHub" tag="GitHub">}}
{{< /cards >}}

## 소개 {#introduction}

지도 학습을 할 때는 `fit()`을 사용하면, 모든 것이 매끄럽게 작동합니다.

하지만 모든 세부 사항을 완전히 제어해야 할 경우,
처음부터 끝까지 직접 당신만의 트레이닝 루프를 작성할 수 있습니다.

그렇지만 커스텀 트레이닝 알고리즘이 필요하면서도,
콜백, 빌트인 분산 지원, 스텝 퓨징(step fusing)과 같은,
`fit()`의 편리한 기능을 그대로 활용하고 싶다면 어떻게 해야 할까요?

Keras의 핵심 원칙 중 하나는 **점진적인 복잡성 공개**입니다.
항상 점진적으로 더 낮은 레벨의 워크플로로 진입할 수 있어야 합니다.
높은 레벨의 기능이 정확히 사용 사례에 맞지 않더라도, 갑작스럽게 어려움에 부딪혀서는 안 됩니다.
높은 레벨의 편리함을 유지하면서, 작은 세부 사항에 대한 제어 권한을 더 많이 가질 수 있어야 합니다.

`fit()`이 수행하는 작업을 커스터마이즈해야 할 때는,
**`Model` 클래스의 트레이닝 스텝 함수를 재정의해야** 합니다.
이 함수는 `fit()`이 각 데이터 배치마다 호출하는 함수입니다.
이렇게 하면, 평소와 같이 `fit()`을 호출할 수 있으며,
그 안에서 사용자가 정의한 트레이닝 알고리즘이 실행됩니다.

이 패턴은 함수형 API로 모델을 만드는 것을 방해하지 않는다는 점에 주의하세요.
`Sequential` 모델, Functional API 모델,
또는 서브클래싱한 모델을 만들 때도 이 방법을 사용할 수 있습니다.

이제 그 방법을 살펴보겠습니다.

## 셋업 {#setup}

```python
import os

# 이 가이드는 JAX 백엔드에서만 실행할 수 있습니다.
os.environ["KERAS_BACKEND"] = "jax"

import jax
import keras
import numpy as np
```

## 첫 번째 간단한 예제 {#a-first-simple-example}

간단한 예제부터 시작해봅시다:

- [`keras.Model`]({{< relref "/docs/api/models/model#model-class" >}})을 서브클래싱하는 새로운 클래스를 만듭니다.
- 손실을 계산하고 모델의 트레이닝 가능하지 않은 변수의 업데이트된 값을 계산하는,
  완전히 상태가 없는(fully-stateless) `compute_loss_and_updates()` 메서드를 구현합니다.
  내부적으로는, `stateless_call()`과 빌트인 `compute_loss()`를 호출합니다.
- 현재의 메트릭 값(손실 포함)과 트레이닝 가능한 변수, 옵티마이저 변수, 메트릭 변수의 업데이트된 값을 계산하기 위한,
  완전히 상태가 없는(fully-stateless) `train_step()` 메서드를 구현합니다.

또한 `sample_weight` 인자를 다음과 같은 방법으로 고려할 수 있습니다:

- 데이터를 `x, y, sample_weight = data`로 언패킹하기
- `sample_weight`를 `compute_loss()`에 전달하기
- `sample_weight`를 `y`와 `y_pred`와 함께 `stateless_update_state()`의 메트릭에 전달하기

```python
class CustomModel(keras.Model):
    def compute_loss_and_updates(
        self,
        trainable_variables,
        non_trainable_variables,
        x,
        y,
        training=False,
    ):
        y_pred, non_trainable_variables = self.stateless_call(
            trainable_variables,
            non_trainable_variables,
            x,
            training=training,
        )
        loss = self.compute_loss(x, y, y_pred)
        return loss, (y_pred, non_trainable_variables)

    def train_step(self, state, data):
        (
            trainable_variables,
            non_trainable_variables,
            optimizer_variables,
            metrics_variables,
        ) = state
        x, y = data

        # 그래디언트 함수 가져오기
        grad_fn = jax.value_and_grad(self.compute_loss_and_updates, has_aux=True)

        # 그래디언트를 계산합니다.
        (loss, (y_pred, non_trainable_variables)), grads = grad_fn(
            trainable_variables,
            non_trainable_variables,
            x,
            y,
            training=True,
        )

        # 트레이닝 가능한 변수와 옵티마이저 변수를 업데이트합니다.
        (
            trainable_variables,
            optimizer_variables,
        ) = self.optimizer.stateless_apply(
            optimizer_variables, grads, trainable_variables
        )

        # 메트릭을 업데이트합니다.
        new_metrics_vars = []
        logs = {}
        for metric in self.metrics:
            this_metric_vars = metrics_variables[
                len(new_metrics_vars) : len(new_metrics_vars) + len(metric.variables)
            ]
            if metric.name == "loss":
                this_metric_vars = metric.stateless_update_state(this_metric_vars, loss)
            else:
                this_metric_vars = metric.stateless_update_state(
                    this_metric_vars, y, y_pred
                )
            logs[metric.name] = metric.stateless_result(this_metric_vars)
            new_metrics_vars += this_metric_vars

        # 메트릭 로그와 업데이트된 상태 변수를 반환합니다.
        state = (
            trainable_variables,
            non_trainable_variables,
            optimizer_variables,
            new_metrics_vars,
        )
        return logs, state
```

이제 시도해봅시다:

```python
# `CustomModel` 인스턴스를 생성하고 컴파일합니다.
inputs = keras.Input(shape=(32,))
outputs = keras.layers.Dense(1)(inputs)
model = CustomModel(inputs, outputs)
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# 평소처럼 `fit`을 사용하세요.
x = np.random.random((1000, 32))
y = np.random.random((1000, 1))
model.fit(x, y, epochs=3)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Epoch 1/3
 32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - mae: 1.0022 - loss: 1.2464
Epoch 2/3
 32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 198us/step - mae: 0.5811 - loss: 0.4912
Epoch 3/3
 32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 231us/step - mae: 0.4386 - loss: 0.2905

<keras.src.callbacks.history.History at 0x14da599c0>
```

{{% /details %}}

## 더 낮은 레벨로 내려가기 {#going-lower-level}

물론, `compile()`에서 손실 함수를 전달하지 않고,
대신 `train_step`에서 모든 작업을 _수동으로_ 수행할 수도 있습니다.
메트릭도 마찬가지입니다.

다음은 옵티마이저를 설정하기 위해서만 `compile()`을 사용하는, 더 낮은 레벨의 예제입니다:

```python
class CustomModel(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.mae_metric = keras.metrics.MeanAbsoluteError(name="mae")
        self.loss_fn = keras.losses.MeanSquaredError()

    def compute_loss_and_updates(
        self,
        trainable_variables,
        non_trainable_variables,
        x,
        y,
        training=False,
    ):
        y_pred, non_trainable_variables = self.stateless_call(
            trainable_variables,
            non_trainable_variables,
            x,
            training=training,
        )
        loss = self.loss_fn(y, y_pred)
        return loss, (y_pred, non_trainable_variables)

    def train_step(self, state, data):
        (
            trainable_variables,
            non_trainable_variables,
            optimizer_variables,
            metrics_variables,
        ) = state
        x, y = data

        # 그래디언트 함수를 가져옵니다.
        grad_fn = jax.value_and_grad(self.compute_loss_and_updates, has_aux=True)

        # 그래디언트를 계산합니다.
        (loss, (y_pred, non_trainable_variables)), grads = grad_fn(
            trainable_variables,
            non_trainable_variables,
            x,
            y,
            training=True,
        )

        # 트레이닝 가능한 변수와 옵티마이저 변수를 업데이트합니다.
        (
            trainable_variables,
            optimizer_variables,
        ) = self.optimizer.stateless_apply(
            optimizer_variables, grads, trainable_variables
        )

        # 메트릭을 업데이트합니다.
        loss_tracker_vars = metrics_variables[: len(self.loss_tracker.variables)]
        mae_metric_vars = metrics_variables[len(self.loss_tracker.variables) :]

        loss_tracker_vars = self.loss_tracker.stateless_update_state(
            loss_tracker_vars, loss
        )
        mae_metric_vars = self.mae_metric.stateless_update_state(
            mae_metric_vars, y, y_pred
        )

        logs = {}
        logs[self.loss_tracker.name] = self.loss_tracker.stateless_result(
            loss_tracker_vars
        )
        logs[self.mae_metric.name] = self.mae_metric.stateless_result(mae_metric_vars)

        new_metrics_vars = loss_tracker_vars + mae_metric_vars

        # 메트릭 로그와 업데이트된 상태 변수를 반환합니다.
        state = (
            trainable_variables,
            non_trainable_variables,
            optimizer_variables,
            new_metrics_vars,
        )
        return logs, state

    @property
    def metrics(self):
        # `Metric` 객체들을 여기에 나열하여,
        # 각 에포크의 시작이나 `evaluate()`의 시작 시에,
        # `reset_states()`가 자동으로 호출될 수 있도록 합니다.
        return [self.loss_tracker, self.mae_metric]


# `CustomModel` 인스턴스를 생성하고 컴파일합니다.
inputs = keras.Input(shape=(32,))
outputs = keras.layers.Dense(1)(inputs)
model = CustomModel(inputs, outputs)

# 여기에서는 손실 함수나 메트릭을 전달하지 않습니다.
model.compile(optimizer="adam")

# 평소처럼 `fit`을 사용하세요 — 콜백 등을 사용할 수 있습니다.
x = np.random.random((1000, 32))
y = np.random.random((1000, 1))
model.fit(x, y, epochs=5)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Epoch 1/5
 32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.6085 - mae: 0.6580
Epoch 2/5
 32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 215us/step - loss: 0.2630 - mae: 0.4141
Epoch 3/5
 32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 202us/step - loss: 0.2271 - mae: 0.3835
Epoch 4/5
 32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 192us/step - loss: 0.2093 - mae: 0.3714
Epoch 5/5
 32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 194us/step - loss: 0.2188 - mae: 0.3818

<keras.src.callbacks.history.History at 0x14de01420>
```

{{% /details %}}

## 당신만의 평가 스텝 제공 {#providing-your-own-evaluation-step}

`model.evaluate()` 호출에 대해서도 동일한 작업을 수행하고 싶다면 어떻게 해야 할까요?
그러면 정확히 같은 방식으로 `test_step`을 재정의하면 됩니다.
다음은 그 예시입니다:

```python
class CustomModel(keras.Model):
    def test_step(self, state, data):
        # 데이터를 언패킹합니다.
        x, y = data
        (
            trainable_variables,
            non_trainable_variables,
            metrics_variables,
        ) = state

        # 예측값과 손실을 계산합니다.
        y_pred, non_trainable_variables = self.stateless_call(
            trainable_variables,
            non_trainable_variables,
            x,
            training=False,
        )
        loss = self.compute_loss(x, y, y_pred)

        # 메트릭을 업데이트합니다.
        new_metrics_vars = []
        for metric in self.metrics:
            this_metric_vars = metrics_variables[
                len(new_metrics_vars) : len(new_metrics_vars) + len(metric.variables)
            ]
            if metric.name == "loss":
                this_metric_vars = metric.stateless_update_state(this_metric_vars, loss)
            else:
                this_metric_vars = metric.stateless_update_state(
                    this_metric_vars, y, y_pred
                )
            logs = metric.stateless_result(this_metric_vars)
            new_metrics_vars += this_metric_vars

        # 메트릭 로그와 업데이트된 상태 변수를 반환합니다.
        state = (
            trainable_variables,
            non_trainable_variables,
            new_metrics_vars,
        )
        return logs, state


# `CustomModel` 인스턴스를 생성합니다.
inputs = keras.Input(shape=(32,))
outputs = keras.layers.Dense(1)(inputs)
model = CustomModel(inputs, outputs)
model.compile(loss="mse", metrics=["mae"])

# 커스텀 `test_step`으로 평가합니다.
x = np.random.random((1000, 32))
y = np.random.random((1000, 1))
model.evaluate(x, y)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
 32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 973us/step - mae: 0.7887 - loss: 0.8385

[0.8385222554206848, 0.7956181168556213]
```

{{% /details %}}

이것으로 끝입니다!
