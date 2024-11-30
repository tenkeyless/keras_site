---
title: TensorFlow에서의 `fit()` 동작을 커스터마이즈
linkTitle: TensorFlow에서 fit() 커스터마이즈
toc: true
weight: 6
type: docs
---

{{< keras/original checkedAt="2024-11-18" >}}

**{{< t f_author >}}** [fchollet](https://twitter.com/fchollet)  
**{{< t f_date_created >}}** 2020/04/15  
**{{< t f_last_modified >}}** 2023/06/27  
**{{< t f_description >}}** TensorFlow에서 `Model` 클래스의 트레이닝 스텝을 재정의.

{{< cards cols="2" >}}
{{< card link="https://colab.research.google.com/github/keras-team/keras-io/blob/master/guides/ipynb/custom_train_step_in_tensorflow.ipynb" title="Colab" tag="Colab" tagType="warning">}}
{{< card link="https://github.com/keras-team/keras-io/blob/master/guides/custom_train_step_in_tensorflow.py" title="GitHub" tag="GitHub">}}
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

# 이 가이드는 TF 백엔드에서만 실행할 수 있습니다.
os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
import keras
from keras import layers
import numpy as np
```

## 첫 번째 간단한 예제 {#a-first-simple-example}

간단한 예제부터 시작해봅시다:

- [`keras.Model`]({{< relref "/docs/api/models/model#model-class" >}})을 서브클래싱하는 새로운 클래스를 만듭니다.
- `train_step(self, data)` 메서드만 재정의합니다.
- 메트릭 이름(손실을 포함한)과 현재 값의 매핑을 반환하는 딕셔너리를 리턴합니다.

입력 인자 `data`는 `fit`에 트레이닝 데이터로 전달되는 것입니다:

- `fit(x, y, ...)`를 호출하면서 NumPy 배열을 전달하면, `data`는 튜플 `(x, y)`가 됩니다.
- [`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset)을 `fit(dataset, ...)`으로 호출하면서 전달하면,
  `data`는 각 배치에서 `dataset`이 생성하는 값이 됩니다.

`train_step()` 메서드의 본문에서는,
이미 익숙한 일반적인 트레이닝 업데이트를 구현합니다.
중요한 것은, **`self.compute_loss()`를 통해 손실을 계산**하는 것입니다.
이 메서드는 `compile()`에 전달된 손실(들)의 함수(들)을 래핑합니다.

마찬가지로, `self.metrics`에서 메트릭에 대해 `metric.update_state(y, y_pred)`를 호출하여,
`compile()`에 전달된 메트릭의 상태를 업데이트하고,
마지막에는 `self.metrics`에서 결과를 쿼리하여 현재 값을 가져옵니다.

```python
class CustomModel(keras.Model):
    def train_step(self, data):
        # 데이터를 언패킹합니다.
        # 데이터의 구조는 모델과 `fit()`에 전달하는 값에 따라 다릅니다.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # 순전파
            # 손실 값을 계산합니다.
            # (손실 함수는 `compile()`에서 설정됩니다)
            loss = self.compute_loss(y=y, y_pred=y_pred)

        # 그래디언트를 계산합니다.
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # 가중치를 업데이트합니다.
        self.optimizer.apply(gradients, trainable_vars)

        # 메트릭을 업데이트합니다. (손실을 추적하는 메트릭 포함)
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)

        # 메트릭 이름을 현재 값에 매핑하는 딕셔너리를 반환합니다.
        return {m.name: m.result() for m in self.metrics}
```

이제 이것을 시도해봅시다:

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
 32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - mae: 0.5089 - loss: 0.3778
Epoch 2/3
 32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 318us/step - mae: 0.3986 - loss: 0.2466
Epoch 3/3
 32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 372us/step - mae: 0.3848 - loss: 0.2319

WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1699222602.443035       1 device_compiler.h:187] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.

<keras.src.callbacks.history.History at 0x2a5599f00>
```

{{% /details %}}

## 더 낮은 레벨로 내려가기 {#going-lower-level}

물론, `compile()`에서 손실 함수를 전달하지 않고,
대신 모든 것을 `train_step`에서 _수동으로_ 수행할 수 있습니다.
메트릭도 마찬가지입니다.

다음은 옵티마이저를 설정하기 위해서만 `compile()`을 사용하는,
더 낮은 레벨의 예제입니다:

- 먼저, 손실과 MAE 점수를 추적하기 위해, `Metric` 인스턴스를 생성합니다. (`__init__()`에서)
- 그런 다음, 이들 메트릭의 상태를 업데이트(그들에 대해 메트릭의 `update_state()` 호출함으로써)한 후,
  현재 평균 값을 반환하기 위해 `result()`를 통해 이를 쿼리하는,
  커스텀 `train_step()`을 구현합니다.
  이렇게 반환된 값은 진행 표시줄에 표시되거나 콜백에 전달됩니다.
- 각 에포크마다 메트릭의 `reset_states()`를 호출해야 한다는 점에 유의하세요!
  그렇지 않으면, `result()`를 호출할 때,
  에포크 시작 시점이 아닌 트레이닝 시작 이후의 평균을 반환하게 됩니다.
  보통 우리는 에포크별 평균을 사용합니다.
  다행히도, 프레임워크가 이를 처리해줍니다: 모델의 `metrics` 속성에 초기화하려는 메트릭을 나열하기만 하면 됩니다.
  모델은 `fit()` 에포크의 시작 시점이나 `evaluate()` 호출의 시작 시점에,
  여기에 나열된 모든 객체에 대해 `reset_states()`를 호출합니다.

```python
class CustomModel(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.mae_metric = keras.metrics.MeanAbsoluteError(name="mae")
        self.loss_fn = keras.losses.MeanSquaredError()

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # 순전파
            # 우리만의 손실을 계산합니다.
            loss = self.loss_fn(y, y_pred)

        # 그래디언트를 계산합니다.
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # 가중치를 업데이트합니다.
        self.optimizer.apply(gradients, trainable_vars)

        # 우리만의 메트릭을 계산합니다.
        self.loss_tracker.update_state(loss)
        self.mae_metric.update_state(y, y_pred)
        return {
            "loss": self.loss_tracker.result(),
            "mae": self.mae_metric.result(),
        }

    @property
    def metrics(self):
        # 각 에포크의 시작 시점이나 `evaluate()`의 시작 시점에,
        # `reset_states()`가 자동으로 호출될 수 있도록,
        # `Metric` 객체를 여기에 나열합니다.
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
 32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 4.0292 - mae: 1.9270
Epoch 2/5
 32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 385us/step - loss: 2.2155 - mae: 1.3920
Epoch 3/5
 32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 336us/step - loss: 1.1863 - mae: 0.9700
Epoch 4/5
 32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 373us/step - loss: 0.6510 - mae: 0.6811
Epoch 5/5
 32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 330us/step - loss: 0.4059 - mae: 0.5094

<keras.src.callbacks.history.History at 0x2a7a02860>
```

{{% /details %}}

## `sample_weight` & `class_weight` 지원 {#supporting-sampleweight-and-classweight}

첫 번째 기본 예제에서, 샘플 가중치에 대한 언급이 없었다는 것을 눈치채셨을 겁니다.
`sample_weight`와 `class_weight` 같은 `fit()` 인자를 지원하고 싶다면,
다음과 같이 간단히 할 수 있습니다:

- `data` 인자에서 `sample_weight`를 언팩합니다.
- `compute_loss`와 `update_state`에 이를 전달합니다.
  (물론, 손실 및 메트릭에 대해 `compile()`을 사용하지 않는다면,
  이를 수동으로 적용할 수도 있습니다)
- 그게 전부입니다.

```python
class CustomModel(keras.Model):
    def train_step(self, data):
        # 데이터를 언팩합니다.
        # 데이터의 구조는 모델과 `fit()`에 전달하는 값에 따라 달라집니다.
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            sample_weight = None
            x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # 순전파
            # 손실 값을 계산합니다.
            # (손실 함수는 `compile()`에서 설정됩니다)
            loss = self.compute_loss(
                y=y,
                y_pred=y_pred,
                sample_weight=sample_weight,
            )

        # 그래디언트를 계산합니다.
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # 가중치를 업데이트합니다.
        self.optimizer.apply(gradients, trainable_vars)

        # 메트릭을 업데이트합니다.
        # 메트릭은 `compile()`에서 설정됩니다.
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred, sample_weight=sample_weight)

        # 메트릭 이름을 현재 값에 매핑하는 딕셔너리를 반환합니다.
        # 여기에는 손실(`self.metrics`에서 추적된)이 포함된다는 점에 유의하세요.
        return {m.name: m.result() for m in self.metrics}


# `CustomModel` 인스턴스를 생성하고 컴파일합니다.
inputs = keras.Input(shape=(32,))
outputs = keras.layers.Dense(1)(inputs)
model = CustomModel(inputs, outputs)
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# 이제 `sample_weight` 인자를 사용할 수 있습니다.
x = np.random.random((1000, 32))
y = np.random.random((1000, 1))
sw = np.random.random((1000, 1))
model.fit(x, y, sample_weight=sw, epochs=3)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Epoch 1/3
 32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - mae: 0.4228 - loss: 0.1420
Epoch 2/3
 32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 449us/step - mae: 0.3751 - loss: 0.1058
Epoch 3/3
 32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 337us/step - mae: 0.3478 - loss: 0.0951

<keras.src.callbacks.history.History at 0x2a7491780>
```

{{% /details %}}

## 당신만의 평가 스텝 제공 {#providing-your-own-evaluation-step}

`model.evaluate()` 호출에 대해서도 동일한 작업을 수행하고 싶다면 어떻게 해야 할까요?
그러면 정확히 같은 방식으로 `test_step`을 재정의하면 됩니다.
예시는 다음과 같습니다:

```python
class CustomModel(keras.Model):
    def test_step(self, data):
        # 데이터를 언팩합니다.
        x, y = data
        # 예측값을 계산합니다.
        y_pred = self(x, training=False)
        # 손실을 추적하는 메트릭을 업데이트합니다.
        loss = self.compute_loss(y=y, y_pred=y_pred)
        # 메트릭을 업데이트합니다.
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)
        # 메트릭 이름을 현재 값에 매핑하는 딕셔너리를 반환합니다.
        # 여기에는 손실(`self.metrics`에서 추적된)이 포함된다는 점에 유의하세요.
        return {m.name: m.result() for m in self.metrics}


# `CustomModel` 인스턴스를 생성하고 컴파일합니다.
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
 32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 927us/step - mae: 0.8518 - loss: 0.9166

[0.912325382232666, 0.8567370176315308]
```

{{% /details %}}

## 마무리: 엔드투엔드 GAN 예제 {#wrapping-up-an-end-to-end-gan-example}

방금 배운 모든 것을 활용하는 엔드 투 엔드 예제를 함께 살펴보겠습니다.

다음의 경우를 고려해봅시다:

- 28x28x1 이미지를 생성하는 생성자(generator) 네트워크.
- 28x28x1 이미지를 두 개의 클래스("가짜"와 "진짜")로 분류하는 판별자(discriminator) 네트워크.
- 각 네트워크에 대한 옵티마이저.
- 판별자를 트레이닝하기 위한 손실 함수.

```python
# 판별자를 생성합니다.
discriminator = keras.Sequential(
    [
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same"),
        layers.LeakyReLU(negative_slope=0.2),
        layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same"),
        layers.LeakyReLU(negative_slope=0.2),
        layers.GlobalMaxPooling2D(),
        layers.Dense(1),
    ],
    name="discriminator",
)

# 생성자를 생성합니다.
latent_dim = 128
generator = keras.Sequential(
    [
        keras.Input(shape=(latent_dim,)),
        # 7x7x128 맵으로 reshape 할 128개의 계수를 생성하려고 합니다.
        layers.Dense(7 * 7 * 128),
        layers.LeakyReLU(negative_slope=0.2),
        layers.Reshape((7, 7, 128)),
        layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
        layers.LeakyReLU(negative_slope=0.2),
        layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
        layers.LeakyReLU(negative_slope=0.2),
        layers.Conv2D(1, (7, 7), padding="same", activation="sigmoid"),
    ],
    name="generator",
)
```

여기 `compile()`을 자체 시그니처로 재정의하고,
`train_step`에서 17줄로 전체 GAN 알고리즘을 구현한,
기능 완성형(feature-complete) GAN 클래스가 있습니다:

```python
class GAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_loss_tracker = keras.metrics.Mean(name="d_loss")
        self.g_loss_tracker = keras.metrics.Mean(name="g_loss")
        self.seed_generator = keras.random.SeedGenerator(1337)

    @property
    def metrics(self):
        return [self.d_loss_tracker, self.g_loss_tracker]

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, real_images):
        if isinstance(real_images, tuple):
            real_images = real_images[0]
        # 잠재 공간에서 랜덤 포인트를 샘플링합니다.
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = keras.random.normal(
            shape=(batch_size, self.latent_dim), seed=self.seed_generator
        )

        # 그것들을 가짜 이미지로 디코딩합니다.
        generated_images = self.generator(random_latent_vectors)

        # 그것들을 진짜 이미지와 결합합니다.
        combined_images = tf.concat([generated_images, real_images], axis=0)

        # 진짜 이미지와 가짜 이미지를 구분하는 레이블을 구성합니다.
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )
        # 레이블에 랜덤 노이즈를 추가합니다 - 중요한 트릭입니다!
        labels += 0.05 * keras.random.uniform(
            tf.shape(labels), seed=self.seed_generator
        )

        # 판별자를 트레이닝합니다.
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply(grads, self.discriminator.trainable_weights)

        # 잠재 공간에서 랜덤 포인트를 샘플링합니다.
        random_latent_vectors = keras.random.normal(
            shape=(batch_size, self.latent_dim), seed=self.seed_generator
        )

        # '모두 진짜 이미지(all real images)'라는 레이블을 구성합니다.
        misleading_labels = tf.zeros((batch_size, 1))

        # 생성자를 트레이닝합니다!
        # (판별자의 가중치는 *업데이트하지 않아야* 한다는 점에 유의하세요)
        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(random_latent_vectors))
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply(grads, self.generator.trainable_weights)

        # 메트릭을 업데이트하고 그 값을 반환합니다.
        self.d_loss_tracker.update_state(d_loss)
        self.g_loss_tracker.update_state(g_loss)
        return {
            "d_loss": self.d_loss_tracker.result(),
            "g_loss": self.g_loss_tracker.result(),
        }
```

시험해봅시다:

```python
# 데이터셋을 준비합니다. 트레이닝 및 테스트 모두 MNIST 숫자 데이터를 사용합니다.
batch_size = 64
(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
all_digits = np.concatenate([x_train, x_test])
all_digits = all_digits.astype("float32") / 255.0
all_digits = np.reshape(all_digits, (-1, 28, 28, 1))
dataset = tf.data.Dataset.from_tensor_slices(all_digits)
dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)

gan = GAN(discriminator=discriminator, generator=generator, latent_dim=latent_dim)
gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
)

# 실행 시간을 제한하기 위해 100개의 배치에서만 트레이닝합니다.
# 전체 데이터셋으로 트레이닝할 수도 있습니다.
# 좋은 결과를 얻으려면 약 20 에포크가 필요합니다.
gan.fit(dataset.take(100), epochs=1)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
 100/100 ━━━━━━━━━━━━━━━━━━━━ 51s 500ms/step - d_loss: 0.5645 - g_loss: 0.7434

<keras.src.callbacks.history.History at 0x14a4f1b10>
```

{{% /details %}}

딥러닝의 아이디어는 간단한데, 왜 구현은 어려워야 할까요?
