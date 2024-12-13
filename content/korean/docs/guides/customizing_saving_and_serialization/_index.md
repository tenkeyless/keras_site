---
title: 저장 및 직렬화 커스터마이징
linkTitle: 커스텀 저장 및 직렬화
toc: true
weight: 12
type: docs
---

{{< keras/original checkedAt="2024-11-18" >}}

**{{< t f_author >}}** Neel Kovelamudi  
**{{< t f_date_created >}}** 2023/03/15  
**{{< t f_last_modified >}}** 2023/03/15  
**{{< t f_description >}}** 레이어와 모델을 위한 저장 커스터마이징에 대한 고급 가이드.

{{< cards cols="2" >}}
{{< card link="https://colab.research.google.com/github/keras-team/keras-io/blob/master/guides/ipynb/customizing_saving_and_serialization.ipynb" title="Colab" tag="Colab" tagType="warning">}}
{{< card link="https://github.com/keras-team/keras-io/blob/master/guides/customizing_saving_and_serialization.py" title="GitHub" tag="GitHub">}}
{{< /cards >}}

## 소개 {#introduction}

이 가이드는 Keras 저장 방식에서 커스터마이징할 수 있는 고급 방법을 다룹니다.
대부분의 사용자에게는, 기본 ({{< titledRelref "/docs/guides/serialization_and_saving" >}}) 가이드에서
설명된 방법으로 충분할 것입니다.

### APIs {#apis}

우리는 다음 API들을 다룰 것입니다:

- `save_assets()` 및 `load_assets()`
- `save_own_variables()` 및 `load_own_variables()`
- `get_build_config()` 및 `build_from_config()`
- `get_compile_config()` 및 `compile_from_config()`

모델을 복원할 때, 다음 순서로 실행됩니다:

- `build_from_config()`
- `compile_from_config()`
- `load_own_variables()`
- `load_assets()`

## 셋업 {#setup}

```python
import os
import numpy as np
import keras
```

## 상태 저장 커스터마이즈 {#state-saving-customization}

이 메서드들은 `model.save()`를 호출할 때 모델 레이어의 상태가 어떻게 저장되는지를 결정합니다.
이 메서드들을 재정의하여 상태 저장 프로세스를 완전히 제어할 수 있습니다.

### `save_own_variables()` 및 `load_own_variables()` {#save_own_variables-and-load_own_variables}

이 메서드들은 각각 `model.save()` 및 `keras.models.load_model()`가 호출될 때,
레이어의 상태 변수를 저장하고 로드합니다.
기본적으로, 저장 및 로드되는 상태 변수는 레이어의 가중치(트레이닝 가능한 것과 트레이닝 불가능한 것 모두)입니다.
다음은 `save_own_variables()`의 기본 구현입니다:

```python
def save_own_variables(self, store):
    all_vars = self._trainable_weights + self._non_trainable_weights
    for i, v in enumerate(all_vars):
        store[f"{i}"] = v.numpy()
```

이 메서드에서 사용되는 저장소는 레이어 변수로 채울 수 있는 딕셔너리입니다.
커스터마이징한 예시를 살펴보겠습니다.

**예시:**

```python
@keras.utils.register_keras_serializable(package="my_custom_package")
class LayerWithCustomVariable(keras.layers.Dense):
    def __init__(self, units, **kwargs):
        super().__init__(units, **kwargs)
        self.my_variable = keras.Variable(
            np.random.random((units,)), name="my_variable", dtype="float32"
        )

    def save_own_variables(self, store):
        super().save_own_variables(store)
        # 저장 시 변수 값을 저장합니다.
        store["variables"] = self.my_variable.numpy()

    def load_own_variables(self, store):
        # 로드 시 변수 값을 할당합니다.
        self.my_variable.assign(store["variables"])
        # 나머지 가중치를 로드합니다.
        for i, v in enumerate(self.weights):
            v.assign(store[f"{i}"])
        # 참고: `load_own_variables`에서는 모든 변수(레이어 가중치 포함)를
        # 어떻게 로드할지 명시해야 합니다.

    def call(self, inputs):
        dense_out = super().call(inputs)
        return dense_out + self.my_variable


model = keras.Sequential([LayerWithCustomVariable(1)])

ref_input = np.random.random((8, 10))
ref_output = np.random.random((8, 10))
model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(ref_input, ref_output)

model.save("custom_vars_model.keras")
restored_model = keras.models.load_model("custom_vars_model.keras")

np.testing.assert_allclose(
    model.layers[0].my_variable.numpy(),
    restored_model.layers[0].my_variable.numpy(),
)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 101ms/step - loss: 0.2908
```

{{% /details %}}

### `save_assets()` 및 `load_assets()` {#save_assets-and-load_assets}

이 메서드들은 모델 클래스 정의에 추가하여, 모델이 필요한 추가 정보를 저장하고 로드할 수 있게 합니다.

예를 들어, NLP 도메인의 레이어인 `TextVectorization` 레이어나
`IndexLookup` 레이어는 저장 시 연관된 어휘(또는 조회 테이블)를 텍스트 파일에 저장할 필요가 있습니다.

이 워크플로의 기본 개념을 간단한 파일 `assets.txt`를 사용하여 살펴보겠습니다.

**예시:**

```python
@keras.saving.register_keras_serializable(package="my_custom_package")
class LayerWithCustomAssets(keras.layers.Dense):
    def __init__(self, vocab=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vocab = vocab

    def save_assets(self, inner_path):
        # 저장 시 어휘(문장)를 텍스트 파일에 작성합니다.
        with open(os.path.join(inner_path, "vocabulary.txt"), "w") as f:
            f.write(self.vocab)

    def load_assets(self, inner_path):
        # 로드 시 어휘(문장)를 텍스트 파일에서 읽어옵니다.
        with open(os.path.join(inner_path, "vocabulary.txt"), "r") as f:
            text = f.read()
        self.vocab = text.replace("<unk>", "little")


model = keras.Sequential(
    [LayerWithCustomAssets(vocab="Mary had a <unk> lamb.", units=5)]
)

x = np.random.random((10, 10))
y = model(x)

model.save("custom_assets_model.keras")
restored_model = keras.models.load_model("custom_assets_model.keras")

np.testing.assert_string_equal(
    restored_model.layers[0].vocab, "Mary had a little lamb."
)
```

## `build` 및 `compile` 저장 커스터마이즈 {#build-and-compile-saving-customization}

### `get_build_config()` 및 `build_from_config()` {#get_build_config-and-build_from_config}

이 메서드들은 레이어의 빌드 상태를 저장하고, 로드할 때 이를 복원하기 위해 함께 작동합니다.

기본적으로는 레이어의 입력 형태를 포함하는 빌드 구성 딕셔너리만 포함되지만,
이 메서드들을 재정의하여 추가 변수 및 조회 테이블을 포함시킬 수 있으며,
이는 빌드된 모델을 복원하는 데 유용할 수 있습니다.

**예시:**

```python
@keras.saving.register_keras_serializable(package="my_custom_package")
class LayerWithCustomBuild(keras.layers.Layer):
    def __init__(self, units=32, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def call(self, inputs):
        return keras.ops.matmul(inputs, self.w) + self.b

    def get_config(self):
        return dict(units=self.units, **super().get_config())

    def build(self, input_shape, layer_init):
        # `build()`를 재정의하여 추가 인자를 받습니다.
        # 따라서, `call()`을 처음 실행하기 전에
        # `layer_init` 인자로 수동으로 `build()`를 호출해야 합니다.
        super().build(input_shape)
        self._input_shape = input_shape
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer=layer_init,
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer=layer_init,
            trainable=True,
        )
        self.layer_init = layer_init

    def get_build_config(self):
        build_config = {
            "layer_init": self.layer_init,
            "input_shape": self._input_shape,
        }  # `build()`의 이니셜라이저 값을 저장합니다.
        return build_config

    def build_from_config(self, config):
        # 로드 시 `build()`를 해당 매개변수로 호출합니다.
        self.build(config["input_shape"], config["layer_init"])


custom_layer = LayerWithCustomBuild(units=16)
custom_layer.build(input_shape=(8,), layer_init="random_normal")

model = keras.Sequential(
    [
        custom_layer,
        keras.layers.Dense(1, activation="sigmoid"),
    ]
)

x = np.random.random((16, 8))
y = model(x)

model.save("custom_build_model.keras")
restored_model = keras.models.load_model("custom_build_model.keras")

np.testing.assert_equal(restored_model.layers[0].layer_init, "random_normal")
np.testing.assert_equal(restored_model.built, True)
```

### `get_compile_config()` 및 `compile_from_config()` {#get_compile_config-and-compile_from_config}

이 메서드들은 모델이 컴파일된 정보(옵티마이저, 손실 함수 등)를 저장하고,
이를 복원하여 다시 컴파일할 때 함께 작동합니다.

이 메서드를 재정의하면, 커스텀 옵티마이저나 커스텀 손실 함수 등을 사용하여 복원된 모델을 컴파일할 수 있습니다.
이러한 커스텀 항목들은 `compile_from_config()`에서 `model.compile`을 호출하기 전에 역직렬화가 필요합니다.

예시를 살펴보겠습니다.

**예시:**

```python
@keras.saving.register_keras_serializable(package="my_custom_package")
def small_square_sum_loss(y_true, y_pred):
    loss = keras.ops.square(y_pred - y_true)
    loss = loss / 10.0
    loss = keras.ops.sum(loss, axis=1)
    return loss


@keras.saving.register_keras_serializable(package="my_custom_package")
def mean_pred(y_true, y_pred):
    return keras.ops.mean(y_pred)


@keras.saving.register_keras_serializable(package="my_custom_package")
class ModelWithCustomCompile(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = keras.layers.Dense(8, activation="relu")
        self.dense2 = keras.layers.Dense(4, activation="softmax")

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

    def compile(self, optimizer, loss_fn, metrics):
        super().compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)
        self.model_optimizer = optimizer
        self.loss_fn = loss_fn
        self.loss_metrics = metrics

    def get_compile_config(self):
        # 이러한 매개변수는 저장 시 직렬화됩니다.
        return {
            "model_optimizer": self.model_optimizer,
            "loss_fn": self.loss_fn,
            "metric": self.loss_metrics,
        }

    def compile_from_config(self, config):
        # 컴파일 매개변수의 역직렬화 (중요: 커스텀 항목이 많기 때문)
        optimizer = keras.utils.deserialize_keras_object(config["model_optimizer"])
        loss_fn = keras.utils.deserialize_keras_object(config["loss_fn"])
        metrics = keras.utils.deserialize_keras_object(config["metric"])

        # 역직렬화된 매개변수로 컴파일을 호출합니다.
        self.compile(optimizer=optimizer, loss_fn=loss_fn, metrics=metrics)


model = ModelWithCustomCompile()
model.compile(
    optimizer="SGD", loss_fn=small_square_sum_loss, metrics=["accuracy", mean_pred]
)

x = np.random.random((4, 8))
y = np.random.random((4,))

model.fit(x, y)

model.save("custom_compile_model.keras")
restored_model = keras.models.load_model("custom_compile_model.keras")

np.testing.assert_equal(model.model_optimizer, restored_model.model_optimizer)
np.testing.assert_equal(model.loss_fn, restored_model.loss_fn)
np.testing.assert_equal(model.loss_metrics, restored_model.loss_metrics)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 79ms/step - accuracy: 0.0000e+00 - loss: 0.0627 - mean_metric_wrapper: 0.2500

```

{{% /details %}}

## 결론 {#conclusion}

이 튜토리얼에서 배운 메서드를 사용하면 다양한 사용 사례에 적용할 수 있으며,
복잡한 모델을 포함한 특이한 자산 및 상태 요소를 저장하고 로드할 수 있습니다. 요약하자면:

- `save_own_variables`와 `load_own_variables`는 상태가 어떻게 저장되고 로드되는지를 결정합니다.
- `save_assets`와 `load_assets`는 모델이 필요로 하는 추가 정보를 저장하고 로드하는 데 사용할 수 있습니다.
- `get_build_config`와 `build_from_config`는 모델의 빌드 상태를 저장하고 복원합니다.
- `get_compile_config`와 `compile_from_config`는 모델의 컴파일된 상태를 저장하고 복원합니다.
