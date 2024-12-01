---
title: 모델 저장, 직렬화 및 export
linkTitle: 직렬화 & 저장
toc: true
weight: 11
type: docs
---

{{< keras/original checkedAt="2024-11-18" >}}

**{{< t f_author >}}** Neel Kovelamudi, Francois Chollet  
**{{< t f_date_created >}}** 2023/06/14  
**{{< t f_last_modified >}}** 2023/06/30  
**{{< t f_description >}}** 모델 저장, 직렬화 및 export에 대한 완벽 가이드.

{{< cards cols="2" >}}
{{< card link="https://colab.research.google.com/github/keras-team/keras-io/blob/master/guides/ipynb/serialization_and_saving.ipynb" title="Colab" tag="Colab" tagType="warning">}}
{{< card link="https://github.com/keras-team/keras-io/blob/master/guides/serialization_and_saving.py" title="GitHub" tag="GitHub">}}
{{< /cards >}}

## 소개 {#introduction}

Keras 모델은 여러 구성 요소로 이루어져 있습니다:

- 아키텍처 또는 설정: 모델에 포함된 레이어와 이들이 어떻게 연결되는지를 정의합니다.
- 가중치 값 세트 ("모델의 상태").
- 옵티마이저 (모델을 컴파일할 때 정의됨).
- 손실 함수 및 메트릭 세트 (모델을 컴파일할 때 정의됨).

Keras API는 이러한 모든 요소를 하나의 통합된 형식으로 저장하며, `.keras` 확장자로 표시됩니다.
이 형식은 다음을 포함하는 zip 아카이브입니다:

- JSON 기반 설정 파일 (`config.json`): 모델, 레이어 및 기타 추적 가능한 객체들의 설정을 기록합니다.
- H5 기반 상태 파일 (예: `model.weights.h5`): 레이어와 그 가중치에 대한 디렉터리 키를 포함하는 전체 모델 상태 파일.
- JSON 형식의 메타데이터 파일: 현재 Keras 버전과 같은 정보를 저장합니다.

이제 이것이 어떻게 작동하는지 살펴보겠습니다.

## 모델 저장 및 로드 방법 {#how-to-save-and-load-a-model}

이 가이드를 읽을 시간이 10초밖에 없다면, 다음 내용을 숙지하세요.

**Keras 모델 저장:**

```python
model = ...  # 모델을 가져옵니다 (Sequential, Functional Model, 또는 Model 서브클래스)
model.save('path/to/location.keras')  # 파일은 .keras 확장자로 끝나야 합니다.
```

**모델 다시 로드:**

```python
model = keras.models.load_model('path/to/location.keras')
```

이제 세부 사항을 살펴보겠습니다.

## 셋업 {#setup}

```python
import numpy as np
import keras
from keras import ops
```

## 저장 {#saving}

이 섹션은 전체 모델을 하나의 파일로 저장하는 방법에 대한 내용입니다. 이 파일에는 다음이 포함됩니다:

- 모델의 아키텍처/구성
- 모델의 가중치 값 (트레이닝 중 학습된 값)
- 모델의 컴파일 정보 (만약 `compile()`이 호출되었다면)
- 옵티마이저 및 그 상태(있을 경우, 트레이닝을 중단했던 지점에서 다시 시작할 수 있도록 해줌)

#### APIs {#apis}

`model.save()` 또는 `keras.models.save_model()`을 사용해,
모델을 저장할 수 있습니다. (둘은 동일합니다)
`keras.models.load_model()`을 사용해 다시 로드할 수 있습니다.

Keras 3에서 지원되는 유일한 형식은 ".keras" 확장자를 사용하는 "Keras v3" 형식입니다.

**예제:**

```python
def get_model():
    # 간단한 모델을 만듭니다.
    inputs = keras.Input(shape=(32,))
    outputs = keras.layers.Dense(1)(inputs)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(), loss="mean_squared_error")
    return model


model = get_model()

# 모델을 트레이닝합니다.
test_input = np.random.random((128, 32))
test_target = np.random.random((128, 1))
model.fit(test_input, test_target)

# `save('my_model.keras')`를 호출하면,
# `my_model.keras`라는 zip 아카이브가 생성됩니다.
model.save("my_model.keras")

# 동일하게 모델을 재구성할 수 있습니다.
reconstructed_model = keras.models.load_model("my_model.keras")

# 확인해봅시다:
np.testing.assert_allclose(
    model.predict(test_input), reconstructed_model.predict(test_input)
)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
 4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - loss: 0.4232
 4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 281us/step
 4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 373us/step
```

{{% /details %}}

### 커스텀 객체 {#custom-objects}

이 섹션에서는 Keras에서 커스텀 레이어, 함수, 모델을 저장하고 다시 불러오는 기본적인 워크플로를 다룹니다.

커스텀 객체(서브클래스된 레이어 등)를 포함하는 모델을 저장할 때,
객체 클래스에 `get_config()` 메서드를 **반드시** 정의해야 합니다.
만약 커스텀 객체의 생성자(`__init__()` 메서드)에 전달된 인자가
기본 타입(정수, 문자열 등)이 아닌 Python 객체라면,
`from_config()` 클래스 메서드에서 이러한 인자들을 명시적으로 **반드시** 역직렬화해야 합니다.

예시는 다음과 같습니다:

```python
class CustomLayer(keras.layers.Layer):
    def __init__(self, sublayer, **kwargs):
        super().__init__(**kwargs)
        self.sublayer = sublayer

    def call(self, x):
        return self.sublayer(x)

    def get_config(self):
        base_config = super().get_config()
        config = {
            "sublayer": keras.saving.serialize_keras_object(self.sublayer),
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        sublayer_config = config.pop("sublayer")
        sublayer = keras.saving.deserialize_keras_object(sublayer_config)
        return cls(sublayer, **config)
```

자세한 내용과 예시는 [config 메서드 정의 섹션](#config_methods)을 참조하세요.

저장된 `.keras` 파일은 경량화되어 있으며,
커스텀 객체에 대한 Python 코드를 저장하지 않습니다.
따라서, 모델을 다시 로드하려면,
`load_model`이 다음 중 하나의 방법을 통해 사용된,
커스텀 객체의 정의에 접근할 수 있어야 합니다:

1.  커스텀 객체를 등록 **(권장 방법)**,
2.  로드할 때 커스텀 객체를 직접 전달, 또는
3.  커스텀 객체 scope를 사용하는 방법

다음은 각각의 워크플로 예시입니다:

#### 커스텀 객체 등록 (**권장 방법**) {#registering-custom-objects-preferred}

이 방법은 권장되는 방법입니다.
커스텀 객체 등록은 저장 및 로드 코드를 크게 단순화합니다.
커스텀 객체의 클래스 정의에 `@keras.saving.register_keras_serializable` 데코레이터를 추가하면,
해당 객체가 전역적으로 마스터 리스트에 등록되어,
Keras가 모델을 로드할 때 객체를 인식할 수 있습니다.

이를 설명하기 위해 커스텀 레이어와 커스텀 활성화 함수를 포함한, 커스텀 모델을 만들어보겠습니다.

**예제:**

```python
# 이전에 등록된 모든 커스텀 객체를 지웁니다.
keras.saving.get_custom_objects().clear()


# 등록할 때 패키지나 이름을 선택적으로 지정할 수 있습니다.
# 지정하지 않으면 패키지는 `Custom`, 이름은 클래스 이름으로 설정됩니다.
@keras.saving.register_keras_serializable(package="MyLayers")
class CustomLayer(keras.layers.Layer):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def call(self, x):
        return x * self.factor

    def get_config(self):
        return {"factor": self.factor}


@keras.saving.register_keras_serializable(package="my_package", name="custom_fn")
def custom_fn(x):
    return x**2


# 모델을 만듭니다.
def get_model():
    inputs = keras.Input(shape=(4,))
    mid = CustomLayer(0.5)(inputs)
    outputs = keras.layers.Dense(1, activation=custom_fn)(mid)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="rmsprop", loss="mean_squared_error")
    return model


# 모델을 트레이닝합니다.
def train_model(model):
    input = np.random.random((4, 4))
    target = np.random.random((4, 1))
    model.fit(input, target)
    return model


test_input = np.random.random((4, 4))
test_target = np.random.random((4, 1))

model = get_model()
model = train_model(model)
model.save("custom_model.keras")

# 이제, 커스텀 객체에 대해 걱정할 필요 없이 간단히 로드할 수 있습니다.
reconstructed_model = keras.models.load_model("custom_model.keras")

# 확인해봅시다:
np.testing.assert_allclose(
    model.predict(test_input), reconstructed_model.predict(test_input)
)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 46ms/step - loss: 0.2571
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 11ms/step
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step
```

{{% /details %}}

#### `load_model()`에 커스텀 객체 전달 {#passing-custom-objects-to-loadmodel}

```python
model = get_model()
model = train_model(model)

# `save('my_model.keras')`를 호출하면, `my_model.keras`라는 zip 아카이브가 생성됩니다.
model.save("custom_model.keras")

# 로드할 때 `keras.models.load_model()`의 `custom_objects` 인자로
# 사용된 커스텀 객체를 포함하는 딕셔너리를 전달합니다.
reconstructed_model = keras.models.load_model(
    "custom_model.keras",
    custom_objects={"CustomLayer": CustomLayer, "custom_fn": custom_fn},
)

# 확인해봅시다:
np.testing.assert_allclose(
    model.predict(test_input), reconstructed_model.predict(test_input)
)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 37ms/step - loss: 0.0535
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step
```

{{% /details %}}

#### 커스텀 객체 scope를 사용 {#using-a-custom-object-scope}

커스텀 객체 scope 내의 코드는 scope 인자로 전달된 커스텀 객체를 인식할 수 있습니다.
따라서, 모델을 해당 scope 내에서 로드하면 커스텀 객체도 함께 로드할 수 있습니다.

**예제:**

```python
model = get_model()
model = train_model(model)
model.save("custom_model.keras")

# 커스텀 객체 딕셔너리를 커스텀 객체 scope에 전달하고,
# `keras.models.load_model()` 호출을 해당 scope 내에 배치합니다.
custom_objects = {"CustomLayer": CustomLayer, "custom_fn": custom_fn}

with keras.saving.custom_object_scope(custom_objects):
    reconstructed_model = keras.models.load_model("custom_model.keras")

# 확인해봅시다:
np.testing.assert_allclose(
    model.predict(test_input), reconstructed_model.predict(test_input)
)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 40ms/step - loss: 0.0868
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 11ms/step
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 11ms/step
```

{{% /details %}}

### 모델 직렬화 {#model-serialization}

이 섹션은 모델의 상태 없이 모델의 구성만 저장하는 방법에 대한 내용입니다.
모델의 구성(또는 아키텍처)은 모델이 어떤 레이어를 포함하고 있으며,
이 레이어들이 어떻게 연결되어 있는지를 명시합니다.
모델의 구성을 가지고 있다면,
가중치나 컴파일 정보 없이 새롭게 초기화된 상태로 모델을 생성할 수 있습니다.

#### APIs {#apis}

다음과 같은 직렬화 API를 사용할 수 있습니다:

- `keras.models.clone_model(model)`: 모델의 (랜덤하게 초기화된) 복사본을 생성합니다.
- `get_config()` 및 `cls.from_config()`: 레이어 또는 모델의 구성을 검색오고, 해당 구성으로 모델 인스턴스를 재생성합니다.
- `keras.models.model_to_json()` 및 `keras.models.model_from_json()`: 유사하지만, JSON 문자열로 처리됩니다.
- `keras.saving.serialize_keras_object()`: 임의의 Keras 객체 구성을 검색합니다.
- `keras.saving.deserialize_keras_object()`: 객체의 구성으로부터 인스턴스를 재생성합니다.

#### 메모리 내에서 모델 복제 {#in-memory-model-cloning}

`keras.models.clone_model()`을 사용하여 메모리 내에서 모델을 복제할 수 있습니다.
이는 config를 가져온 다음 해당 config에서 모델을 재생성하는 것과 동일하며,
컴파일 정보나 레이어 가중치 값은 유지되지 않습니다.

**예제:**

```python
new_model = keras.models.clone_model(model)
```

#### `get_config()` 및 `from_config()` {#getconfig-and-fromconfig}

`model.get_config()` 또는 `layer.get_config()`을 호출하면,
각각 모델 또는 레이어의 구성을 포함한 Python 딕셔너리가 반환됩니다.
모델이나 레이어의 `__init__()` 메서드에 필요한 인자를 `get_config()`에 정의해야 합니다.
로딩 시, `from_config(config)` 메서드가 이러한 인자들과 함께,
`__init__()`을 호출하여 모델 또는 레이어를 재구성합니다.

**레이어 예시:**

```python
layer = keras.layers.Dense(3, activation="relu")
layer_config = layer.get_config()
print(layer_config)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
{'name': 'dense_4', 'trainable': True, 'dtype': 'float32', 'units': 3, 'activation': 'relu', 'use_bias': True, 'kernel_initializer': {'module': 'keras.src.initializers.random_initializers', 'class_name': 'GlorotUniform', 'config': {'seed': None}, 'registered_name': 'GlorotUniform'}, 'bias_initializer': {'module': 'keras.src.initializers.constant_initializers', 'class_name': 'Zeros', 'config': {}, 'registered_name': 'Zeros'}, 'kernel_regularizer': None, 'bias_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None}
```

{{% /details %}}

이제 `from_config()` 메서드를 사용하여 레이어를 재구성해봅시다:

```python
new_layer = keras.layers.Dense.from_config(layer_config)
```

**Sequential 모델 예시:**

```python
model = keras.Sequential([keras.Input((32,)), keras.layers.Dense(1)])
config = model.get_config()
new_model = keras.Sequential.from_config(config)
```

**Functional 모델 예시:**

```python
inputs = keras.Input((32,))
outputs = keras.layers.Dense(1)(inputs)
model = keras.Model(inputs, outputs)
config = model.get_config()
new_model = keras.Model.from_config(config)
```

#### `to_json()` 및 `keras.models.model_from_json()` {#tojson-and-kerasmodelsmodelfromjson}

이것은 `get_config` / `from_config`과 유사하지만,
모델을 JSON 문자열로 변환하여 원본 모델 클래스 없이도 로드할 수 있습니다.
또한, 이는 모델에만 적용되며 레이어에는 사용되지 않습니다.

**예제:**

```python
model = keras.Sequential([keras.Input((32,)), keras.layers.Dense(1)])
json_config = model.to_json()
new_model = keras.models.model_from_json(json_config)
```

#### 임의 객체 직렬화 및 역직렬화 {#arbitrary-object-serialization-and-deserialization}

`keras.saving.serialize_keras_object()` 및 `keras.saving.deserialize_keras_object()` API는
Keras 객체와 커스텀 객체를 직렬화하거나 역직렬화할 수 있는 범용 API입니다.
이는 모델 아키텍처 저장의 기반이 되며,
Keras 내의 모든 `serialize()`/`deserialize()` 호출의 배경에 있습니다.

**예제**:

```python
my_reg = keras.regularizers.L1(0.005)
config = keras.saving.serialize_keras_object(my_reg)
print(config)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
{'module': 'keras.src.regularizers.regularizers', 'class_name': 'L1', 'config': {'l1': 0.004999999888241291}, 'registered_name': 'L1'}
```

{{% /details %}}

올바르게 재구성하기 위한, 모든 필수 정보를 포함하는 직렬화 형식을 확인하세요:

- `module`: 객체가 속한 Keras 모듈 또는 기타 식별 모듈의 이름을 포함
- `class_name`: 객체 클래스의 이름을 포함
- `config`: 객체를 재구성하는 데 필요한 모든 정보
- `registered_name`: 커스텀 객체의 경우. [여기](#custom_object_serialization)를 참조하세요.

이제 정규화기(regularizer)를 재구성할 수 있습니다.

```python
new_reg = keras.saving.deserialize_keras_object(config)
```

### 모델 가중치 저장 {#model-weights-saving}

모델의 가중치만 저장하고 로드하는 것도 선택할 수 있습니다.
이는 다음과 같은 경우에 유용할 수 있습니다:

- 모델을 추론에만 사용할 경우: 이 경우 트레이닝을 다시 시작할 필요가 없으므로, 컴파일 정보나 옵티마이저 상태가 필요하지 않습니다.
- 전이 학습을 할 경우: 이전 모델의 상태를 재사용하여 새 모델을 트레이닝하려는 경우, 이전 모델의 컴파일 정보는 필요하지 않습니다.

#### 메모리 내에서 가중치 전송을 위한 API {#apis-for-in-memory-weight-transfer}

`get_weights()`와 `set_weights()`를 사용하여, 다른 객체 간에 가중치를 복사할 수 있습니다:

- `keras.layers.Layer.get_weights()`: 가중치 값을 NumPy 배열 목록으로 반환합니다.
- `keras.layers.Layer.set_weights(weights)`: 제공된 NumPy 배열로 모델의 가중치를 설정합니다.

예제:

**_메모리 내에서, 한 레이어에서 다른 레이어로 가중치 전송_**

```python
def create_layer():
    layer = keras.layers.Dense(64, activation="relu", name="dense_2")
    layer.build((None, 784))
    return layer


layer_1 = create_layer()
layer_2 = create_layer()

# 레이어 1에서 레이어 2로 가중치 복사
layer_2.set_weights(layer_1.get_weights())
```

**_메모리 내에서, 호환되는 아키텍처를 가진 두 모델 간에 가중치 전송_**

```python
# 간단한 Functional 모델 생성
inputs = keras.Input(shape=(784,), name="digits")
x = keras.layers.Dense(64, activation="relu", name="dense_1")(inputs)
x = keras.layers.Dense(64, activation="relu", name="dense_2")(x)
outputs = keras.layers.Dense(10, name="predictions")(x)
functional_model = keras.Model(inputs=inputs, outputs=outputs, name="3_layer_mlp")


# 동일한 아키텍처를 가진 서브클래스된 모델 정의
class SubclassedModel(keras.Model):
    def __init__(self, output_dim, name=None):
        super().__init__(name=name)
        self.output_dim = output_dim
        self.dense_1 = keras.layers.Dense(64, activation="relu", name="dense_1")
        self.dense_2 = keras.layers.Dense(64, activation="relu", name="dense_2")
        self.dense_3 = keras.layers.Dense(output_dim, name="predictions")

    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        x = self.dense_3(x)
        return x

    def get_config(self):
        return {"output_dim": self.output_dim, "name": self.name}


subclassed_model = SubclassedModel(10)
# 서브클래스된 모델을 한 번 호출하여 가중치를 생성합니다.
subclassed_model(np.ones((1, 784)))

# functional_model에서 subclassed_model로 가중치를 복사합니다.
subclassed_model.set_weights(functional_model.get_weights())

assert len(functional_model.weights) == len(subclassed_model.weights)
for a, b in zip(functional_model.weights, subclassed_model.weights):
    np.testing.assert_allclose(a.numpy(), b.numpy())
```

**_stateless 레이어의 경우_**

stateless 레이어는 가중치의 순서나 수를 변경하지 않기 때문에,
stateless 레이어가 추가되거나 빠지더라도(extra/missing) 모델은 호환되는 아키텍처를 가질 수 있습니다.

```python
inputs = keras.Input(shape=(784,), name="digits")
x = keras.layers.Dense(64, activation="relu", name="dense_1")(inputs)
x = keras.layers.Dense(64, activation="relu", name="dense_2")(x)
outputs = keras.layers.Dense(10, name="predictions")(x)
functional_model = keras.Model(inputs=inputs, outputs=outputs, name="3_layer_mlp")

inputs = keras.Input(shape=(784,), name="digits")
x = keras.layers.Dense(64, activation="relu", name="dense_1")(inputs)
x = keras.layers.Dense(64, activation="relu", name="dense_2")(x)

# 가중치를 포함하지 않는, 드롭아웃 레이어를 추가합니다.
x = keras.layers.Dropout(0.5)(x)
outputs = keras.layers.Dense(10, name="predictions")(x)
functional_model_with_dropout = keras.Model(
    inputs=inputs, outputs=outputs, name="3_layer_mlp"
)

functional_model_with_dropout.set_weights(functional_model.get_weights())
```

#### 가중치를 디스크에 저장하고 다시 로드하는 API {#apis-for-saving-weights-to-disk-and-loading-them-back}

`model.save_weights(filepath)`을 호출하여, 가중치를 디스크에 저장할 수 있습니다.
파일 이름은 `.weights.h5`로 끝나야 합니다.

**예제:**

```python
# 실행 가능한 예시
sequential_model = keras.Sequential(
    [
        keras.Input(shape=(784,), name="digits"),
        keras.layers.Dense(64, activation="relu", name="dense_1"),
        keras.layers.Dense(64, activation="relu", name="dense_2"),
        keras.layers.Dense(10, name="predictions"),
    ]
)
sequential_model.save_weights("my_model.weights.h5")
sequential_model.load_weights("my_model.weights.h5")
```

모델에 중첩된 레이어가 있을 경우,
`layer.trainable`을 변경하면 `layer.weights`의 순서가 달라질 수 있습니다.

```python
class NestedDenseLayer(keras.layers.Layer):
    def __init__(self, units, name=None):
        super().__init__(name=name)
        self.dense_1 = keras.layers.Dense(units, name="dense_1")
        self.dense_2 = keras.layers.Dense(units, name="dense_2")

    def call(self, inputs):
        return self.dense_2(self.dense_1(inputs))


nested_model = keras.Sequential([keras.Input((784,)), NestedDenseLayer(10, "nested")])
variable_names = [v.name for v in nested_model.weights]
print("variables: {}".format(variable_names))

print("\nChanging trainable status of one of the nested layers...")
nested_model.get_layer("nested").dense_1.trainable = False

variable_names_2 = [v.name for v in nested_model.weights]
print("\nvariables: {}".format(variable_names_2))
print("variable ordering changed:", variable_names != variable_names_2)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
variables: ['kernel', 'bias', 'kernel', 'bias']
```

```plain
Changing trainable status of one of the nested layers...
```

```plain
variables: ['kernel', 'bias', 'kernel', 'bias']
variable ordering changed: False
```

{{% /details %}}

##### **전이 학습 예시** {#transfer-learning-example}

사전 트레이닝된 가중치를 가중치 파일에서 로드할 때는,
원래 체크포인트된 모델에 가중치를 로드한 후,
원하는 가중치/레이어를 새로운 모델에 추출하는 것이 좋습니다.

**예제:**

```python
def create_functional_model():
    inputs = keras.Input(shape=(784,), name="digits")
    x = keras.layers.Dense(64, activation="relu", name="dense_1")(inputs)
    x = keras.layers.Dense(64, activation="relu", name="dense_2")(x)
    outputs = keras.layers.Dense(10, name="predictions")(x)
    return keras.Model(inputs=inputs, outputs=outputs, name="3_layer_mlp")


functional_model = create_functional_model()
functional_model.save_weights("pretrained.weights.h5")

# 별도의 프로그램에서:
pretrained_model = create_functional_model()
pretrained_model.load_weights("pretrained.weights.h5")

# 원본 모델에서 레이어를 추출하여, 새로운 모델을 생성합니다:
extracted_layers = pretrained_model.layers[:-1]
extracted_layers.append(keras.layers.Dense(5, name="dense_3"))
model = keras.Sequential(extracted_layers)
model.summary()
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Model: "sequential_4"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape              ┃    Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ dense_1 (Dense)                 │ (None, 64)                │     50,240 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ dense_2 (Dense)                 │ (None, 64)                │      4,160 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ dense_3 (Dense)                 │ (None, 5)                 │        325 │
└─────────────────────────────────┴───────────────────────────┴────────────┘
 Total params: 54,725 (213.77 KB)
 Trainable params: 54,725 (213.77 KB)
 Non-trainable params: 0 (0.00 B)
```

{{% /details %}}

### 부록: 커스텀 객체 다루기 {#appendix-handling-custom-objects}

#### config 메서드 정의 {#config_methods}

명세:

- `get_config()`은
  Keras 아키텍처 및 모델 저장 API와 호환되기 위해, JSON 직렬화가 가능한 딕셔너리를 반환해야 합니다.
- `from_config(config)` (`classmethod`)은
  config에서 생성된 새로운 레이어 또는 모델 객체를 반환해야 합니다.
  기본 구현은 `cls(**config)`을 반환합니다.

**참고**: 모든 생성자 인자가 이미 직렬화 가능하다면, 예: 문자열, 정수, 또는 커스텀 Keras 객체가 아니라면,
`from_config`를 재정의할 필요가 없습니다.
그러나, `__init__`에 전달된 레이어 또는 모델과 같은 더 복잡한 객체에 대해서는,
`__init__` 자체에서 역직렬화를 명시적으로 처리하거나 `from_config()` 메서드를 재정의해야 합니다.

**예제:**

```python
@keras.saving.register_keras_serializable(package="MyLayers", name="KernelMult")
class MyDense(keras.layers.Layer):
    def __init__(
        self,
        units,
        *,
        kernel_regularizer=None,
        kernel_initializer=None,
        nested_model=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_units = units
        self.kernel_regularizer = kernel_regularizer
        self.kernel_initializer = kernel_initializer
        self.nested_model = nested_model

    def get_config(self):
        config = super().get_config()
        # 커스텀 레이어의 매개변수로 config를 업데이트합니다.
        config.update(
            {
                "units": self.hidden_units,
                "kernel_regularizer": self.kernel_regularizer,
                "kernel_initializer": self.kernel_initializer,
                "nested_model": self.nested_model,
            }
        )
        return config

    def build(self, input_shape):
        input_units = input_shape[-1]
        self.kernel = self.add_weight(
            name="kernel",
            shape=(input_units, self.hidden_units),
            regularizer=self.kernel_regularizer,
            initializer=self.kernel_initializer,
        )

    def call(self, inputs):
        return ops.matmul(inputs, self.kernel)


layer = MyDense(units=16, kernel_regularizer="l1", kernel_initializer="ones")
layer3 = MyDense(units=64, nested_model=layer)

config = keras.layers.serialize(layer3)

print(config)

new_layer = keras.layers.deserialize(config)

print(new_layer)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
{'module': None, 'class_name': 'MyDense', 'config': {'name': 'my_dense_1', 'trainable': True, 'dtype': 'float32', 'units': 64, 'kernel_regularizer': None, 'kernel_initializer': None, 'nested_model': {'module': None, 'class_name': 'MyDense', 'config': {'name': 'my_dense', 'trainable': True, 'dtype': 'float32', 'units': 16, 'kernel_regularizer': 'l1', 'kernel_initializer': 'ones', 'nested_model': None}, 'registered_name': 'MyLayers>KernelMult'}}, 'registered_name': 'MyLayers>KernelMult'}
<MyDense name=my_dense_1, built=False>
```

{{% /details %}}

위의 `MyDense`에서는 `from_config`를 재정의할 필요가 없다는 점에 유의하세요.
왜냐하면 `hidden_units`, `kernel_initializer`, 그리고 `kernel_regularizer`가
각각 정수, 문자열, 그리고 빌트인 Keras 객체이기 때문입니다.
이는 기본 `from_config` 구현인 `cls(**config)`가 의도한 대로 작동한다는 것을 의미합니다.

`__init__`에 전달되는 레이어나 모델과 같은, 더 복잡한 객체에 대해서는 명시적으로 해당 객체를 역직렬화해야 합니다.
`from_config` 재정의가 필요한 모델의 예시를 살펴보겠습니다.

<span id="registration_example"></span>

**예제:**

```python
@keras.saving.register_keras_serializable(package="ComplexModels")
class CustomModel(keras.layers.Layer):
    def __init__(self, first_layer, second_layer=None, **kwargs):
        super().__init__(**kwargs)
        self.first_layer = first_layer
        if second_layer is not None:
            self.second_layer = second_layer
        else:
            self.second_layer = keras.layers.Dense(8)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "first_layer": self.first_layer,
                "second_layer": self.second_layer,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        # 여기서 [`keras.saving.deserialize_keras_object`](/api/models/model_saving_apis/serialization_utils#deserializekerasobject-function)
        # 를 사용할 수도 있습니다.
        config["first_layer"] = keras.layers.deserialize(config["first_layer"])
        config["second_layer"] = keras.layers.deserialize(config["second_layer"])
        return cls(**config)

    def call(self, inputs):
        return self.first_layer(self.second_layer(inputs))


# 첫 번째 레이어로 이전 예시에서 만든 커스텀 레이어 (MyDense)를 사용합니다.
inputs = keras.Input((32,))
outputs = CustomModel(first_layer=layer)(inputs)
model = keras.Model(inputs, outputs)

config = model.get_config()
new_model = keras.Model.from_config(config)
```

#### 커스텀 객체가 직렬화되는 방식 {#custom_object_serialization}

직렬화 형식에는 `@keras.saving.register_keras_serializable`로 등록된 커스텀 객체에 대한 특별한 키가 있습니다.
이 `registered_name` 키는 로딩/역직렬화 시 Keras 마스터 리스트에서 쉽게 검색할 수 있도록 하며,
사용자들이 커스텀 이름을 추가할 수 있도록 합니다.

위에서 정의한 커스텀 레이어 `MyDense`를 직렬화한 후의 구성을 살펴보겠습니다.

**예제**:

```python
layer = MyDense(
    units=16,
    kernel_regularizer=keras.regularizers.L1L2(l1=1e-5, l2=1e-4),
    kernel_initializer="ones",
)
config = keras.layers.serialize(layer)
print(config)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
{'module': None, 'class_name': 'MyDense', 'config': {'name': 'my_dense_2', 'trainable': True, 'dtype': 'float32', 'units': 16, 'kernel_regularizer': {'module': 'keras.src.regularizers.regularizers', 'class_name': 'L1L2', 'config': {'l1': 1e-05, 'l2': 0.0001}, 'registered_name': 'L1L2'}, 'kernel_initializer': 'ones', 'nested_model': None}, 'registered_name': 'MyLayers>KernelMult'}
```

{{% /details %}}

보시다시피, `registered_name` 키에는 Keras 마스터 리스트의 조회 정보를 포함하고 있으며,
`MyLayers`라는 패키지와 `@keras.saving.register_keras_serializable` 데코레이터에서 지정한 커스텀 이름 `KernelMult`가 포함되어 있습니다.
커스텀 클래스 정의/등록에 대해서는 다시 [여기](#registration_example)를 참조하세요.

`class_name` 키는 클래스의 원본 이름을 포함하여, `from_config`에서 적절한 재초기화를 가능하게 합니다.

또한, `module` 키가 `None`인 것을 주목하세요.
이는 커스텀 객체이기 때문입니다.
