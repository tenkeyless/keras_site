---
title: Keras 2 코드를 멀티 백엔드 Keras 3로 마이그레이션
linkTitle: Keras 3으로 마이그레이션
toc: true
weight: 19
type: docs
---

{{< keras/original checkedAt="2024-11-18" >}}

**{{< t f_author >}}** [Divyashree Sreepathihalli](https://github.com/divyashreepathihalli)  
**{{< t f_date_created >}}** 2023/10/23  
**{{< t f_last_modified >}}** 2023/10/30  
**{{< t f_description >}}** Keras 2 코드를 멀티 백엔드 Keras 3로 마이그레이션하기 위한 지침 및 문제 해결.

{{< cards cols="2" >}}
{{< card link="https://colab.research.google.com/github/keras-team/keras-io/blob/master/guides/ipynb/migrating_to_keras_3.ipynb" title="Colab" tag="Colab" tagType="warning">}}
{{< card link="https://github.com/keras-team/keras-io/blob/master/guides/migrating_to_keras_3.py" title="GitHub" tag="GitHub">}}
{{< /cards >}}

이 가이드는 TensorFlow 전용 Keras 2 코드를 멀티 백엔드 Keras 3 코드로 마이그레이션하는 데 도움이 됩니다.
마이그레이션에 필요한 작업은 최소화되며,
마이그레이션 후에는 Keras 워크플로를 JAX, TensorFlow 또는 PyTorch 위에서 실행할 수 있습니다.

이 가이드는 두 부분으로 구성되어 있습니다:

1. TensorFlow 백엔드에서 실행되는 Keras 3로 기존 Keras 2 코드를 마이그레이션합니다.
   이 과정은 대체로 매우 쉽지만, 주의해야 할 몇 가지 사소한 문제가 있습니다.
   이를 자세히 설명하겠습니다.
2. Keras 3 + TensorFlow 코드를 추가로 마이그레이션하여,
   다중 백엔드 Keras 3로 전환해 JAX 및 PyTorch에서도 실행 가능하도록 합니다.

시작해봅시다.

## 셋업 {#setup}

먼저, `keras-nightly`를 설치합시다.

이 예제는 TensorFlow 백엔드를 사용합니다.
(`os.environ["KERAS_BACKEND"] = "tensorflow"`)
코드를 마이그레이션한 후에는, `"tensorflow"` 문자열을 `"jax"` 또는 `"torch"`로 변경하고,
Colab에서 "Restart runtime"을 클릭하면, 코드가 JAX 또는 PyTorch 백엔드에서 실행됩니다.

```python
!pip install -q keras-nightly
```

```python
import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import tensorflow as tf
import numpy as np
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
 [[34;49mnotice[1;39;49m][39;49m A new release of pip is available: [31;49m23.3.1[39;49m -> [32;49m24.0
 [[34;49mnotice[1;39;49m][39;49m To update, run: [32;49mpip install --upgrade pip
```

{{% /details %}}

## Keras 2에서 TensorFlow 백엔드를 사용하는 Keras 3으로 마이그레이션 {#going-from-keras-2-to-keras-3-with-the-tensorflow-backend}

먼저, import를 변경하세요:

1.  `from tensorflow import keras`를 `import keras`로 변경하세요.
2.  `from tensorflow.keras import xyz` (예: `from tensorflow.keras import layers`)를
    `from keras import xyz` (예: `from keras import layers`)로 변경하세요.
3.  [`tf.keras.*`](https://www.tensorflow.org/api_docs/python/tf/keras/*)을 `keras.*`로 변경하세요.

이제 테스트를 실행해보세요.
대부분의 경우, 코드는 Keras 3에서 잘 실행될 것입니다.
만약 문제가 발생하면, 아래에 자세히 설명된 문제 해결 방법을 참고하세요.

### `jit_compile`이 GPU에서 기본적으로 `True`로 설정됩니다. {#jit_compile-is-set-to-true-by-default-on-gpu}

Keras 3에서 `Model` 생성자의 `jit_compile` 인수의 기본값이 GPU에서 `True`로 설정됩니다.
이는 모델이 기본적으로 GPU에서 JIT(Just-In-Time) 컴파일로 컴파일된다는 의미입니다.

JIT 컴파일은 일부 모델의 성능을 향상시킬 수 있습니다.
하지만 모든 TensorFlow 연산에서 작동하지 않을 수 있습니다.
커스텀 모델이나 레이어를 사용 중이고 XLA 관련 오류가 발생하면,
`jit_compile` 인수를 `False`로 설정해야 할 수 있습니다.
TensorFlow에서 XLA를 사용할 때 발생할 수 있는
[알려진 문제](https://www.tensorflow.org/xla/known_issues)를 참조하세요.
또한 XLA에서 지원되지 않는 일부 연산도 있습니다.

발생할 수 있는 오류 메시지는 다음과 같습니다:

```plain
Detected unsupported operations when trying to compile graph
__inference_one_step_on_data_125[] on XLA_GPU_JIT
```

예를 들어, 아래 코드 스니펫은 위 오류를 재현할 수 있습니다:

```python
class MyModel(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def call(self, inputs):
        string_input = tf.strings.as_string(inputs)
        return tf.strings.to_number(string_input)


subclass_model = MyModel()
x_train = np.array([[1, 2, 3], [4, 5, 6]])
subclass_model.compile(optimizer="sgd", loss="mse")
subclass_model.predict(x_train)
```

**해결 방법:**

`model.compile(..., jit_compile=False)`에서 `jit_compile=False`로 설정하거나,
`jit_compile` 속성을 다음과 같이 `False`로 설정하세요:

```python
class MyModel(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def call(self, inputs):
        # tf.strings 연산은 XLA에서 지원되지 않음
        string_input = tf.strings.as_string(inputs)
        return tf.strings.to_number(string_input)


subclass_model = MyModel()
x_train = np.array([[1, 2, 3], [4, 5, 6]])
subclass_model.jit_compile = False
subclass_model.predict(x_train)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 51ms/step

array([[1., 2., 3.],
       [4., 5., 6.]], dtype=float32)
```

{{% /details %}}

### TF SavedModel 형식으로 모델 저장하기 {#saving-a-model-in-the-tf-savedmodel-format}

Keras 3에서는 `model.save()`를 통해 TF SavedModel 형식으로 저장하는 기능이 더 이상 지원되지 않습니다.

발생할 수 있는 오류 메시지는 다음과 같습니다:

```console
>>> model.save("mymodel")
ValueError: Invalid filepath extension for saving. Please add either a `.keras` extension
for the native Keras format (recommended) or a `.h5` extension. Use
`model.export(filepath)` if you want to export a SavedModel for use with
TFLite/TFServing/etc. Received: filepath=saved_model.
```

다음 코드 스니펫은 위 오류를 재현할 수 있습니다:

```python
sequential_model = keras.Sequential([
    keras.layers.Dense(2)
])
sequential_model.save("saved_model")
```

**해결 방법:**

`model.save(filepath)` 대신 `model.export(filepath)`를 사용하세요.

```python
sequential_model = keras.Sequential([keras.layers.Dense(2)])
sequential_model(np.random.rand(3, 5))
sequential_model.export("saved_model")
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
INFO:tensorflow:Assets written to: saved_model/assets

INFO:tensorflow:Assets written to: saved_model/assets

Saved artifact at 'saved_model'. The following endpoints are available:
```

```plain
* Endpoint 'serve'
  args_0 (POSITIONAL_ONLY): TensorSpec(shape=(3, 5), dtype=tf.float32, name='keras_tensor')
Output Type:
  TensorSpec(shape=(3, 2), dtype=tf.float32, name=None)
Captures:
  14428321600: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14439128528: TensorSpec(shape=(), dtype=tf.resource, name=None)
```

{{% /details %}}

### TF SavedModel 로드하기 {#loading-a-tf-savedmodel}

Keras 3에서는 `keras.models.load_model()`을 사용하여
TF SavedModel 파일을 로드하는 기능이 더 이상 지원되지 않습니다.
`keras.models.load_model()`을 사용하려고 하면 다음과 같은 오류가 발생합니다:

```plain
ValueError: File format not supported: filepath=saved_model. Keras 3 only supports V3
`.keras` files and legacy H5 format files (`.h5` extension). Note that the legacy
SavedModel format is not supported by `load_model()` in Keras 3. In order to reload a
TensorFlow SavedModel as an inference-only layer in Keras 3, use
`keras.layers.TFSMLayer(saved_model, call_endpoint='serving_default')` (note that your
`call_endpoint` might have a different name).
```

다음 코드 스니펫은 위 오류를 재현할 수 있습니다:

```python
keras.models.load_model("saved_model")
```

**해결 방법:**

TF SavedModel을 Keras 레이어로 다시 로드하려면,
`keras.layers.TFSMLayer(filepath, call_endpoint="serving_default")`를 사용하세요.
이는 Keras에서 생성된 SavedModel에만 국한되지 않으며,
TF-Hub 모델을 포함한 모든 SavedModel에 대해 작동합니다.

```python
keras.layers.TFSMLayer("saved_model", call_endpoint="serving_default")
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
<TFSMLayer name=tfsm_layer, built=True>
```

{{% /details %}}

### Functional 모델에서 깊게 중첩된 입력 사용하기 {#using-deeply-nested-inputs-in-functional-models}

Keras 3에서는 `Model()`에 깊게 중첩된
입력/출력(예: 텐서의 리스트 안에 리스트처럼, 1단계 이상 중첩된 구조)을 전달할 수 없습니다.

이를 시도하면 다음과 같은 오류가 발생할 수 있습니다:

```plain
ValueError: When providing `inputs` as a dict, all values in the dict must be
KerasTensors. Received: inputs={'foo': <KerasTensor shape=(None, 1), dtype=float32,
sparse=None, name=foo>, 'bar': {'baz': <KerasTensor shape=(None, 1), dtype=float32,
sparse=None, name=bar>}} including invalid value {'baz': <KerasTensor shape=(None, 1),
dtype=float32, sparse=None, name=bar>} of type <class 'dict'>
```

다음 코드 스니펫은 위 오류를 재현할 수 있습니다:

```python
inputs = {
    "foo": keras.Input(shape=(1,), name="foo"),
    "bar": {
        "baz": keras.Input(shape=(1,), name="bar"),
    },
}
outputs = inputs["foo"] + inputs["bar"]["baz"]
keras.Model(inputs, outputs)
```

**해결 방법:**

중첩된 입력을 사전(dict), 리스트(list), 또는 튜플(tuple) 형태의 입력 텐서로 교체하세요.

```python
inputs = {
    "foo": keras.Input(shape=(1,), name="foo"),
    "bar": keras.Input(shape=(1,), name="bar"),
}
outputs = inputs["foo"] + inputs["bar"]
keras.Model(inputs, outputs)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
<Functional name=functional_2, built=True>
```

{{% /details %}}

### TF 오토그래프 {#tf-autograph}

Keras 2에서는, 커스텀 레이어의 `call()` 메서드에 대해 TF Autograph가 기본적으로 활성화되어 있었습니다.
그러나 Keras 3에서는 활성화되지 않습니다.
즉, 제어 흐름을 사용하는 경우 `cond` 연산을 사용해야 하거나,
대안으로 `call()` 메서드를 `@tf.function`으로 데코레이트해야 합니다.

다음과 같은 오류가 발생할 수 있습니다:

```plain
OperatorNotAllowedInGraphError: Exception encountered when calling MyCustomLayer.call().

Using a symbolic [`tf.Tensor`](https://www.tensorflow.org/api_docs/python/tf/Tensor) as a Python `bool` is not allowed. You can attempt the
following resolutions to the problem: If you are running in Graph mode, use Eager
execution mode or decorate this function with @tf.function. If you are using AutoGraph,
you can try decorating this function with @tf.function. If that does not work, then you
may be using an unsupported feature or your source code may not be visible to AutoGraph.
Here is a [link for more information](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/autograph/g3doc/ref
erence/limitations.md#access-to-source-code).
```

다음 코드 스니펫은 위 오류를 재현할 수 있습니다:

```python
class MyCustomLayer(keras.layers.Layer):

  def call(self, inputs):
    if tf.random.uniform(()) > 0.5:
      return inputs * 2
    else:
      return inputs / 2


layer = MyCustomLayer()
data = np.random.uniform(size=[3, 3])
model = keras.models.Sequential([layer])
model.compile(optimizer="adam", loss="mse")
model.predict(data)
```

**해결 방법:**

`call()` 메서드를 `@tf.function`으로 데코레이트하세요.

```python
class MyCustomLayer(keras.layers.Layer):
    @tf.function()
    def call(self, inputs):
        if tf.random.uniform(()) > 0.5:
            return inputs * 2
        else:
            return inputs / 2


layer = MyCustomLayer()
data = np.random.uniform(size=[3, 3])
model = keras.models.Sequential([layer])
model.compile(optimizer="adam", loss="mse")
model.predict(data)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 43ms/step

array([[0.59727275, 1.9986179 , 1.5514829 ],
       [0.56239295, 1.6529864 , 0.33085832],
       [0.67086476, 1.5208522 , 1.99276   ]], dtype=float32)
```

{{% /details %}}

### `KerasTensor`로 TF 연산 호출 {#calling-tf-ops-with-a-kerastensor}

Functional 모델을 구성할 때,
Keras 텐서에서 TF 연산을 사용하는 것은 허용되지 않습니다:
"A KerasTensor cannot be used as input to a TensorFlow function"
(KerasTensor는 TensorFlow 함수의 입력으로 사용할 수 없습니다).

다음과 같은 오류가 발생할 수 있습니다:

```plain
ValueError: A KerasTensor cannot be used as input to a TensorFlow function. A KerasTensor
is a symbolic placeholder for a shape and dtype, used when constructing Keras Functional
models or Keras Functions. You can only use it as input to a Keras layer or a Keras
operation (from the namespaces `keras.layers` and `keras.operations`).
```

다음 코드 스니펫은 이 오류를 재현할 수 있습니다:

```python
input = keras.layers.Input([2, 2, 1])
tf.squeeze(input)
```

**해결 방법:**

`keras.ops`에서 동등한 연산을 사용하세요.

```python
input = keras.layers.Input([2, 2, 1])
keras.ops.squeeze(input)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
<KerasTensor shape=(None, 2, 2), dtype=float32, sparse=None, name=keras_tensor_6>
```

{{% /details %}}

### 다중 출력 모델 `evaluate()` {#multi-output-model-evaluate}

The following snippet of code will reproduce the above behavior:

다중 출력 모델의 `evaluate()` 메서드는 더 이상 개별 출력 손실을 따로 반환하지 않습니다.
대신, 각 손실을 추적하려면, `compile()` 메서드에서 `metrics` 인수를 명시적으로 사용해야 합니다.

`output_a`와 `output_b`와 같은 여러 명명된 출력을 다룰 때,
이전 [`tf.keras`](https://www.tensorflow.org/api_docs/python/tf/keras)에서는
`_loss` 및 메트릭에서 유사한 항목이 자동으로 추가되었으나,
Keras 3.0에서는, 이러한 항목이 자동으로 메트릭에 추가되지 않습니다.
각 출력에 대해 개별적으로 메트릭 목록에 명시해야 합니다.

다음 코드 스니펫은 이러한 동작을 재현합니다:

```python
from keras import layers
# 여러 출력이 있는 functional 모델
inputs = layers.Input(shape=(10,))
x1 = layers.Dense(5, activation='relu')(inputs)
x2 = layers.Dense(5, activation='relu')(x1)
output_1 = layers.Dense(5, activation='softmax', name="output_1")(x1)
output_2 = layers.Dense(5, activation='softmax', name="output_2")(x2)
model = keras.Model(inputs=inputs, outputs=[output_1, output_2])
model.compile(optimizer='adam', loss='categorical_crossentropy')
# 임의의 데이터
x_test = np.random.uniform(size=[10, 10])
y_test = np.random.uniform(size=[10, 5])

model.evaluate(x_test, y_test)
```

```python
from keras import layers

# 여러 출력이 있는 functional 모델
inputs = layers.Input(shape=(10,))
x1 = layers.Dense(5, activation="relu")(inputs)
x2 = layers.Dense(5, activation="relu")(x1)
output_1 = layers.Dense(5, activation="softmax", name="output_1")(x1)
output_2 = layers.Dense(5, activation="softmax", name="output_2")(x2)
# 임의의 데이터
x_test = np.random.uniform(size=[10, 10])
y_test = np.random.uniform(size=[10, 5])
multi_output_model = keras.Model(inputs=inputs, outputs=[output_1, output_2])
multi_output_model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["categorical_crossentropy", "categorical_crossentropy"],
)
multi_output_model.evaluate(x_test, y_test)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 112ms/step - loss: 4.0217 - output_1_categorical_crossentropy: 4.0217

[4.021683692932129, 4.021683692932129]
```

{{% /details %}}

### TensorFlow 변수 추적 {#tensorflow-variables-tracking}

Keras 2와 달리, Keras 3 레이어나 모델의 속성으로
[`tf.Variable`](https://www.tensorflow.org/api_docs/python/tf/Variable)을 설정해도
변수가 자동으로 추적되지 않습니다.
아래 코드 스니펫은 [`tf.Variables`](https://www.tensorflow.org/api_docs/python/tf/Variables)가
추적되지 않는 예시를 보여줍니다.

```python
class MyCustomLayer(keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.units = units

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.w = tf.Variable(initial_value=tf.zeros([input_dim, self.units]))
        self.b = tf.Variable(initial_value=tf.zeros([self.units,]))

    def call(self, inputs):
        return keras.ops.matmul(inputs, self.w) + self.b


layer = MyCustomLayer(3)
data = np.random.uniform(size=[3, 3])
model = keras.models.Sequential([layer])
model.compile(optimizer="adam", loss="mse")
model.predict(data)
# 모델에 트레이닝 가능한 변수가 없습니다.
for layer in model.layers:
    print(layer.trainable_variables)
```

다음과 같은 경고를 볼 수 있습니다:

```plain
UserWarning: The model does not have any trainable weights.
  warnings.warn("The model does not have any trainable weights.")
```

**해결 방법:**

`self.add_weight()` 메서드를 사용하거나, `keras.Variable`을 사용하는 것을 권장합니다.
현재 [`tf.variable`](https://www.tensorflow.org/api_docs/python/tf/variable)을 사용하고 있다면,
`keras.Variable`로 전환할 수 있습니다.

```python
class MyCustomLayer(keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.units = units

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.w = self.add_weight(
            shape=[input_dim, self.units],
            initializer="zeros",
        )
        self.b = self.add_weight(
            shape=[
                self.units,
            ],
            initializer="zeros",
        )

    def call(self, inputs):
        return keras.ops.matmul(inputs, self.w) + self.b


layer = MyCustomLayer(3)
data = np.random.uniform(size=[3, 3])
model = keras.models.Sequential([layer])
model.compile(optimizer="adam", loss="mse")
model.predict(data)
# 변수가 이제 추적되는지 확인하세요.
for layer in model.layers:
    print(layer.trainable_variables)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step
[<KerasVariable shape=(3, 3), dtype=float32, path=sequential_2/my_custom_layer_1/variable>, <KerasVariable shape=(3,), dtype=float32, path=sequential_2/my_custom_layer_1/variable_1>]
```

{{% /details %}}

### 중첩된 `call()` 메서드의 인자에 있는 `None` 항목 {#none-entries-in-nested-call-arguments}

`Layer.call()` 메서드의 중첩된 (예: 리스트/튜플) 텐서 인자에서 `None` 항목은 허용되지 않으며,
`call()` 메서드의 중첩된 반환 값에서도 `None`이 허용되지 않습니다.

인자에 있는 `None`이 의도적이고 특정 목적을 가진 경우,
해당 인자를 선택적 인자로 처리하고 별도의 매개변수로 구조화해야 합니다.
예를 들어, `call` 메서드를 선택적 인자로 정의하는 것을 고려할 수 있습니다.

아래 코드 스니펫은 이 오류를 재현할 수 있습니다.

```python
class CustomLayer(keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        foo = inputs["foo"]
        baz = inputs["bar"]["baz"]
        if baz is not None:
            return foo + baz
        return foo

layer = CustomLayer()
inputs = {
    "foo": keras.Input(shape=(1,), name="foo"),
    "bar": {
        "baz": None,
    },
}
layer(inputs)
```

**해결 방법:**

**해결책 1:** `None`을 값으로 대체합니다. 예를 들어:

```python
class CustomLayer(keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        foo = inputs["foo"]
        baz = inputs["bar"]["baz"]
        return foo + baz


layer = CustomLayer()
inputs = {
    "foo": keras.Input(shape=(1,), name="foo"),
    "bar": {
        "baz": keras.Input(shape=(1,), name="bar"),
    },
}
layer(inputs)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
<KerasTensor shape=(None, 1), dtype=float32, sparse=False, name=keras_tensor_14>
```

{{% /details %}}

**해결책 2:**

선택적 인자를 사용하여 call 메서드를 정의합니다. 다음은 이 수정의 예시입니다:

```python
class CustomLayer(keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, foo, baz=None):
        if baz is not None:
            return foo + baz
        return foo


layer = CustomLayer()
foo = keras.Input(shape=(1,), name="foo")
baz = None
layer(foo, baz=baz)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
<KerasTensor shape=(None, 1), dtype=float32, sparse=False, name=keras_tensor_15>
```

{{% /details %}}

### 상태 생성 문제 {#state-building-issues}

Keras 3는 상태(예: 수치 가중치 변수)가 생성되는 시점에 대해 Keras 2보다 훨씬 엄격합니다.
Keras 3는 모델이 트레이닝되기 전에 모든 상태가 생성되기를 원합니다.
이는 JAX를 사용하는 데 필수적인 요구 사항이며,
TensorFlow는 상태 생성 시점에 대해 매우 관대했습니다.

Keras 레이어는 상태를 생성자(`__init__()` 메서드)나 `build()` 메서드에서 생성해야 합니다.
`call()` 메서드에서 상태를 생성하는 것은 피해야 합니다.

이 권장 사항을 무시하고 `call()`에서 상태를 생성하는 경우(예: 아직 빌드되지 않은 레이어를 호출하는 경우),
그러면 Keras는 트레이닝 전에 `call()` 메서드를 상징적 입력(symbolic inputs)에 대해 호출하여,
레이어를 자동으로 빌드하려고 시도할 것입니다.
그러나 이 자동 상태 생성 시도가 특정 경우에는 실패할 수 있으며,
이로 인해 다음과 같은 오류가 발생할 수 있습니다:

```plain
Layer 'frame_position_embedding' looks like it has unbuilt state,
but Keras is not able to trace the layer `call()` in order to build it automatically.
Possible causes:
1. The `call()` method of your layer may be crashing.
Try to `__call__()` the layer eagerly on some test input first to see if it works.
E.g. `x = np.random.random((3, 4)); y = layer(x)`
2. If the `call()` method is correct, then you may need to implement
the `def build(self, input_shape)` method on your layer.
It should create all variables used by the layer
(e.g. by calling `layer.build()` on all its children layers).
```

아래와 같은 레이어를, JAX 백엔드로 사용할 때, 이 오류를 재현할 수 있습니다:

```python
class PositionalEmbedding(keras.layers.Layer):
    def __init__(self, sequence_length, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=output_dim
        )
        self.sequence_length = sequence_length
        self.output_dim = output_dim

    def call(self, inputs):
        inputs = keras.ops.cast(inputs, self.compute_dtype)
        length = keras.ops.shape(inputs)[1]
        positions = keras.ops.arange(start=0, stop=length, step=1)
        embedded_positions = self.position_embeddings(positions)
        return inputs + embedded_positions
```

**해결 방법:**

오류 메시지가 요청하는 대로 수행하세요.
먼저, 레이어를 즉시 실행(eagerly) 모드로 실행하여,
`call()` 메서드가 실제로 올바른지 확인하십시오.
(참고: Keras 2에서 정상적으로 작동했다면, `call()` 메서드는 올바르며 수정할 필요가 없습니다)
`call()` 메서드가 올바른 경우,
`build(self, input_shape)` 메서드를 구현하여 모든 레이어의 상태를 생성해야 합니다.
여기에는 하위 레이어의 상태도 포함됩니다.
다음은 위 레이어에 적용된 수정 사항입니다(`build()` 메서드를 참고하세요):

```python
class PositionalEmbedding(keras.layers.Layer):
    def __init__(self, sequence_length, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=output_dim
        )
        self.sequence_length = sequence_length
        self.output_dim = output_dim

    def build(self, input_shape):
        self.position_embeddings.build(input_shape)

    def call(self, inputs):
        inputs = keras.ops.cast(inputs, self.compute_dtype)
        length = keras.ops.shape(inputs)[1]
        positions = keras.ops.arange(start=0, stop=length, step=1)
        embedded_positions = self.position_embeddings(positions)
        return inputs + embedded_positions
```

### 제거된 기능 {#removed-features}

Keras 3에서 사용 빈도가 매우 낮은 몇 가지 레거시 기능이 정리 차원에서 제거되었습니다:

- `keras.layers.ThresholdedReLU`가 제거되었습니다.
  - 대신, `ReLU` 레이어에서 `threshold` 인수를 사용하면 됩니다.
- Symbolic `Layer.add_loss()`.
  - Symbolic `add_loss()`는 제거되었습니다.
    (여전히 레이어/모델의 `call()` 메서드 내에서 `add_loss()`를 사용할 수 있습니다)
- Locally connected 레이어 (`LocallyConnected1D`, `LocallyConnected2D`)는 매우 낮은 사용 빈도로 인해 제거되었습니다.
  - 로컬로 연결된 레이어를 사용하려면, 레이어 구현을 코드베이스에 복사하여 사용하세요.
- `keras.layers.experimental.RandomFourierFeatures`는 매우 낮은 사용 빈도로 인해 제거되었습니다.
  - 이를 사용하려면, 레이어 구현을 코드베이스에 복사하여 사용하세요.
- 제거된 레이어 속성:
  - `metrics`, `dynamic` 속성이 제거되었습니다.
  - `metrics`는 여전히 `Model` 클래스에서 사용할 수 있습니다.
- RNN 레이어의 `constants` 및 `time_major` 인수가 제거되었습니다.
  - `constants` 인수는 Theano의 유산이었으며 사용 빈도가 매우 낮았습니다.
  - `time_major` 인수도 사용 빈도가 매우 낮았습니다.
- `reset_metrics` 인수:
  - `reset_metrics` 인수가 `model.*_on_batch()` 메서드에서 제거되었습니다.
  - 이 인수는 사용 빈도가 매우 낮았습니다.
- `keras.constraints.RadialConstraint` 객체가 제거되었습니다.
  - 이 객체는 사용 빈도가 매우 낮았습니다.

## 백엔드에 독립적인 Keras 3로의 전환 {#transitioning-to-backend-agnostic-keras-3}

TensorFlow 백엔드를 사용하는 Keras 3 코드는 기본적으로 TensorFlow API와 함께 작동합니다.
그러나 코드가 백엔드에 독립적이게 하려면, 다음을 수행해야 합니다:

- 모든 [`tf.*`](https://www.tensorflow.org/api_docs/python/tf/*) API 호출을,
  해당하는 Keras API로 교체합니다.
- 커스텀 `train_step`/`test_step` 메서드를 멀티 프레임워크 구현으로 변환합니다.
- 레이어에서 stateless `keras.random` 연산자를 올바르게 사용하는지 확인합니다.

각 포인트를 자세히 살펴보겠습니다.

### Keras 연산자로 전환하기 {#switching-to-keras-ops}

많은 경우, JAX와 PyTorch에서 커스텀 레이어와 메트릭을 실행할 수 있게 하려면 해야 할 유일한 일은
[`tf.*`](https://www.tensorflow.org/api_docs/python/tf/*),
[`tf.math*`](https://www.tensorflow.org/api_docs/python/tf/math*),
[`tf.linalg.*`](https://www.tensorflow.org/api_docs/python/tf/linalg/*) 등의 호출을,
`keras.ops.*`로 교체하는 것입니다.
대부분의 TensorFlow 연산자는 Keras 3와 일치해야 합니다.
이름이 다른 경우, 이 가이드에서 강조하여 설명할 것입니다.

#### NumPy ops {#numpy-ops}

Keras는 `keras.ops`의 일부로 NumPy API를 구현합니다.

아래 표는 TensorFlow와 Keras의 연산자 중 일부분만 나열한 것입니다.
표에 나열되지 않은 연산자는, 두 프레임워크에서 동일한 이름을 사용하는 경우가 많습니다.
(예: `reshape`, `matmul`, `cast` 등)

| TensorFlow                                                                                      | Keras 3.0                                                                                       |
| ----------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| [`tf.abs`](https://www.tensorflow.org/api_docs/python/tf/abs)                                   | [`keras.ops.absolute`]({{< relref "/docs/api/ops/numpy#absolute-function" >}})                  |
| [`tf.reduce_all`](https://www.tensorflow.org/api_docs/python/tf/reduce_all)                     | [`keras.ops.all`]({{< relref "/docs/api/ops/numpy#all-function" >}})                            |
| [`tf.reduce_max`](https://www.tensorflow.org/api_docs/python/tf/reduce_max)                     | [`keras.ops.amax`]({{< relref "/docs/api/ops/numpy#amax-function" >}})                          |
| [`tf.reduce_min`](https://www.tensorflow.org/api_docs/python/tf/reduce_min)                     | [`keras.ops.amin`]({{< relref "/docs/api/ops/numpy#amin-function" >}})                          |
| [`tf.reduce_any`](https://www.tensorflow.org/api_docs/python/tf/reduce_any)                     | [`keras.ops.any`]({{< relref "/docs/api/ops/numpy#any-function" >}})                            |
| [`tf.concat`](https://www.tensorflow.org/api_docs/python/tf/concat)                             | [`keras.ops.concatenate`]({{< relref "/docs/api/ops/numpy#concatenate-function" >}})            |
| [`tf.range`](https://www.tensorflow.org/api_docs/python/tf/range)                               | [`keras.ops.arange`]({{< relref "/docs/api/ops/numpy#arange-function" >}})                      |
| [`tf.acos`](https://www.tensorflow.org/api_docs/python/tf/acos)                                 | [`keras.ops.arccos`]({{< relref "/docs/api/ops/numpy#arccos-function" >}})                      |
| [`tf.asin`](https://www.tensorflow.org/api_docs/python/tf/asin)                                 | [`keras.ops.arcsin`]({{< relref "/docs/api/ops/numpy#arcsin-function" >}})                      |
| [`tf.asinh`](https://www.tensorflow.org/api_docs/python/tf/asinh)                               | [`keras.ops.arcsinh`]({{< relref "/docs/api/ops/numpy#arcsinh-function" >}})                    |
| [`tf.atan`](https://www.tensorflow.org/api_docs/python/tf/atan)                                 | [`keras.ops.arctan`]({{< relref "/docs/api/ops/numpy#arctan-function" >}})                      |
| [`tf.atan2`](https://www.tensorflow.org/api_docs/python/tf/atan2)                               | [`keras.ops.arctan2`]({{< relref "/docs/api/ops/numpy#arctan2-function" >}})                    |
| [`tf.atanh`](https://www.tensorflow.org/api_docs/python/tf/atanh)                               | [`keras.ops.arctanh`]({{< relref "/docs/api/ops/numpy#arctanh-function" >}})                    |
| [`tf.convert_to_tensor`](https://www.tensorflow.org/api_docs/python/tf/convert_to_tensor)       | [`keras.ops.convert_to_tensor`]({{< relref "/docs/api/ops/core#convert_to_tensor-function" >}}) |
| [`tf.reduce_mean`](https://www.tensorflow.org/api_docs/python/tf/reduce_mean)                   | [`keras.ops.mean`]({{< relref "/docs/api/ops/numpy#mean-function" >}})                          |
| [`tf.clip_by_value`](https://www.tensorflow.org/api_docs/python/tf/clip_by_value)               | [`keras.ops.clip`]({{< relref "/docs/api/ops/numpy#clip-function" >}})                          |
| [`tf.math.conj`](https://www.tensorflow.org/api_docs/python/tf/math/conj)                       | [`keras.ops.conjugate`]({{< relref "/docs/api/ops/numpy#conjugate-function" >}})                |
| [`tf.linalg.diag_part`](https://www.tensorflow.org/api_docs/python/tf/linalg/diag_part)         | [`keras.ops.diagonal`]({{< relref "/docs/api/ops/numpy#diagonal-function" >}})                  |
| [`tf.reverse`](https://www.tensorflow.org/api_docs/python/tf/reverse)                           | [`keras.ops.flip`]({{< relref "/docs/api/ops/numpy#flip-function" >}})                          |
| [`tf.gather`](https://www.tensorflow.org/api_docs/python/tf/gather)                             | [`keras.ops.take`]({{< relref "/docs/api/ops/numpy#take-function" >}})                          |
| [`tf.math.is_finite`](https://www.tensorflow.org/api_docs/python/tf/math/is_finite)             | [`keras.ops.isfinite`]({{< relref "/docs/api/ops/numpy#isfinite-function" >}})                  |
| [`tf.math.is_inf`](https://www.tensorflow.org/api_docs/python/tf/math/is_inf)                   | [`keras.ops.isinf`]({{< relref "/docs/api/ops/numpy#isinf-function" >}})                        |
| [`tf.math.is_nan`](https://www.tensorflow.org/api_docs/python/tf/math/is_nan)                   | [`keras.ops.isnan`]({{< relref "/docs/api/ops/numpy#isnan-function" >}})                        |
| [`tf.reduce_max`](https://www.tensorflow.org/api_docs/python/tf/reduce_max)                     | [`keras.ops.max`]({{< relref "/docs/api/ops/numpy#max-function" >}})                            |
| [`tf.reduce_mean`](https://www.tensorflow.org/api_docs/python/tf/reduce_mean)                   | [`keras.ops.mean`]({{< relref "/docs/api/ops/numpy#mean-function" >}})                          |
| [`tf.reduce_min`](https://www.tensorflow.org/api_docs/python/tf/reduce_min)                     | [`keras.ops.min`]({{< relref "/docs/api/ops/numpy#min-function" >}})                            |
| [`tf.rank`](https://www.tensorflow.org/api_docs/python/tf/rank)                                 | [`keras.ops.ndim`]({{< relref "/docs/api/ops/numpy#ndim-function" >}})                          |
| [`tf.math.pow`](https://www.tensorflow.org/api_docs/python/tf/math/pow)                         | [`keras.ops.power`]({{< relref "/docs/api/ops/numpy#power-function" >}})                        |
| [`tf.reduce_prod`](https://www.tensorflow.org/api_docs/python/tf/reduce_prod)                   | [`keras.ops.prod`]({{< relref "/docs/api/ops/numpy#prod-function" >}})                          |
| [`tf.math.reduce_std`](https://www.tensorflow.org/api_docs/python/tf/math/reduce_std)           | [`keras.ops.std`]({{< relref "/docs/api/ops/numpy#std-function" >}})                            |
| [`tf.reduce_sum`](https://www.tensorflow.org/api_docs/python/tf/reduce_sum)                     | [`keras.ops.sum`]({{< relref "/docs/api/ops/numpy#sum-function" >}})                            |
| [`tf.gather`](https://www.tensorflow.org/api_docs/python/tf/gather)                             | [`keras.ops.take`]({{< relref "/docs/api/ops/numpy#take-function" >}})                          |
| [`tf.gather_nd`](https://www.tensorflow.org/api_docs/python/tf/gather_nd)                       | [`keras.ops.take_along_axis`]({{< relref "/docs/api/ops/numpy#take_along_axis-function" >}})    |
| [`tf.math.reduce_variance`](https://www.tensorflow.org/api_docs/python/tf/math/reduce_variance) | [`keras.ops.var`]({{< relref "/docs/api/ops/numpy#var-function" >}})                            |

#### 기타 ops {#others-ops}

| TensorFlow                                                                                                                                                                                                                                                                                                                                                                             | Keras 3.0                                                                                                                                                                                    |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`tf.nn.sigmoid_cross_entropy_with_logits`](https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits)                                                                                                                                                                                                                                                        | [`keras.ops.binary_crossentropy`]({{< relref "/docs/api/ops/nn#binary_crossentropy-function" >}}) (`from_logits` 인자에 유의하세요)                                                          |
| [`tf.nn.sparse_softmax_cross_entropy_with_logits`](https://www.tensorflow.org/api_docs/python/tf/nn/sparse_softmax_cross_entropy_with_logits)                                                                                                                                                                                                                                          | [`keras.ops.sparse_categorical_crossentropy`]({{< relref "/docs/api/ops/nn#sparse_categorical_crossentropy-function" >}}) (`from_logits` 인자에 유의하세요)                                  |
| [`tf.nn.sparse_softmax_cross_entropy_with_logits`](https://www.tensorflow.org/api_docs/python/tf/nn/sparse_softmax_cross_entropy_with_logits)                                                                                                                                                                                                                                          | `keras.ops.categorical_crossentropy(target, output, from_logits=False, axis=-1)`                                                                                                             |
| [`tf.nn.conv1d`](https://www.tensorflow.org/api_docs/python/tf/nn/conv1d), [`tf.nn.conv2d`](https://www.tensorflow.org/api_docs/python/tf/nn/conv2d), [`tf.nn.conv3d`](https://www.tensorflow.org/api_docs/python/tf/nn/conv3d), [`tf.nn.convolution`](https://www.tensorflow.org/api_docs/python/tf/nn/convolution)                                                                   | [`keras.ops.conv`]({{< relref "/docs/api/ops/nn#conv-function" >}})                                                                                                                          |
| [`tf.nn.conv_transpose`](https://www.tensorflow.org/api_docs/python/tf/nn/conv_transpose), [`tf.nn.conv1d_transpose`](https://www.tensorflow.org/api_docs/python/tf/nn/conv1d_transpose), [`tf.nn.conv2d_transpose`](https://www.tensorflow.org/api_docs/python/tf/nn/conv2d_transpose), [`tf.nn.conv3d_transpose`](https://www.tensorflow.org/api_docs/python/tf/nn/conv3d_transpose) | [`keras.ops.conv_transpose`]({{< relref "/docs/api/ops/nn#conv_transpose-function" >}})                                                                                                      |
| [`tf.nn.depthwise_conv2d`](https://www.tensorflow.org/api_docs/python/tf/nn/depthwise_conv2d)                                                                                                                                                                                                                                                                                          | [`keras.ops.depthwise_conv`]({{< relref "/docs/api/ops/nn#depthwise_conv-function" >}})                                                                                                      |
| [`tf.nn.separable_conv2d`](https://www.tensorflow.org/api_docs/python/tf/nn/separable_conv2d)                                                                                                                                                                                                                                                                                          | [`keras.ops.separable_conv`]({{< relref "/docs/api/ops/nn#separable_conv-function" >}})                                                                                                      |
| [`tf.nn.batch_normalization`](https://www.tensorflow.org/api_docs/python/tf/nn/batch_normalization)                                                                                                                                                                                                                                                                                    | 직접적으로 동등한 것은 없습니다. 대신 [`keras.layers.BatchNormalization`]({{< relref "/docs/api/layers/normalization_layers/batch_normalization#batchnormalization-class" >}})를 사용하세요. |
| [`tf.nn.dropout`](https://www.tensorflow.org/api_docs/python/tf/nn/dropout)                                                                                                                                                                                                                                                                                                            | [`keras.random.dropout`]({{< relref "/docs/api/random/random_ops#dropout-function" >}})                                                                                                      |
| [`tf.nn.embedding_lookup`](https://www.tensorflow.org/api_docs/python/tf/nn/embedding_lookup)                                                                                                                                                                                                                                                                                          | [`keras.ops.take`]({{< relref "/docs/api/ops/numpy#take-function" >}})                                                                                                                       |
| [`tf.nn.l2_normalize`](https://www.tensorflow.org/api_docs/python/tf/nn/l2_normalize)                                                                                                                                                                                                                                                                                                  | [`keras.utils.normalize`]({{< relref "/docs/api/utils/python_utils#normalize-function" >}}) (op 아님)                                                                                        |
| `x.numpy`                                                                                                                                                                                                                                                                                                                                                                              | [`keras.ops.convert_to_numpy`]({{< relref "/docs/api/ops/core#convert_to_numpy-function" >}})                                                                                                |
| [`tf.scatter_nd_update`](https://www.tensorflow.org/api_docs/python/tf/scatter_nd_update)                                                                                                                                                                                                                                                                                              | [`keras.ops.scatter_update`]({{< relref "/docs/api/ops/core#scatter_update-function" >}})                                                                                                    |
| [`tf.tensor_scatter_nd_update`](https://www.tensorflow.org/api_docs/python/tf/tensor_scatter_nd_update)                                                                                                                                                                                                                                                                                | [`keras.ops.slice_update`]({{< relref "/docs/api/ops/core#slice_update-function" >}})                                                                                                        |
| [`tf.signal.fft2d`](https://www.tensorflow.org/api_docs/python/tf/signal/fft2d)                                                                                                                                                                                                                                                                                                        | [`keras.ops.fft2`]({{< relref "/docs/api/ops/fft#fft2-function" >}})                                                                                                                         |
| [`tf.signal.inverse_stft`](https://www.tensorflow.org/api_docs/python/tf/signal/inverse_stft)                                                                                                                                                                                                                                                                                          | [`keras.ops.istft`]({{< relref "/docs/api/ops/fft#istft-function" >}})                                                                                                                       |
| [`tf.image.crop_to_bounding_box`](https://www.tensorflow.org/api_docs/python/tf/image/crop_to_bounding_box)                                                                                                                                                                                                                                                                            | [`keras.ops.image.crop_images`]({{< relref "/docs/api/ops/image#crop_images-function" >}})                                                                                                   |
| [`tf.image.pad_to_bounding_box`](https://www.tensorflow.org/api_docs/python/tf/image/pad_to_bounding_box)                                                                                                                                                                                                                                                                              | [`keras.ops.image.pad_images`]({{< relref "/docs/api/ops/image#pad_images-function" >}})                                                                                                     |

### 커스텀 `train_step()` 메서드 {#custom-train_step-methods}

당신의 모델에는 TensorFlow 전용 API를 사용하는
커스텀 `train_step()` 또는 `test_step()` 메서드가 포함될 수 있습니다.
예를 들어, `train_step()` 메서드는 TensorFlow의
[`tf.GradientTape`](https://www.tensorflow.org/api_docs/python/tf/GradientTape)를 사용할 수 있습니다.
이러한 모델을 JAX 또는 PyTorch에서 실행할 수 있도록 변환하려면,
지원하려는 각 백엔드에 맞는 별도의 `train_step()` 구현을 작성해야 합니다.

일부 경우에는, `train_step()`을 재정의하는 대신, `Model.compute_loss()` 메서드를 재정의하여,
백엔드에 구애받지 않는 방식으로 만들 수 있습니다.
다음은 JAX, TensorFlow 및 PyTorch에서 작동하는,
커스텀 `compute_loss()` 메서드를 포함한 레이어의 예입니다:

```python
class MyModel(keras.Model):
    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None):
        loss = keras.ops.sum(keras.losses.mean_squared_error(y, y_pred, sample_weight))
        return loss
```

최적화 메커니즘 자체를 수정해야 한다면,
손실 계산을 넘어 `train_step()`을 재정의해야 하며,
백엔드마다 하나씩 `train_step` 메서드를 구현해야 합니다.
아래 예시와 같이 구현할 수 있습니다.

각 백엔드를 처리하는 방법에 대한 자세한 내용은 다음 가이드를 참조하십시오:

- {{< titledRelref "/docs/guides/custom_train_step_in_jax" >}}
- {{< titledRelref "/docs/guides/custom_train_step_in_tensorflow" >}}
- {{< titledRelref "/docs/guides/custom_train_step_in_torch" >}}

```python
class MyModel(keras.Model):
    def train_step(self, *args, **kwargs):
        if keras.backend.backend() == "jax":
            return self._jax_train_step(*args, **kwargs)
        elif keras.backend.backend() == "tensorflow":
            return self._tensorflow_train_step(*args, **kwargs)
        elif keras.backend.backend() == "torch":
            return self._torch_train_step(*args, **kwargs)

    def _jax_train_step(self, state, data):
        pass  # 가이드를 참고하세요: keras.io/guides/custom_train_step_in_jax/

    def _tensorflow_train_step(self, data):
        pass  # 가이드를 참고하세요: keras.io/guides/custom_train_step_in_tensorflow/

    def _torch_train_step(self, data):
        pass  # 가이드를 참고하세요: keras.io/guides/custom_train_step_in_torch/
```

### RNG를 사용하는 레이어 {#rng-using-layers}

Keras 3에는 새로운 `keras.random` 네임스페이스가 추가되었으며, 다음과 같은 기능들이 포함되어 있습니다:

- [`keras.random.normal`]({{< relref "/docs/api/random/random_ops#normal-function" >}})
- [`keras.random.uniform`]({{< relref "/docs/api/random/random_ops#uniform-function" >}})
- [`keras.random.shuffle`]({{< relref "/docs/api/random/random_ops#shuffle-function" >}})
- 등.

이 연산들은 **stateless**하며, 이는 `seed` 인자를 전달하면,
매번 동일한 결과를 반환한다는 것을 의미합니다.
예를 들어:

```python
print(keras.random.normal(shape=(), seed=123))
print(keras.random.normal(shape=(), seed=123))
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
tf.Tensor(0.7832616, shape=(), dtype=float32)
tf.Tensor(0.7832616, shape=(), dtype=float32)
```

{{% /details %}}

이 점은 stateful [`tf.random`](https://www.tensorflow.org/api_docs/python/tf/random) 연산과 다릅니다:

```python
print(tf.random.normal(shape=(), seed=123))
print(tf.random.normal(shape=(), seed=123))
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
tf.Tensor(2.4435377, shape=(), dtype=float32)
tf.Tensor(-0.6386405, shape=(), dtype=float32)
```

{{% /details %}}

RNG를 사용하는 레이어를 작성하는 경우, 호출 시마다 다른 시드 값을 사용하고 싶을 것입니다.
그러나, Python 정수를 그냥 증가시켜 전달하는 것은 적절하지 않습니다.
이는 eager 실행 시에는 문제가 없지만,
JAX, TensorFlow, PyTorch에서 지원하는 컴파일을 사용할 경우, 예상대로 작동하지 않기 때문입니다.
레이어가 처음으로 본 Python 정수 시드 값이 컴파일된 그래프에 하드코딩될 수 있습니다.

이를 해결하기 위해,
seed 인자로 stateful [`keras.random.SeedGenerator`]({{< relref "/docs/api/random/seed_generator#seedgenerator-class" >}}) 객체를 전달해야 합니다. 예를 들어:

```python
seed_generator = keras.random.SeedGenerator(1337)
print(keras.random.normal(shape=(), seed=seed_generator))
print(keras.random.normal(shape=(), seed=seed_generator))
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
tf.Tensor(0.6077996, shape=(), dtype=float32)
tf.Tensor(0.8211102, shape=(), dtype=float32)
```

{{% /details %}}

따라서 RNG를 사용하는 레이어를 작성할 때는, 다음 패턴을 사용해야 합니다:

```python
class RandomNoiseLayer(keras.layers.Layer):
    def __init__(self, noise_rate, **kwargs):
        super().__init__(**kwargs)
        self.noise_rate = noise_rate
        self.seed_generator = keras.random.SeedGenerator(1337)

    def call(self, inputs):
        noise = keras.random.uniform(
            minval=0, maxval=self.noise_rate, seed=self.seed_generator
        )
        return inputs + noise
```

이렇게 작성된 레이어는 eager 실행 또는 컴파일된 모델 어느 환경에서도 안전하게 사용할 수 있습니다.
레이어 호출 시마다 예상대로 다른 시드 값을 사용하게 됩니다.
