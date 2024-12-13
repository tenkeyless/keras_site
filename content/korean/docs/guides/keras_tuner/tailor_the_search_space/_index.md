---
title: 검색 공간 맞춤 설정
linkTitle: 검색 공간 맞춤 설정
toc: true
weight: 6
type: docs
---

{{< keras/original checkedAt="2024-11-19" >}}

**{{< t f_author >}}** Luca Invernizzi, James Long, Francois Chollet, Tom O'Malley, Haifeng Jin  
**{{< t f_date_created >}}** 2019/05/31  
**{{< t f_last_modified >}}** 2021/10/27  
**{{< t f_description >}}** 하이퍼모델을 변경하지 않고, 하이퍼파라미터의 일부만 튜닝하기.

{{< cards cols="2" >}}
{{< card link="https://colab.research.google.com/github/keras-team/keras-io/blob/master/guides/ipynb/keras_tuner/tailor_the_search_space.ipynb" title="Colab" tag="Colab" tagType="warning">}}
{{< card link="https://github.com/keras-team/keras-io/blob/master/guides/keras_tuner/tailor_the_search_space.py" title="GitHub" tag="GitHub">}}
{{< /cards >}}

```python
!pip install keras-tuner -q
```

이 가이드에서는, `HyperModel` 코드를 직접 변경하지 않고, 검색 공간을 맞춤 설정하는 방법을 보여줍니다.
예를 들어, 일부 하이퍼파라미터만 튜닝하고 나머지는 고정하거나,
`optimizer`, `loss`, `metrics`와 같은 컴파일 인수를 재정의할 수 있습니다.

## 하이퍼파라미터의 기본값 {#the-default-value-of-a-hyperparameter}

검색 공간을 맞춤 설정하기 전에,
모든 하이퍼파라미터에는 기본값이 있다는 것을 알아야 합니다.
이 기본값은 검색 공간을 맞춤 설정할 때,
해당 하이퍼파라미터를 튜닝하지 않는 경우에 사용됩니다.

하이퍼파라미터를 등록할 때 `default` 인수를 사용하여, 기본값을 지정할 수 있습니다.

```python
hp.Int("units", min_value=32, max_value=128, step=32, default=64)
```

기본값을 지정하지 않으면, 하이퍼파라미터에는 기본값이 자동으로 지정됩니다.
(`Int`의 경우 `min_value`와 동일한 값)

다음 모델 빌딩 함수에서, `units` 하이퍼파라미터의 기본값을 64로 지정했습니다.

```python
import keras
from keras import layers
import keras_tuner
import numpy as np


def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Flatten())
    model.add(
        layers.Dense(
            units=hp.Int("units", min_value=32, max_value=128, step=32, default=64)
        )
    )
    if hp.Boolean("dropout"):
        model.add(layers.Dropout(rate=0.25))
    model.add(layers.Dense(units=10, activation="softmax"))
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
```

우리는 튜토리얼의 나머지 부분에서 새로운 검색 공간을 정의하지 않고,
하이퍼파라미터를 재정의하여 이 검색 공간을 재사용할 것입니다.

## 일부만 검색하고 나머지는 고정 {#search-a-few-and-fix-the-rest}

기존의 하이퍼모델이 있고, 일부 하이퍼파라미터만 검색하고 나머지는 고정하고 싶다면,
모델 빌딩 함수나 `HyperModel` 코드에서 변경할 필요가 없습니다.
튜너 생성자에서 `hyperparameters` 인수로 검색하려는 모든 하이퍼파라미터를 포함한,
`HyperParameters` 객체를 전달할 수 있습니다.
다른 하이퍼파라미터의 튜닝을 방지하려면, `tune_new_entries=False`를 지정하여,
나머지 하이퍼파라미터는 기본값을 사용하도록 설정합니다.

다음 예시에서 우리는 `learning_rate` 하이퍼파라미터만 튜닝하며, 그 타입과 값 범위를 변경했습니다.

```python
hp = keras_tuner.HyperParameters()

# `learning_rate` 파라미터를 사용자가 선택한 값으로 재정의
hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")

tuner = keras_tuner.RandomSearch(
    hypermodel=build_model,
    hyperparameters=hp,
    # 나열되지 않은 파라미터는 튜닝하지 않음
    tune_new_entries=False,
    objective="val_accuracy",
    max_trials=3,
    overwrite=True,
    directory="my_dir",
    project_name="search_a_few",
)

# 랜덤 데이터 생성
x_train = np.random.rand(100, 28, 28, 1)
y_train = np.random.randint(0, 10, (100, 1))
x_val = np.random.rand(20, 28, 28, 1)
y_val = np.random.randint(0, 10, (20, 1))

# 검색 실행
tuner.search(x_train, y_train, epochs=1, validation_data=(x_val, y_val))
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Trial 3 Complete [00h 00m 01s]
val_accuracy: 0.20000000298023224
```

```plain
Best val_accuracy So Far: 0.25
Total elapsed time: 00h 00m 03s
```

{{% /details %}}

하이퍼파라미터 검색 공간을 요약하면, 단 하나의 하이퍼파라미터만 볼 수 있습니다.

```python
tuner.search_space_summary()
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Search space summary
Default search space size: 1
learning_rate (Float)
{'default': 0.0001, 'conditions': [], 'min_value': 0.0001, 'max_value': 0.01, 'step': None, 'sampling': 'log'}
```

{{% /details %}}

## 일부를 고정하고 나머지를 튜닝 {#fix-a-few-and-tune-the-rest}

위 예시에서는 일부 하이퍼파라미터만 튜닝하고 나머지는 고정하는 방법을 보여주었습니다.
반대로, 일부 하이퍼파라미터만 고정하고 나머지를 모두 튜닝할 수도 있습니다.

다음 예시에서는 `learning_rate` 하이퍼파라미터의 값을 고정했습니다.
`Fixed` 항목을 포함한 `hyperparameters` 인수를 전달하고,
`tune_new_entries=True`로 설정하여 나머지 하이퍼파라미터를 튜닝할 수 있습니다.

```python
hp = keras_tuner.HyperParameters()
hp.Fixed("learning_rate", value=1e-4)

tuner = keras_tuner.RandomSearch(
    build_model,
    hyperparameters=hp,
    tune_new_entries=True,
    objective="val_accuracy",
    max_trials=3,
    overwrite=True,
    directory="my_dir",
    project_name="fix_a_few",
)

tuner.search(x_train, y_train, epochs=1, validation_data=(x_val, y_val))
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Trial 3 Complete [00h 00m 01s]
val_accuracy: 0.15000000596046448
```

```plain
Best val_accuracy So Far: 0.15000000596046448
Total elapsed time: 00h 00m 03s
```

{{% /details %}}

검색 공간을 요약하면, `learning_rate`는 고정된 것으로 표시되고,
나머지 하이퍼파라미터는 튜닝되고 있음을 확인할 수 있습니다.

```python
tuner.search_space_summary()
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Search space summary
Default search space size: 3
learning_rate (Fixed)
{'conditions': [], 'value': 0.0001}
units (Int)
{'default': 64, 'conditions': [], 'min_value': 32, 'max_value': 128, 'step': 32, 'sampling': 'linear'}
dropout (Boolean)
{'default': False, 'conditions': []}
```

{{% /details %}}

## 컴파일 인수 재정의 {#overriding-compilation-arguments}

기존 하이퍼모델에서 옵티마이저, 손실 함수, 또는 메트릭스를 변경하고 싶다면,
이러한 인수를 튜너 생성자에 전달하여 변경할 수 있습니다.

```python
tuner = keras_tuner.RandomSearch(
    build_model,
    optimizer=keras.optimizers.Adam(1e-3),
    loss="mse",
    metrics=[
        "sparse_categorical_crossentropy",
    ],
    objective="val_loss",
    max_trials=3,
    overwrite=True,
    directory="my_dir",
    project_name="override_compile",
)

tuner.search(x_train, y_train, epochs=1, validation_data=(x_val, y_val))
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Trial 3 Complete [00h 00m 01s]
val_loss: 29.39796257019043
```

```plain
Best val_loss So Far: 29.39630699157715
Total elapsed time: 00h 00m 04s
```

{{% /details %}}

최상의 모델을 얻으면, 손실 함수가 MSE로 변경된 것을 확인할 수 있습니다.

```python
tuner.get_best_models()[0].loss
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
/usr/local/python/3.10.13/lib/python3.10/site-packages/keras/src/saving/saving_lib.py:388: UserWarning: Skipping variable loading for optimizer 'adam', because it has 2 variables whereas the saved optimizer has 10 variables.
  trackable.load_own_variables(weights_store.get(inner_path))

'mse'
```

{{% /details %}}

## 미리 빌드된 하이퍼모델의 검색 공간 맞춤 설정 {#tailor-the-search-space-of-pre-build-hypermodels}

이 기술은 KerasTuner의 `HyperResNet`이나 `HyperXception`과 같은
미리 빌드된 모델에서도 사용할 수 있습니다.
그러나, 이러한 미리 빌드된 `HyperModel`에서
어떤 하이퍼파라미터가 있는지 확인하려면, 소스 코드를 읽어야 합니다.

다음 예시에서는, `HyperXception`의 `learning_rate`만 튜닝하고,
나머지 하이퍼파라미터는 모두 고정했습니다.
`HyperXception`의 기본 손실 함수는 `categorical_crossentropy`인데,
이는 라벨이 원-핫 인코딩된 데이터를 기대합니다.
우리의 정수형 라벨 데이터와 맞지 않으므로,
컴파일 인수에서 `loss`를 `sparse_categorical_crossentropy`로 재정의해야 합니다.

```python
hypermodel = keras_tuner.applications.HyperXception(input_shape=(28, 28, 1), classes=10)

hp = keras_tuner.HyperParameters()

# `learning_rate` 파라미터를 사용자가 선택한 값으로 재정의
hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])

tuner = keras_tuner.RandomSearch(
    hypermodel,
    hyperparameters=hp,
    # 나열되지 않은 파라미터는 튜닝하지 않음
    tune_new_entries=False,
    # 손실 함수 재정의
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
    objective="val_accuracy",
    max_trials=3,
    overwrite=True,
    directory="my_dir",
    project_name="helloworld",
)

# 검색 실행
tuner.search(x_train, y_train, epochs=1, validation_data=(x_val, y_val))
tuner.search_space_summary()
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Trial 3 Complete [00h 00m 19s]
val_accuracy: 0.15000000596046448
```

```plain
Best val_accuracy So Far: 0.20000000298023224
Total elapsed time: 00h 00m 58s
Search space summary
Default search space size: 1
learning_rate (Choice)
{'default': 0.01, 'conditions': [], 'values': [0.01, 0.001, 0.0001], 'ordered': True}
```

{{% /details %}}
