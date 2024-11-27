---
title: KerasTuner
toc: true
weight: 7
type: docs
---

{{< keras/original checkedAt="2024-11-19" >}}

KerasTuner는 하이퍼파라미터 탐색의 문제를 해결하는,
사용하기 쉽고 확장 가능한 하이퍼파라미터 최적화 프레임워크입니다.
define-by-run 구문을 사용해 탐색 공간을 쉽게 설정하고,
다양한 검색 알고리즘을 활용하여 모델에 가장 적합한 하이퍼파라미터 값을 찾아낼 수 있습니다.
KerasTuner는 기본적으로 베이지안 최적화, 하이퍼밴드, 그리고 랜덤 서치 알고리즘을 제공하며,
연구자들이 새로운 탐색 알고리즘을 실험할 수 있도록 확장하기 쉽게 설계되었습니다.

## 빠른 링크

- [KerasTuner 시작하기]({{< relref "/docs/guides/keras_tuner/getting_started" >}})
- [KerasTuner 개발자 가이드]({{< relref "/docs/guides/keras_tuner" >}})
- [KerasTuner API 참조]({{< relref "/docs/api/keras_tuner" >}})
- [KerasTuner GitHub](https://github.com/keras-team/keras-tuner)

## 설치

최신 릴리스를 설치하세요:

```python
pip install keras-tuner --upgrade
```

다른 버전은 [GitHub 저장소](https://github.com/keras-team/keras-tuner)에서 확인할 수 있습니다.

## 빠른 소개

KerasTuner와 TensorFlow를 임포트하세요:

```python
import keras_tuner
import keras
```

모델을 생성하고 반환하는 함수를 작성하세요.
모델 생성 시 `hp` 인자를 사용하여 하이퍼파라미터를 정의합니다.

```python
def build_model(hp):
  model = keras.Sequential()
  model.add(keras.layers.Dense(
      hp.Choice('units', [8, 16, 32]),
      activation='relu'))
  model.add(keras.layers.Dense(1, activation='relu'))
  model.compile(loss='mse')
  return model
```

튜너를 초기화하세요. (여기서는 `RandomSearch`)
`objective`를 사용해 최적의 모델을 선택할 목표를 지정하고,
`max_trials`를 사용해 시도할 모델의 수를 설정합니다.

```python
tuner = keras_tuner.RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=5)
```

탐색을 시작하고 최적의 모델을 가져오세요:

```python
tuner.search(x_train, y_train, epochs=5, validation_data=(x_val, y_val))
best_model = tuner.get_best_models()[0]
```

KerasTuner에 대해 더 알고 싶다면,
[이 시작 가이드]({{< relref "/docs/guides/keras_tuner/getting_started" >}})를 확인하세요.

## KerasTuner 인용

KerasTuner가 연구에 도움이 되었다면, 인용해 주시면 감사하겠습니다. 아래는 BibTeX 항목입니다:

```latex
@misc{omalley2019kerastuner,
    title        = {KerasTuner},
    author       = {O'Malley, Tom and Bursztein, Elie and Long, James and Chollet, Fran\c{c}ois and Jin, Haifeng and Invernizzi, Luca and others},
    year         = 2019,
    howpublished = {\url{https://github.com/keras-team/keras-tuner}}
}
```
