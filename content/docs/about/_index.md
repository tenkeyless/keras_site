---
title: Keras 3에 대하여
linkTitle: Keras에 대하여
toc: true
weight: 1
type: docs
cascade:
  type: docs
---

> - 원본 링크 : [https://keras.io/about/](https://keras.io/about/)
> - 최종 확인 : 2024-03-28

# Keras 3에 대하여

<!-- # About Keras 3 -->

Keras는 Python으로 작성된 딥러닝 API로, [JAX](https://jax.readthedocs.io/), [TensorFlow](https://github.com/tensorflow/tensorflow) 또는 [PyTorch](https://pytorch.org/) 위에서 실행할 수 있습니다.

Keras는:

- **단순성** - 단순하지만, 단순하지 않습니다. Keras는 개발자의 *인지적 부하(cognitive load)*를 줄여, 개발자가 정말 중요한 문제에 집중할 수 있도록 합니다.
- **유연성** - Keras는 *복잡성의 점진적 공개*라는 원칙을 채택합니다. 간단한 워크플로는 빠르고 쉬워야 하며, 한편으로는 이미 학습한 내용을 바탕으로 명확한 경로를 통해 임의로 고급 워크플로를 _만들 수 있어야_ 합니다.
- **강력함** - Keras는 업계 최고 수준의 성능과 확장성을 제공하며, NASA, YouTube, Waymo 등의 조직에서 사용하고 있습니다.

## Keras 3는 멀티 프레임워크 딥러닝 API입니다.

멀티 프레임워크 API인 Keras는 JAX, TensorFlow, PyTorch 등의 프레임워크와 호환되는 모듈식 컴포넌트를 개발하는데 사용할 수 있습니다.

이 접근 방식에는 몇 가지 주요 이점이 있습니다:

- **모델에 대해 항상 최상의 성능을 얻을 수 있습니다.** 우리의 벤치마크 결과, JAX는 일반적으로 GPU, TPU, CPU에서 최고의 트레이닝 및 추론 성능을 제공하지만, 비-XLA TensorFlow가 GPU에서 더 빠른 경우도 있기 때문에, 결과는 모델마다 다릅니다. _당신의 코드를 변경하지 않고도_ 모델에 가장 적합한 성능을 제공하는 백엔드를 동적으로 선택할 수 있으므로, 항상 최고의 효율로 트레이닝하고 서비스를 제공할 수 있습니다.
- **모델을 위해 사용 가능한 에코시스템 표면을 극대화하세요.** 모든 Keras 모델은 PyTorch `Module`로 인스턴스화할 수 있고, TensorFlow `SavedModel`로 내보낼 수 있으며, 상태없는 JAX 함수로 인스턴스화할 수 있습니다. 즉, PyTorch 에코시스템 패키지, 모든 범위의 TensorFlow 배포 및 프로덕션 도구, JAX 대규모 TPU 트레이닝 인프라와 함께, Keras 모델을 사용할 수 있습니다. Keras API를 사용해 하나의 `model.py`를 작성하고, ML 세계에서 제공하는 모든 것을 이용할 수 있습니다.
- **오픈 소스 모델 릴리스의 배포를 극대화하세요.** 사전 트레이닝된 모델을 릴리스하고 싶으신가요? 가능한 한 많은 사람이 사용할 수 있기를 원하시나요? 순수 TensorFlow 또는 PyTorch로 구현하면, 시장의 약 절반이 사용할 수 있습니다. Keras로 구현하면, 선택한 프레임워크에 관계없이(Keras 사용자가 아니더라도) 누구나 즉시 사용할 수 있습니다. 추가 개발 비용 없이 두 배의 효과를 얻을 수 있습니다.
- **모든 소스의 데이터 파이프라인을 사용하세요.** Keras `fit()`/`evaluate()`/`predict()` 루틴은 당신이 사용 중인 백엔드에 관계없이 [`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset) 객체, PyTorch `DataLoader` 객체, NumPy 배열, Pandas 데이터프레임과 호환이 가능합니다. PyTorch `DataLoader`에서 Keras + TensorFlow 모델을 트레이닝하거나, [`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset)에 대해 Keras + PyTorch 모델을 트레이닝할 수 있습니다.

## Keras와의 첫 만남

Keras의 핵심 데이터 구조는 **레이어(layers)**와 **모델(models)**입니다. 가장 간단한 모델 유형은 선형 레이어 스택인 [`Sequential` 모델](/docs/guides/sequential_model)입니다. 더 복잡한 아키텍처의 경우, 임의의 레이어 그래프를 만들 수 있는 [Keras Functional API](/docs/guides/functional_api)를 사용하거나, [서브클래싱을 통해 완전히 처음부터 모델 작성](/docs/guides/making_new_layers_and_models_via_subclassing) 방법을 사용해야 합니다.

다음은 `Sequential` 모델입니다:

```python
import keras

model = keras.Sequential()
```

레이어를 쌓는 것은 `.add()`를 사용하면 간단합니다:

```python
from keras import layers

model.add(layers.Dense(units=64, activation='relu'))
model.add(layers.Dense(units=10, activation='softmax'))
```

모델이 좋아 보이면, `.compile()`을 사용하여 학습 프로세스를 구성합니다:

```python
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
```

필요한 경우, 옵티마이저를 추가로 구성할 수 있습니다. Keras의 철학은 단순한 것을 단순하게 유지하면서, 사용자가 필요할 때 완전히 제어할 수 있도록 하는 것입니다. (궁극적인 제어는 서브클래싱을 통해 소스 코드를 쉽게 확장할 수 있는 것입니다)

```python
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True))
```

이제 트레이닝 데이터를 배치에서 반복할 수 있습니다:

```python
# x_train 및 y_train은 Numpy 배열
model.fit(x_train, y_train, epochs=5, batch_size=32)
```

테스트 손실과 지표를 한 줄로 평가하세요:

```python
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
```

또는 새로운 데이터에 대한 예측을 생성할 수도 있습니다:

```python
classes = model.predict(x_test, batch_size=128)
```

방금 보신 것은 Keras를 사용하는 가장 기본적인 방법입니다.

하지만, Keras는 최첨단 연구 아이디어를 반복하는 데 적합한 매우 유연한 프레임워크이기도 합니다. Keras는 **복잡성의 점진적 공개**라는 원칙을 따르기 때문에, 쉽게 시작할 수 있지만, 각 단계마다 점진적인 학습만 거치면, 임의의 고급 사용 사례도 처리할 수 있습니다.

위에서 몇 줄로 간단한 신경망을 트레이닝하고 평가할 수 있었던 것과 마찬가지로, Keras를 사용하여 새로운 트레이닝 절차나 최첨단 모델 아키텍처를 빠르게 개발할 수 있습니다.

다음은 사용자 정의 Keras 레이어의 예시이며, JAX, TensorFlow 또는 PyTorch의 저수준(low-level) 워크플로우에서 서로 바꿔서(interchangeably) 사용할 수 있습니다:

```python
import keras
from keras import ops

class TokenAndPositionEmbedding(keras.Layer):
    def __init__(self, max_length, vocab_size, embed_dim):
        super().__init__()
        self.token_embed = self.add_weight(
            shape=(vocab_size, embed_dim),
            initializer="random_uniform",
            trainable=True,
        )
        self.position_embed = self.add_weight(
            shape=(max_length, embed_dim),
            initializer="random_uniform",
            trainable=True,
        )

    def call(self, token_ids):
        # 위치 임베딩
        length = token_ids.shape[-1]
        positions = ops.arange(0, length, dtype="int32")
        positions_vectors = ops.take(self.position_embed, positions, axis=0)
        # 토큰 임베딩
        token_ids = ops.cast(token_ids, dtype="int32")
        token_vectors = ops.take(self.token_embed, token_ids, axis=0)
        # 둘을 합하기
        embed = token_vectors + positions_vectors
        # 임베딩을 Normalize
        power_sum = ops.sum(ops.square(embed), axis=-1, keepdims=True)
        return embed / ops.sqrt(ops.maximum(power_sum, 1e-7))
```

Keras에 대한 자세한 튜토리얼은 여기에서 확인할 수 있습니다:

- [엔지니어를 위한 Keras 소개](/docs/getting_started/intro_to_keras_for_engineers)
- [개발자 가이드](/docs/guides/)

## 지원

[Keras Google 그룹](https://groups.google.com/forum/#!forum/keras-users)에서 질문하고 개발 토론에 참여할 수 있습니다.

[GitHub 이슈](https://github.com/keras-team/keras/issues)에 **버그 리포트 및 기능 요청**(이것만 가능)을 게시할 수도 있습니다. [가이드라인](https://github.com/keras-team/keras-io/blob/master/templates/contributing.md)을 먼저 읽어주세요.

## 왜 Keras라는 이름인가요?

Keras(κέρας)는 고대 그리스어로 *horn*을 의미합니다. 고대 그리스와 라틴 문학에 등장하는 문학적 이미지로, *오디세이*에서 처음 등장하며, 꿈의 정령(_Oneiroi_, 단수형 _Oneiros_)을 거짓 환상으로 몽상가를 속이는(deceive) 자, ivory의 문을 통해 지구에 도착하는 자, 다가올 미래를 알리는 자, horn의 문을 통해 도착하는 자로 나누고 있습니다. κέρας(뿔)/κραίνω(성취하다), ἐλέφας(상아)/ἐλεφαίρομαι(속이다)라는 단어에서 유래한 말입니다.

Keras는 처음에 프로젝트 ONEIROS(Open-ended Neuro-Electronic Intelligent Robot Operating System, 개방형 신경 전자 지능형 로봇 운영체제)의 연구 노력의 일환으로 개발되었습니다.

> \*"Oneiroi는 우리가 풀지 못한 수수께끼의 존재로, 어떤 이야기를 들려줄지 누가 확신할 수 있을까요? 인간이 찾는 모든 것이 이루어지지는 않습니다. 덧없는 Oneiroi에게 통로를 제공하는 두 개의 문이 있는데, 하나는 뿔로, 다른 하나는 상아로 만들어져 있습니다. 톱질한 상아를 통과하는 Oneiroi는, 성취되지 않을 메시지를 담고 있는 속임수이고, **광택이 나는 뿔을 통해 나오는 Oneiroi는 그것을 보는 사람들에게 성취될 진실을 담고 있다."\*** 호머, 오디세이 19. 562쪽(셰링 번역).
