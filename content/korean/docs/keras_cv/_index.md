---
title: KerasCV
toc: true
weight: 9
type: docs
---

{{< keras/original checkedAt="2024-11-19" >}}

KerasCV는 TensorFlow, JAX 또는 PyTorch에서 네이티브하게 작동하는 모듈식 컴퓨터 비전 구성 요소 라이브러리입니다.
Keras 3 기반으로 구축된 이러한 모델, 레이어, 메트릭, 콜백 등은,
모든 프레임워크에서 트레이닝되고 직렬화할 수 있으며,
비용이 많이 드는 마이그레이션 없이 다른 프레임워크에서 재사용할 수 있습니다.

KerasCV는 Keras API의 수평적 확장으로 이해할 수 있습니다.
구성 요소는 core Keras에 추가하기에는 너무 특수화된 새로운 퍼스트파티 Keras 객체입니다.
이들은 core Keras API와 동일한 레벨의 세련미와 이전 버전과의 호환성을 보장받으며, Keras 팀에서 유지 관리합니다.

우리의 API는 데이터 보강, 분류, 객체 감지, 세그멘테이션, 이미지 생성 등과 같은 일반적인 컴퓨터 비전 작업을 지원합니다.
응용 컴퓨터 비전 엔지니어는 KerasCV를 활용하여,
이러한 모든 일반적인 작업에 대한 프로덕션 등급의 최첨단 트레이닝 및 추론 파이프라인을 빠르게 조립할 수 있습니다.

![gif](/images/keras-cv-augmentations.gif)

## 빠른 링크

- [사용 가능한 모델 및 사전 설정 목록]({{< relref "/docs/api/keras_cv/models" >}})
- [개발자 가이드]({{< relref "/docs/guides/keras_cv" >}})
- [기여 가이드](https://github.com/keras-team/keras-cv/blob/master/CONTRIBUTING.md)
- [API 디자인 가이드라인](https://github.com/keras-team/keras-cv/blob/master/API_DESIGN.md)

## 설치

KerasCV는 Keras 2와 Keras 3을 모두 지원합니다.
모든 신규 사용자에게는 Keras 3을 권장합니다.
KerasCV 모델과 레이어를 JAX, TensorFlow, PyTorch에서 사용할 수 있기 때문입니다.

### Keras 2 설치

Keras 2에서 최신 KerasCV 릴리스를 설치하려면, 다음을 실행하기만 하면 됩니다.

```shell
pip install --upgrade keras-cv tensorflow
```

### Keras 3 설치

현재 KerasCV로 Keras 3를 설치하는 방법은 두 가지가 있습니다.
KerasCV와 Keras 3의 안정적인 버전을 설치하려면, 반드시 KerasCV를 설치한 **이후에** Keras 3를 설치해야 합니다.
이는 TensorFlow가 Keras 2에 고정되어 있는 동안의 임시 단계이며,
TensorFlow 2.16 이후에는 더 이상 필요하지 않습니다.

```shell
pip install --upgrade keras-cv tensorflow
pip install --upgrade keras
```

KerasCV 및 Keras의 최신 변경 사항을 nightly 설치하려면, nightly 패키지를 사용하면 됩니다.

```shell
pip install --upgrade keras-cv-nightly tf-nightly
```

**참고:** Keras 3은 TensorFlow 2.14 이하에서는 작동하지 않습니다.

Keras를 일반적으로 설치하는 방법과 다양한 프레임워크와의 호환성에 대한 자세한 내용은,
[Keras 시작하기]({{< relref "/docs/getting_started" >}})를 참조하세요.

## 빠른 시작

```python
import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # 또는 "jax" 또는 "torch"!

import tensorflow as tf
import keras_cv
import tensorflow_datasets as tfds
import keras

# 보강을 통한 전처리 파이프라인 생성
BATCH_SIZE = 16
NUM_CLASSES = 3
augmenter = keras_cv.layers.Augmenter(
    [
        keras_cv.layers.RandomFlip(),
        keras_cv.layers.RandAugment(value_range=(0, 255)),
        keras_cv.layers.CutMix(),
    ],
)

def preprocess_data(images, labels, augment=False):
    labels = tf.one_hot(labels, NUM_CLASSES)
    inputs = {"images": images, "labels": labels}
    outputs = inputs
    if augment:
        outputs = augmenter(outputs)
    return outputs['images'], outputs['labels']

train_dataset, test_dataset = tfds.load(
    'rock_paper_scissors',
    as_supervised=True,
    split=['train', 'test'],
)
train_dataset = train_dataset.batch(BATCH_SIZE).map(
    lambda x, y: preprocess_data(x, y, augment=True),
        num_parallel_calls=tf.data.AUTOTUNE).prefetch(
            tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).map(
    preprocess_data, num_parallel_calls=tf.data.AUTOTUNE).prefetch(
        tf.data.AUTOTUNE)

# 사전 트레이닝된 백본을 사용하여 모델 생성
backbone = keras_cv.models.EfficientNetV2Backbone.from_preset(
    "efficientnetv2_b0_imagenet"
)
model = keras_cv.models.ImageClassifier(
    backbone=backbone,
    num_classes=NUM_CLASSES,
    activation="softmax",
)
model.compile(
    loss='categorical_crossentropy',
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    metrics=['accuracy']
)

# 모델 트레이닝
model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=8,
)
```

## 면책 조항

KerasCV는 `keras_cv.models` API를 통해 사전 트레이닝된 모델에 대한 액세스를 제공합니다.
이러한 사전 트레이닝된 모델은 어떠한 종류의 보증이나 조건 없이 "있는 그대로" 제공됩니다.
다음 기본 모델은 타사에서 제공하며, 별도의 라이선스가 적용됩니다. StableDiffusion, Vision Transfomer

## KerasCV 인용

KerasCV가 당신의 연구에 도움이 된다면, 당신의 인용에 감사드립니다. BibTeX 항목은 다음과 같습니다.

```latex
@misc{wood2022kerascv,
  title={KerasCV},
  author={Wood, Luke and Tan, Zhenyu and Stenbit, Ian and Bischof, Jonathan and Zhu, Scott and Chollet, Fran\c{c}ois and Sreepathihalli, Divyashree and Sampath, Ramesh and others},
  year={2022},
  howpublished={\url{https://github.com/keras-team/keras-cv}},
}
```
