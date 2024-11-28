---
title: KerasHub
toc: true
weight: 8
type: docs
---

{{< keras/original checkedAt="2024-11-19" >}}

**KerasHub**는 간단하고 유연하며 빠른 것을 목표로 하는 사전 트레이닝된 모델링 라이브러리입니다.
이 라이브러리는 인기 있는 모델 아키텍처의 [Keras 3]({{< relref "/docs/keras_3" >}}) 구현을 제공하며,
[Kaggle Models](https://kaggle.com/models/)에서 사용할 수 있는 사전 트레이닝된 체크포인트 컬렉션과 함께 제공됩니다.
모델은 TensorFlow, Jax, Torch 백엔드에서 트레이닝과 추론에 모두 사용할 수 있습니다.

KerasHub는 core Keras API의 확장입니다.
KerasHub 구성 요소는 [`Layers`]({{< relref "/docs/api/layers" >}}) 및
[`Models`]({{< relref "/docs/api/models" >}})로 제공됩니다.
Keras에 익숙하다면 축하합니다! 이미 KerasHub의 대부분을 이해하고 계십니다.

API 학습을 시작하려면 [시작 가이드]({{< relref "/docs/guides/keras_hub/getting_started" >}})를 참조하세요.
[기여](https://github.com/keras-team/keras-hub/issues/1835)를 환영합니다.

## 빠른 링크 {#quick-links}

- [KerasHub API 참조]({{< relref "/docs/api/keras_hub" >}})
- [KerasHub GitHub](https://github.com/keras-team/keras-hub)
- [Kaggle에 있는 KerasHub 모델](https://www.kaggle.com/organizations/keras/models)
- [사용 가능한 사전 트레이닝된 모델 목록]({{< relref "/docs/api/keras_hub/models" >}})

## 가이드 {#guides}

- [KerasHub 시작하기]({{< relref "/docs/guides/keras_hub/getting_started" >}})
- [KerasHub를 사용한 분류]({{< relref "/docs/guides/keras_hub/classification_with_keras_hub" >}})
- [KerasHub에서 Segment Anything]({{< relref "/docs/guides/keras_hub/segment_anything_in_keras_hub" >}})
- [KerasHub를 사용한 시맨틱 세그멘테이션]({{< relref "/docs/guides/keras_hub/semantic_segmentation_deeplab_v3" >}})
- [KerasHub의 Stable Diffusion 3]({{< relref "/docs/guides/keras_hub/stable_diffusion_3_in_keras_hub" >}})
- [KerasHub를 사용하여 처음부터 Transformer 사전 트레이닝]({{< relref "/docs/guides/keras_hub/transformer_pretraining" >}})
- [KerasHub를 사용하여 모델 업로드]({{< relref "/docs/guides/keras_hub/upload" >}})

## 설치 {#installation}

Keras 3와 함께 최신 KerasHub 릴리스를 설치하려면, 다음을 실행하기만 하면 됩니다.

```shell
pip install --upgrade keras-hub
```

KerasHub와 Keras의 최신 nightly 변경 사항을 설치하려면, nightly 패키지를 사용하면 됩니다.

```shell
pip install --upgrade keras-hub-nightly
```

현재 KerasHub를 설치하면,
항상 TensorFlow를 가져와 [`tf.data`](https://www.tensorflow.org/api_docs/python/tf/data) API를 사용하여 전처리합니다.
[`tf.data`](https://www.tensorflow.org/api_docs/python/tf/data)로 전처리하더라도 모든 백엔드에서 트레이닝이 가능합니다.

Keras 3 설치 및 다양한 프레임워크와의 호환성에 대한 자세한 내용은
[Keras 시작하기]({{< relref "/docs/getting_started" >}})를 참조하세요.

**참고:** TF 2.16은 기본적으로 Keras 3를 패키징하므로, TensorFlow 2.16 이상과 함께 KerasHub를 사용하는 것이 좋습니다.

## 빠른 시작 {#quickstart}

아래는 ResNet을 사용하여 이미지를 예측하고, BERT를 사용하여 분류기를 트레이닝하는 간단한 예입니다.

```python
import os
os.environ["KERAS_BACKEND"] = "jax"  # 또는 "tensorflow"나 "torch"!

import keras
import keras_hub
import numpy as np
import tensorflow_datasets as tfds

# ResNet 모델을 불러옵니다.
classifier = keras_hub.models.ImageClassifier.from_preset(
    "resnet_50_imagenet",
    activation="softmax",
)
# 단일 이미지에 대해 레이블을 예측합니다.
image_url = "https://upload.wikimedia.org/wikipedia/commons/a/aa/California_quail.jpg"
image_path = keras.utils.get_file(origin=image_url)
image = keras.utils.load_img(image_path)
batch = np.array([image])
preds = classifier.predict(batch)
print(keras_hub.utils.decode_imagenet_predictions(preds))

# BERT 모델을 불러옵니다.
classifier = keras_hub.models.BertClassifier.from_preset(
    "bert_base_en_uncased",
    activation="softmax",
    num_classes=2,
)

# IMDb 영화 리뷰에 대해 파인 튜닝합니다.
imdb_train, imdb_test = tfds.load(
    "imdb_reviews",
    split=["train", "test"],
    as_supervised=True,
    batch_size=16,
)
classifier.fit(imdb_train, validation_data=imdb_test)
# 새로운 두 개의 예제를 예측합니다.
preds = classifier.predict(
    ["What an amazing movie!", "A total waste of my time."]
)
print(preds)
```

## 호환성 {#compatibility}

우리는 [Semantic Versioning](https://semver.org/)을 따르며,
코드와 구성 요소로 빌드된 저장된 모델에 대해 하위 호환성을 보장할 계획입니다.
그러나 현재 `0.y.z`의 사전 릴리스 개발 단계에서는 언제든지 호환성이 깨질 수 있으며,
API는 stable로 간주되어서는 안 됩니다.

## 면책 조항 {#disclaimer}

KerasHub는 `keras_hub.models` API를 통해 사전 트레이닝된 모델에 대한 액세스를 제공합니다.
이러한 사전 트레이닝된 모델은 어떠한 종류의 보증이나 조건 없이 "있는 그대로" 제공됩니다.

## KerasHub 인용 {#citing-kerashub}

KerasHub가 연구에 도움이 되었다면, 인용을 해주시면 감사하겠습니다. 아래는 BibTeX 항목입니다:

```latex
@misc{kerashub2024,
  title={KerasHub},
  author={Watson, Matthew, and  Chollet, Fran\c{c}ois and Sreepathihalli,
  Divyashree, and Saadat, Samaneh and Sampath, Ramesh, and Rasskin, Gabriel and
  and Zhu, Scott and Singh, Varun and Wood, Luke and Tan, Zhenyu and Stenbit,
  Ian and Qian, Chen, and Bischof, Jonathan and others},
  year={2024},
  howpublished={\url{https://github.com/keras-team/keras-hub}},
}
```
