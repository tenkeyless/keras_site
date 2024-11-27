---
title: KerasNLP
toc: true
weight: 10
type: docs
---

{{< keras/original checkedAt="2024-11-19" >}}

KerasNLP는 TensorFlow, JAX, 또는 PyTorch와 네이티브로 작동하는 자연어 처리 라이브러리입니다.
Keras 3를 기반으로 구축된 이 모델, 레이어, 메트릭, 그리고 토크나이저는
어떤 프레임워크에서라도 트레이닝 및 직렬화할 수 있으며,
비용이 많이 드는 마이그레이션 없이도 재사용할 수 있습니다.

KerasNLP는 사용자의 전체 개발 주기를 지원합니다.
우리의 워크플로우는 모듈식 구성 요소로 이루어져 있으며,
즉시 사용 가능한 최첨단 사전 트레이닝된 가중치와 아키텍처를 제공하며,
더 많은 제어가 필요할 때 쉽게 커스터마이즈할 수 있습니다.

이 라이브러리는 Keras의 core API의 확장입니다.
모든 상위 모듈은 [`Layers`]({{< relref "/docs/api/layers" >}}) 또는
[`Models`]({{< relref "/docs/api/models" >}})로,
core Keras와 동일한 레벨의 완성도를 자랑합니다.
Keras에 익숙하다면, 이미 KerasNLP의 대부분을 이해한 것입니다.

[시작하기]({{< relref "/docs/guides/keras_nlp/getting_started" >}})를 확인하고 API를 배워보세요.
우리는 [기여](https://github.com/keras-team/keras-nlp/blob/master/CONTRIBUTING.md)를 환영합니다.

## 빠른 링크 {#quick-links}

- [KerasNLP API 참조]({{< relref "/docs/api/keras_nlp" >}})
- [KerasNLP GitHub](https://github.com/keras-team/keras-nlp)
- [사전 트레이닝된 모델 리스트]({{< relref "/docs/api/keras_nlp/models" >}})

## 가이드 {#guides}

- [KerasNLP 시작하기]({{< relref "/docs/guides/keras_nlp/getting_started" >}})
- [KerasNLP로 모델 업로드]({{< relref "/docs/guides/keras_nlp/upload" >}})
- [트랜스포머를 처음부터 사전 트레이닝하기]({{< relref "/docs/guides/keras_nlp/transformer_pretraining" >}})

## 예제 {#examples}

- [GPT-2 텍스트 생성]({{< relref "/docs/examples/generative/gpt2_text_generation_with_kerasnlp" >}})
- [LoRA를 사용한 GPT-2의 파라미터 효율적 파인 튜닝]({{< relref "/docs/examples/nlp/parameter_efficient_finetuning_of_gpt2_with_lora" >}})
- [시맨틱 유사성]({{< relref "/docs/examples/nlp/semantic_similarity_with_keras_nlp" >}})
- [Siamese RoBERTa 네트워크를 사용한 문장 임베딩]({{< relref "/docs/examples/nlp/sentence_embeddings_with_sbert" >}})
- [tf.distribute로 데이터 병렬 트레이닝]({{< relref "/docs/examples/nlp/data_parallel_training_with_keras_nlp" >}})
- [영어-스페인어 번역]({{< relref "/docs/examples/nlp/neural_machine_translation_with_keras_nlp" >}})
- [처음부터 GPT 텍스트 생성]({{< relref "/docs/examples/generative/text_generation_gpt" >}})
- [FNet을 사용한 텍스트 분류]({{< relref "/docs/examples/nlp/fnet_classification_with_keras_nlp" >}})

## 설치 {#installation}

KerasNLP는 Keras 2와 Keras 3 모두를 지원합니다.
JAX, TensorFlow, PyTorch와 함께 KerasNLP 모델과 레이어를 사용하려면 Keras 3을 권장합니다.

### Keras 2 설치 {#keras-2-installation}

최신 KerasNLP 릴리스를 Keras 2와 함께 설치하려면 다음 명령어를 실행하세요:

```shell
pip install --upgrade keras-nlp
```

### Keras 3 설치 {#keras-3-installation}

Keras 3과 KerasNLP를 설치하는 방법은 두 가지가 있습니다.
반드시 KerasNLP를 먼저 설치한 **이후**, Keras 3을 설치하여 stable 버전을 사용할 수 있습니다.
이는 TensorFlow가 Keras 2에 고정되어 있는 동안 필요한 임시 단계이며,
TensorFlow 2.16 이후에는 더 이상 필요하지 않을 것입니다.

```shell
pip install --upgrade keras-nlp
pip install --upgrade keras
```

KerasNLP와 Keras의 최신 nightly 빌드를 설치하려면, nightly 패키지를 사용할 수 있습니다.

```shell
pip install --upgrade keras-nlp-nightly
```

**참고:** Keras 3는 TensorFlow 2.14 이하 버전에서는 작동하지 않습니다.

자세한 설치 정보와 다양한 프레임워크와의 호환성에 대한 내용은,
[Keras 시작하기]({{< relref "/docs/getting_started" >}})에서 확인하세요.

## 빠른 시작 {#quickstart}

BERT를 사용한 소규모 감정 분석 작업에 대해,
[`keras_nlp.models`]({{< relref "/docs/api/keras_nlp/models" >}}) API를 사용하여 파인 튜닝하기:

```python
import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # 또는 "jax"나 "torch"!

import keras_nlp
import tensorflow_datasets as tfds

imdb_train, imdb_test = tfds.load(
    "imdb_reviews",
    split=["train", "test"],
    as_supervised=True,
    batch_size=16,
)
# BERT 모델 로드.
classifier = keras_nlp.models.BertClassifier.from_preset(
    "bert_base_en_uncased",
    num_classes=2,
)
# IMDb 영화 리뷰로 파인 튜닝.
classifier.fit(imdb_train, validation_data=imdb_test)
# 두 개의 새로운 예시 예측.
classifier.predict(["What an amazing movie!", "A total waste of my time."])
```

## 호환성 {#compatibility}

우리는 [Semantic Versioning](https://semver.org/)을 따르며,
KerasNLP 구성 요소로 작성된 코드와 저장된 모델에 대해 하위 호환성을 보장할 계획입니다.
그러나 현재 `0.y.z`의 사전 릴리스 개발 단계에서는 언제든지 호환성이 깨질 수 있으며,
API는 stable로 간주되어서는 안 됩니다.

## 면책 조항 {#disclaimer}

KerasNLP는 `keras_nlp.models` API를 통해 사전 트레이닝된 모델에 접근할 수 있게 합니다.
이 사전 트레이닝된 모델은 어떠한 종류의 보증이나 조건 없이 "있는 그대로" 제공됩니다.
다음의 기본 모델은 제3자에 의해 제공되며, 별도의 라이선스가 적용됩니다:
BART, DeBERTa, DistilBERT, GPT-2, OPT, RoBERTa, Whisper, XLM-RoBERTa.

## KerasNLP 인용 {#citing-kerasnlp}

KerasNLP가 연구에 도움이 되었다면, 인용을 해주시면 감사하겠습니다. 아래는 BibTeX 항목입니다:

```latex
@misc{kerasnlp2022,
  title={KerasNLP},
  author={Watson, Matthew, and Qian, Chen, and Bischof, Jonathan and Chollet,
  Fran\c{c}ois and others},
  year={2022},
  howpublished={\url{https://github.com/keras-team/keras-nlp}},
}
```
