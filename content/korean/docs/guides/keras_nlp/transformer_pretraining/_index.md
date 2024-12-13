---
title: KerasNLP로 처음부터 트랜스포머 사전 트레이닝
linkTitle: 트랜스포머 사전 트레이닝
toc: true
weight: 2
type: docs
---

{{< keras/original checkedAt="2024-11-19" >}}

**{{< t f_author >}}** [Matthew Watson](https://github.com/mattdangerw/)  
**{{< t f_date_created >}}** 2022/04/18  
**{{< t f_last_modified >}}** 2023/07/15  
**{{< t f_description >}}** KerasNLP를 사용하여 처음부터 트랜스포머 모델을 트레이닝하는 방법.

{{< cards cols="2" >}}
{{< card link="https://colab.research.google.com/github/keras-team/keras-io/blob/master/guides/ipynb/keras_nlp/transformer_pretraining.ipynb" title="Colab" tag="Colab" tagType="warning">}}
{{< card link="https://github.com/keras-team/keras-io/blob/master/guides/keras_nlp/transformer_pretraining.py" title="GitHub" tag="GitHub">}}
{{< /cards >}}

KerasNLP는 최신 텍스트 처리 모델을 쉽게 구축할 수 있도록 설계되었습니다.
이 가이드에서는, 라이브러리 구성 요소가 처음부터 트랜스포머 모델을 사전 트레이닝하고,
미세 조정하는 과정을 얼마나 간소화하는지 보여줍니다.

이 가이드는 세 가지 부분으로 나누어져 있습니다:

1.  _셋업_, 작업 정의 및 베이스라인 설정.
2.  트랜스포머 모델 _사전 트레이닝_.
3.  분류 작업에 대한 트랜스포머 모델 _미세 조정_.

## 셋업 {#setup}

다음 가이드는 `tensorflow`, `jax` 또는 `torch`에서 작동하는 Keras 3을 사용합니다.
아래에서는 특히 빠른 트레이닝 단계를 제공하는 `jax` 백엔드를 선택했지만,
다른 백엔드도 자유롭게 사용할 수 있습니다.

```python
!pip install -q --upgrade keras-nlp
!pip install -q --upgrade keras  # Keras 3으로 업그레이드.
```

```python
import os

os.environ["KERAS_BACKEND"] = "jax"  # 또는 "tensorflow" 또는 "torch"


import keras_nlp
import tensorflow as tf
import keras
```

다음으로, 두 개의 데이터셋을 다운로드합니다.

- [SST-2](https://paperswithcode.com/sota/sentiment-analysis-on-sst-2-binary):
  - 텍스트 분류 데이터셋으로, 우리의 "최종 목표"입니다.
  - 이 데이터셋은 종종 언어 모델의 성능을 벤치마킹하는 데 사용됩니다.
- [WikiText-103](https://paperswithcode.com/dataset/wikitext-103):
  - 영어 위키피디아의 주요 기사로 구성된 중간 크기의 컬렉션으로, 사전 트레이닝에 사용할 것입니다.

마지막으로, 이 가이드 후반에서 사용할 서브워드 토크나이징(sub-word tokenization)을 위해
WordPiece 어휘를 다운로드합니다.

```python
# 사전 트레이닝 데이터 다운로드.
keras.utils.get_file(
    origin="https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip",
    extract=True,
)
wiki_dir = os.path.expanduser("~/.keras/datasets/wikitext-103-raw/")

# 미세 조정 데이터 다운로드.
keras.utils.get_file(
    origin="https://dl.fbaipublicfiles.com/glue/data/SST-2.zip",
    extract=True,
)
sst_dir = os.path.expanduser("~/.keras/datasets/SST-2/")

# 어휘 데이터 다운로드.
vocab_file = keras.utils.get_file(
    origin="https://storage.googleapis.com/tensorflow/keras-nlp/examples/bert/bert_vocab_uncased.txt",
)
```

다음으로, 트레이닝 중에 사용할 하이퍼파라미터를 정의합니다.

```python
# 전처리 파라미터.
PRETRAINING_BATCH_SIZE = 128
FINETUNING_BATCH_SIZE = 32
SEQ_LENGTH = 128
MASK_RATE = 0.25
PREDICTIONS_PER_SEQ = 32

# 모델 파라미터.
NUM_LAYERS = 3
MODEL_DIM = 256
INTERMEDIATE_DIM = 512
NUM_HEADS = 4
DROPOUT = 0.1
NORM_EPSILON = 1e-5

# 트레이닝 파라미터.
PRETRAINING_LEARNING_RATE = 5e-4
PRETRAINING_EPOCHS = 8
FINETUNING_LEARNING_RATE = 5e-5
FINETUNING_EPOCHS = 3
```

### 데이터 로드 {#load-data}

[tf.data](https://www.tensorflow.org/guide/data)를 사용하여 데이터를 로드합니다.
이를 통해 텍스트를 토큰화하고 전처리하는 입력 파이프라인을 정의할 수 있습니다.

```python
# SST-2 데이터 로드.
sst_train_ds = tf.data.experimental.CsvDataset(
    sst_dir + "train.tsv", [tf.string, tf.int32], header=True, field_delim="\t"
).batch(FINETUNING_BATCH_SIZE)
sst_val_ds = tf.data.experimental.CsvDataset(
    sst_dir + "dev.tsv", [tf.string, tf.int32], header=True, field_delim="\t"
).batch(FINETUNING_BATCH_SIZE)

# wikitext-103 데이터 로드 및 짧은 줄 필터링.
wiki_train_ds = (
    tf.data.TextLineDataset(wiki_dir + "wiki.train.raw")
    .filter(lambda x: tf.strings.length(x) > 100)
    .batch(PRETRAINING_BATCH_SIZE)
)
wiki_val_ds = (
    tf.data.TextLineDataset(wiki_dir + "wiki.valid.raw")
    .filter(lambda x: tf.strings.length(x) > 100)
    .batch(PRETRAINING_BATCH_SIZE)
)

# sst-2 데이터셋을 살펴봅니다.
print(sst_train_ds.unbatch().batch(4).take(1).get_single_element())
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
(<tf.Tensor: shape=(4,), dtype=string, numpy=
array([b'hide new secretions from the parental units ',
       b'contains no wit , only labored gags ',
       b'that loves its characters and communicates something rather beautiful about human nature ',
       b'remains utterly satisfied to remain the same throughout '],
      dtype=object)>, <tf.Tensor: shape=(4,), dtype=int32, numpy=array([0, 0, 1, 0], dtype=int32)>)
```

{{% /details %}}

`SST-2` 데이터셋에는 영화 리뷰 텍스트의 짧은 문장이 포함되어 있습니다.
우리의 목표는 해당 문장의 감정을 예측하는 것입니다.
레이블이 1이면 긍정적인 감정을, 0이면 부정적인 감정을 나타냅니다.

### 베이스라인 설정 {#establish-a-baseline}

첫 번째 단계로, 좋은 성능의 베이스라인을 설정합니다.
이 과정에서는 KerasNLP가 필요하지 않으며, 기본적인 Keras 레이어만으로도 가능합니다.

간단한 bag-of-words 모델을 트레이닝할 것입니다.
이 모델은 우리의 어휘에서 각 단어에 대해 긍정 또는 부정 가중치를 학습합니다.
샘플의 점수는 해당 샘플에 포함된 모든 단어의 가중치 합계로 계산됩니다.

```python
# 이 레이어는 입력 문장을 어휘 크기와 같은 크기의 1과 0 리스트로 변환합니다.
# 이 리스트는 단어가 존재하거나 존재하지 않음을 나타냅니다.
multi_hot_layer = keras.layers.TextVectorization(
    max_tokens=4000, output_mode="multi_hot"
)
multi_hot_layer.adapt(sst_train_ds.map(lambda x, y: x))
multi_hot_ds = sst_train_ds.map(lambda x, y: (multi_hot_layer(x), y))
multi_hot_val_ds = sst_val_ds.map(lambda x, y: (multi_hot_layer(x), y))

# 그런 다음 이 레이어에 대해 선형 회귀를 학습합니다. 이것이 베이스라인 모델입니다!

inputs = keras.Input(shape=(4000,), dtype="int32")
outputs = keras.layers.Dense(1, activation="sigmoid")(inputs)
baseline_model = keras.Model(inputs, outputs)
baseline_model.compile(loss="binary_crossentropy", metrics=["accuracy"])
baseline_model.fit(multi_hot_ds, validation_data=multi_hot_val_ds, epochs=5)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Epoch 1/5
 2105/2105 ━━━━━━━━━━━━━━━━━━━━ 2s 698us/step - accuracy: 0.6421 - loss: 0.6469 - val_accuracy: 0.7567 - val_loss: 0.5391
Epoch 2/5
 2105/2105 ━━━━━━━━━━━━━━━━━━━━ 1s 493us/step - accuracy: 0.7524 - loss: 0.5392 - val_accuracy: 0.7868 - val_loss: 0.4891
Epoch 3/5
 2105/2105 ━━━━━━━━━━━━━━━━━━━━ 1s 513us/step - accuracy: 0.7832 - loss: 0.4871 - val_accuracy: 0.7991 - val_loss: 0.4671
Epoch 4/5
 2105/2105 ━━━━━━━━━━━━━━━━━━━━ 1s 475us/step - accuracy: 0.7991 - loss: 0.4543 - val_accuracy: 0.8069 - val_loss: 0.4569
Epoch 5/5
 2105/2105 ━━━━━━━━━━━━━━━━━━━━ 1s 476us/step - accuracy: 0.8100 - loss: 0.4313 - val_accuracy: 0.8036 - val_loss: 0.4530

<keras.src.callbacks.history.History at 0x7f13902967a0>
```

{{% /details %}}

Bag-of-words 접근 방식은 입력 예제가 많은 단어를 포함할 때,
빠르고 놀라울 정도로 강력할 수 있습니다.
그러나 짧은 시퀀스의 경우, 성능에 한계가 있을 수 있습니다.

더 나은 성능을 위해, 우리는 단어를 _문맥_ 내에서 평가할 수 있는 모델을 빌드하고 싶습니다.
각 단어를 개별적으로 평가하는 대신,
입력의 _전체 순서 있는 시퀀스(entire ordered sequence)_ 에서 포함된 정보를 사용해야 합니다.

이 경우, `SST-2`는 매우 작은 데이터셋이므로,
시퀀스를 학습할 수 있는 더 큰 파라미터 모델을 구축하기에는, 예제 텍스트가 부족합니다.
트레이닝 데이터셋을 외워버리게 될 위험이 있으며, 새로운 예제에 대한 일반화 능력은 향상되지 않습니다.

이때 **사전 트레이닝**을 사용하면, 더 큰 코퍼스에 대해 학습하고,
그 지식을 `SST-2` 작업에 적용할 수 있습니다.
그리고 **KerasNLP**를 사용하면,
강력한 모델인 트랜스포머를 손쉽게 사전 트레이닝할 수 있습니다.

## 사전 트레이닝 {#pretraining}

베이스라인을 뛰어넘기 위해, 우리는 `WikiText103` 데이터셋을 활용할 것입니다.
이 데이터셋은 `SST-2`보다 훨씬 큰 비지도 학습 Wikipedia 기사 모음입니다.

우리는 _트랜스포머_ 를 트레이닝할 것입니다.
이 모델은 우리의 입력에서 각 단어를 저차원 벡터로 임베딩하는 것을 학습하는, 고도의 표현력이 있는 모델입니다.
Wikipedia 데이터셋에는 레이블이 없기 때문에,
_Masked Language Modeling_ (MaskedLM)이라는 비지도 트레이닝 목표를 사용할 것입니다.

본질적으로, 우리는 "숨겨진 단어 맞추기"라는 큰 게임을 할 것입니다.
각 입력 샘플에서 입력 데이터의 25%를 가리고, 그 부분을 예측하도록 모델을 트레이닝할 것입니다.

### MaskedLM 작업을 위한 데이터 전처리 {#preprocess-data-for-the-maskedlm-task}

MaskedLM 작업을 위한 텍스트 전처리는 두 단계로 이루어집니다.

1. 입력 텍스트를 정수 토큰 ID 시퀀스로 토큰화합니다.
2. 예측할 입력의 일부 위치를 마스킹합니다.

토큰화를 위해, 우리는 텍스트를 정수 토큰 ID 시퀀스로 변환하는 KerasNLP의 빌딩 블록인,
[`keras_nlp.tokenizers.Tokenizer`]({{< relref "/docs/api/keras_nlp/tokenizers/tokenizer#tokenizer-class" >}})를 사용할 수 있습니다.

특히, 우리는 _서브워드(sub-word)_ 토큰화를 수행하는 [`keras_nlp.tokenizers.WordPieceTokenizer`]({{< relref "/docs/api/keras_nlp/tokenizers/word_piece_tokenizer#wordpiecetokenizer-class" >}})를 사용할 것입니다.
서브워드 토큰화는 큰 텍스트 코퍼스에 대해 모델을 트레이닝할 때 널리 사용됩니다.
본질적으로, 이 방식은 모델이 드문 단어에서 학습할 수 있도록 하면서도,
우리의 트레이닝 세트에서 모든 단어를 포함하는 큰 어휘집을 필요로 하지 않게 합니다.

두 번째로 필요한 것은 MaskedLM 작업을 위해 입력을 마스킹하는 것입니다.
이를 위해 [`keras_nlp.layers.MaskedLMMaskGenerator`]({{< relref "/docs/api/keras_nlp/preprocessing_layers/masked_lm_mask_generator#maskedlmmaskgenerator-class" >}})를 사용할 수 있으며,
이는 각 입력에서 랜덤으로 선택된 토큰 세트를 마스킹합니다.

토크나이저와 마스킹 레이어는 모두 [tf.data.Dataset.map](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map) 호출 내에서 사용할 수 있습니다.
우리는 [`tf.data`](https://www.tensorflow.org/api_docs/python/tf/data)를 사용하여
CPU에서 각 배치를 미리 계산하고,
그 전에 GPU 또는 TPU가 트레이닝을 진행할 수 있도록 할 수 있습니다.
마스킹 레이어는 각 에포크마다 새로운 단어 세트를 마스킹하므로,
데이터셋을 순회할 때마다 완전히 새로운 레이블 세트를 트레이닝할 수 있게 됩니다.

```python
# sequence_length를 설정하면,
# 토큰 출력을 (batch_size, SEQ_LENGTH) 모양으로 자르거나(trim) 패딩합니다.
tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
    vocabulary=vocab_file,
    sequence_length=SEQ_LENGTH,
    lowercase=True,
    strip_accents=True,
)
# mask_selection_length를 설정하면,
# 마스크 출력을 (batch_size, PREDICTIONS_PER_SEQ) 형태로 자르거나(trim) 패딩합니다.
masker = keras_nlp.layers.MaskedLMMaskGenerator(
    vocabulary_size=tokenizer.vocabulary_size(),
    mask_selection_rate=MASK_RATE,
    mask_selection_length=PREDICTIONS_PER_SEQ,
    mask_token_id=tokenizer.token_to_id("[MASK]"),
)


def preprocess(inputs):
    inputs = tokenizer(inputs)
    outputs = masker(inputs)
    # 마스킹 레이어 출력을 keras.Model.fit()에서 사용할 수 있는
    # (features, labels, and weights) 튜플로 분리합니다.
    features = {
        "token_ids": outputs["token_ids"],
        "mask_positions": outputs["mask_positions"],
    }
    labels = outputs["mask_ids"]
    weights = outputs["mask_weights"]
    return features, labels, weights


# prefetch()를 사용하여 CPU에서 전처리된 배치를 실시간으로 미리 계산합니다.
pretrain_ds = wiki_train_ds.map(
    preprocess, num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)
pretrain_val_ds = wiki_val_ds.map(
    preprocess, num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)

# 단일 입력 예시 미리보기
# 마스크는 셀을 실행할 때마다 변경됩니다.
print(pretrain_val_ds.take(1).get_single_element())
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
({'token_ids': <tf.Tensor: shape=(128, 128), dtype=int32, numpy=
array([[7570, 7849, 2271, ..., 9673,  103, 7570],
       [7570, 7849,  103, ..., 1007, 1012, 2023],
       [1996, 2034, 3940, ...,    0,    0,    0],
       ...,
       [2076, 1996, 2307, ...,    0,    0,    0],
       [3216,  103, 2083, ...,    0,    0,    0],
       [ 103, 2007, 1045, ...,    0,    0,    0]], dtype=int32)>, 'mask_positions': <tf.Tensor: shape=(128, 32), dtype=int64, numpy=
array([[  5,   6,   7, ..., 118, 120, 126],
       [  2,   3,  14, ..., 105, 106, 113],
       [  4,   9,  10, ...,   0,   0,   0],
       ...,
       [  4,  11,  19, ..., 117, 118,   0],
       [  1,  14,  17, ...,   0,   0,   0],
       [  0,   3,   6, ...,   0,   0,   0]])>}, <tf.Tensor: shape=(128, 32), dtype=int32, numpy=
array([[ 1010,  2124,  2004, ...,  2095, 11300,  1012],
       [ 2271, 13091,  2303, ...,  2029,  2027,  1010],
       [23976,  2007,  1037, ...,     0,     0,     0],
       ...,
       [ 1010,  1996,  1010, ...,  1999,  7511,     0],
       [ 2225,  1998, 10722, ...,     0,     0,     0],
       [ 9794,  1030,  2322, ...,     0,     0,     0]], dtype=int32)>, <tf.Tensor: shape=(128, 32), dtype=float32, numpy=
array([[1., 1., 1., ..., 1., 1., 1.],
       [1., 1., 1., ..., 1., 1., 1.],
       [1., 1., 1., ..., 0., 0., 0.],
       ...,
       [1., 1., 1., ..., 1., 1., 0.],
       [1., 1., 1., ..., 0., 0., 0.],
       [1., 1., 1., ..., 0., 0., 0.]], dtype=float32)>)
```

{{% /details %}}

위 코드 블록은 `(features, labels, weights)` 튜플로 데이터셋을 정리하여,
`keras.Model.fit()`에 직접 전달할 수 있습니다.

우리는 두 가지 특징을 가지고 있습니다:

1. `"token_ids"`: 일부 토큰이 마스크 토큰 ID로 대체된 토큰들입니다.
2. `"mask_positions"`: 마스킹된 토큰의 위치를 추적하는 역할을 합니다.

레이블은 마스킹된 토큰의 ID입니다.

모든 시퀀스에 동일한 수의 마스크가 있는 것은 아니기 때문에,
`sample_weight` 텐서를 사용하여,
패딩된 레이블을 손실 함수에서 제외하기 위해 가중치를 0으로 설정합니다.

### 트랜스포머 인코더 생성 {#create-the-transformer-encoder}

KerasNLP는 트랜스포머 인코더를 빠르게 빌드할 수 있는 모든 빌딩 블록을 제공합니다.

우리는 [`keras_nlp.layers.TokenAndPositionEmbedding`]({{< relref "/docs/api/keras_nlp/modeling_layers/token_and_position_embedding#tokenandpositionembedding-class" >}})를 사용하여, 입력 토큰 ID를 처음에 임베딩합니다.
이 레이어는 (문장의 단어와 문장의 정수 위치를 위한) 두 가지 임베딩을 동시에 학습합니다.
출력 임베딩은 이 두 가지 임베딩의 합으로 제공됩니다.

그런 다음, 여러 개의 [`keras_nlp.layers.TransformerEncoder`]({{< relref "/docs/api/keras_nlp/modeling_layers/transformer_encoder#transformerencoder-class" >}}) 레이어를 추가할 수 있습니다.
이 레이어들은 트랜스포머 모델의 핵심으로,
입력 문장의 다른 부분을 주의(attend)하게 하는 어텐션 메커니즘과, 그 뒤의 멀티 레이어 퍼셉트론 블록을 사용합니다.

이 모델의 출력은 입력 토큰 ID마다 인코딩된 벡터가 됩니다.
우리가 기준 성능으로 사용했던 bag-of-words 모델과 달리,
이 모델은 각 토큰을 해당 문맥에 맞게 임베딩합니다.

```python
inputs = keras.Input(shape=(SEQ_LENGTH,), dtype="int32")

# 위치 임베딩을 사용해 토큰을 임베딩합니다.
embedding_layer = keras_nlp.layers.TokenAndPositionEmbedding(
    vocabulary_size=tokenizer.vocabulary_size(),
    sequence_length=SEQ_LENGTH,
    embedding_dim=MODEL_DIM,
)
outputs = embedding_layer(inputs)

# 임베딩에 레이어 정규화와 드롭아웃을 적용합니다.
outputs = keras.layers.LayerNormalization(epsilon=NORM_EPSILON)(outputs)
outputs = keras.layers.Dropout(rate=DROPOUT)(outputs)

# 여러 개의 인코더 블록을 추가합니다.
for i in range(NUM_LAYERS):
    outputs = keras_nlp.layers.TransformerEncoder(
        intermediate_dim=INTERMEDIATE_DIM,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
        layer_norm_epsilon=NORM_EPSILON,
    )(outputs)

encoder_model = keras.Model(inputs, outputs)
encoder_model.summary()
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Model: "functional_3"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape              ┃    Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ input_layer_1 (InputLayer)      │ (None, 128)               │          0 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ token_and_position_embedding    │ (None, 128, 256)          │  7,846,400 │
│ (TokenAndPositionEmbedding)     │                           │            │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ layer_normalization             │ (None, 128, 256)          │        512 │
│ (LayerNormalization)            │                           │            │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ dropout (Dropout)               │ (None, 128, 256)          │          0 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ transformer_encoder             │ (None, 128, 256)          │    527,104 │
│ (TransformerEncoder)            │                           │            │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ transformer_encoder_1           │ (None, 128, 256)          │    527,104 │
│ (TransformerEncoder)            │                           │            │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ transformer_encoder_2           │ (None, 128, 256)          │    527,104 │
│ (TransformerEncoder)            │                           │            │
└─────────────────────────────────┴───────────────────────────┴────────────┘
 Total params: 9,428,224 (287.73 MB)
 Trainable params: 9,428,224 (287.73 MB)
 Non-trainable params: 0 (0.00 B)
```

{{% /details %}}

### 트랜스포머 사전 트레이닝 {#pretrain-the-transformer}

`encoder_model`을 독립된 모듈 단위로 생각할 수 있습니다.
이는 후속 작업에서 우리가 관심을 갖는 모델의 주요 부분입니다.
그러나 여전히 MaskedLM 작업에 대해 트레이닝할 수 있도록 설정해야 합니다.
이를 위해 [`keras_nlp.layers.MaskedLMHead`]({{< relref "/docs/api/keras_nlp/modeling_layers/masked_lm_head#maskedlmhead-class" >}})를 연결합니다.

이 레이어는 하나의 입력으로 토큰 인코딩을,
다른 입력으로는 원본 입력에서 마스킹된 위치를 받습니다.
이 레이어는 마스킹된 토큰 인코딩을 수집하고, 이를 전체 어휘에 대한 예측으로 변환합니다.

이제 컴파일하고 사전 트레이닝을 실행할 준비가 되었습니다.
Colab에서 실행하는 경우, 약 한 시간이 소요될 수 있습니다.
트랜스포머 모델 트레이닝은 매우 많은 계산 리소스를 요구하는 작업으로,
상대적으로 작은 이 트랜스포머 모델도 시간이 꽤 걸릴 것입니다.

```python
# 마스킹된 언어 모델 헤드를 연결하여 사전 트레이닝된 모델을 만듭니다.
inputs = {
    "token_ids": keras.Input(shape=(SEQ_LENGTH,), dtype="int32", name="token_ids"),
    "mask_positions": keras.Input(
        shape=(PREDICTIONS_PER_SEQ,), dtype="int32", name="mask_positions"
    ),
}

# 토큰을 인코딩합니다.
encoded_tokens = encoder_model(inputs["token_ids"])

# 마스크된 입력 토큰마다 출력 단어를 예측합니다.
# 입력 토큰 임베딩을 사용하여 인코딩된 벡터를 어휘 로짓으로 변환하는 것은,
# 트레이닝 효율성을 높이는 것으로 알려져 있습니다.
outputs = keras_nlp.layers.MaskedLMHead(
    token_embedding=embedding_layer.token_embedding,
    activation="softmax",
)(encoded_tokens, mask_positions=inputs["mask_positions"])

# 사전 트레이닝 모델을 정의하고 컴파일합니다.
pretraining_model = keras.Model(inputs, outputs)
pretraining_model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.AdamW(PRETRAINING_LEARNING_RATE),
    weighted_metrics=["sparse_categorical_accuracy"],
    jit_compile=True,
)

# wiki 텍스트 데이터셋으로 모델을 사전 트레이닝합니다.
pretraining_model.fit(
    pretrain_ds,
    validation_data=pretrain_val_ds,
    epochs=PRETRAINING_EPOCHS,
)

# 이후의 미세 조정을 위해 이 베이스 모델을 저장합니다.
encoder_model.save("encoder_model.keras")
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Epoch 1/8
 5857/5857 ━━━━━━━━━━━━━━━━━━━━ 242s 41ms/step - loss: 5.4679 - sparse_categorical_accuracy: 0.1353 - val_loss: 3.4570 - val_sparse_categorical_accuracy: 0.3522
Epoch 2/8
 5857/5857 ━━━━━━━━━━━━━━━━━━━━ 234s 40ms/step - loss: 3.6031 - sparse_categorical_accuracy: 0.3396 - val_loss: 3.0514 - val_sparse_categorical_accuracy: 0.4032
Epoch 3/8
 5857/5857 ━━━━━━━━━━━━━━━━━━━━ 232s 40ms/step - loss: 3.2609 - sparse_categorical_accuracy: 0.3802 - val_loss: 2.8858 - val_sparse_categorical_accuracy: 0.4240
Epoch 4/8
 5857/5857 ━━━━━━━━━━━━━━━━━━━━ 233s 40ms/step - loss: 3.1099 - sparse_categorical_accuracy: 0.3978 - val_loss: 2.7897 - val_sparse_categorical_accuracy: 0.4375
Epoch 5/8
 5857/5857 ━━━━━━━━━━━━━━━━━━━━ 235s 40ms/step - loss: 3.0145 - sparse_categorical_accuracy: 0.4090 - val_loss: 2.7504 - val_sparse_categorical_accuracy: 0.4419
Epoch 6/8
 5857/5857 ━━━━━━━━━━━━━━━━━━━━ 252s 43ms/step - loss: 2.9530 - sparse_categorical_accuracy: 0.4157 - val_loss: 2.6925 - val_sparse_categorical_accuracy: 0.4474
Epoch 7/8
 5857/5857 ━━━━━━━━━━━━━━━━━━━━ 232s 40ms/step - loss: 2.9088 - sparse_categorical_accuracy: 0.4210 - val_loss: 2.6554 - val_sparse_categorical_accuracy: 0.4513
Epoch 8/8
 5857/5857 ━━━━━━━━━━━━━━━━━━━━ 236s 40ms/step - loss: 2.8721 - sparse_categorical_accuracy: 0.4250 - val_loss: 2.6389 - val_sparse_categorical_accuracy: 0.4548
```

{{% /details %}}

## 미세 조정 {#fine-tuning}

사전 트레이닝 후, 이제 `SST-2` 데이터셋에서 모델을 미세 조정할 수 있습니다.
우리가 빌드한 인코더의 문맥에서 단어를 예측하는 능력을 활용하여,
다운스트림 작업에서 성능을 향상시킬 수 있습니다.

### 분류를 위한 데이터 전처리 {#preprocess-data-for-classification}

미세 조정을 위한 전처리는 사전 학습 MaskedLM 작업에 비해 훨씬 간단합니다.
입력 문장을 토큰화하면 바로 트레이닝할 준비가 됩니다!

```python
def preprocess(sentences, labels):
    return tokenizer(sentences), labels


# prefetch()를 사용하여 CPU에서 전처리된 배치를 실시간으로 미리 계산합니다.
finetune_ds = sst_train_ds.map(
    preprocess, num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)
finetune_val_ds = sst_val_ds.map(
    preprocess, num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)

# 단일 입력 예제를 미리 확인합니다.
print(finetune_val_ds.take(1).get_single_element())
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
(<tf.Tensor: shape=(32, 128), dtype=int32, numpy=
array([[ 2009,  1005,  1055, ...,     0,     0,     0],
       [ 4895, 10258,  2378, ...,     0,     0,     0],
       [ 4473,  2149,  2000, ...,     0,     0,     0],
       ...,
       [ 1045,  2018,  2000, ...,     0,     0,     0],
       [ 4283,  2000,  3660, ...,     0,     0,     0],
       [ 1012,  1012,  1012, ...,     0,     0,     0]], dtype=int32)>, <tf.Tensor: shape=(32,), dtype=int32, numpy=
array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0,
       0, 1, 1, 0, 0, 1, 0, 0, 1, 0], dtype=int32)>)
```

{{% /details %}}

### 트랜스포머 미세 조정 {#fine-tune-the-transformer}

인코딩된 토큰 출력을 분류 예측으로 전환하려면,
트랜스포머 모델에 또 다른 "헤드"를 붙여야 합니다.
여기서는 간단하게 접근할 수 있습니다.
인코딩된 토큰을 풀링한 후, 단일 dense 레이어를 사용하여 예측을 수행합니다.

```python
# 디스크에서 인코더 모델을 다시 로드하여 처음부터 미세 조정을 시작합니다.
encoder_model = keras.models.load_model("encoder_model.keras", compile=False)

# 토큰화된 입력을 입력으로 받습니다.
inputs = keras.Input(shape=(SEQ_LENGTH,), dtype="int32")

# 토큰을 인코딩하고 풀링합니다.
encoded_tokens = encoder_model(inputs)
pooled_tokens = keras.layers.GlobalAveragePooling1D()(encoded_tokens[0])

# 출력 레이블을 예측합니다.
outputs = keras.layers.Dense(1, activation="sigmoid")(pooled_tokens)

# 미세 조정 모델을 정의하고 컴파일합니다.
finetuning_model = keras.Model(inputs, outputs)
finetuning_model.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.AdamW(FINETUNING_LEARNING_RATE),
    metrics=["accuracy"],
)

# SST-2 작업에 대해 모델을 미세 조정합니다.
finetuning_model.fit(
    finetune_ds,
    validation_data=finetune_val_ds,
    epochs=FINETUNING_EPOCHS,
)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Epoch 1/3
 2105/2105 ━━━━━━━━━━━━━━━━━━━━ 21s 9ms/step - accuracy: 0.7500 - loss: 0.4891 - val_accuracy: 0.8036 - val_loss: 0.4099
Epoch 2/3
 2105/2105 ━━━━━━━━━━━━━━━━━━━━ 16s 8ms/step - accuracy: 0.8826 - loss: 0.2779 - val_accuracy: 0.8482 - val_loss: 0.3964
Epoch 3/3
 2105/2105 ━━━━━━━━━━━━━━━━━━━━ 16s 8ms/step - accuracy: 0.9176 - loss: 0.2066 - val_accuracy: 0.8549 - val_loss: 0.4142

<keras.src.callbacks.history.History at 0x7f12d85c21a0>
```

{{% /details %}}

사전 트레이닝 만으로도 성능을 84%까지 끌어올렸으며, 이는 트랜스포머 모델의 성능 한계와는 거리가 멉니다.
사전 트레이닝 과정에서 검증 성능이 지속적으로 증가하는 것을 확인했을 것입니다.
모델은 여전히 충분히 트레이닝되지 않았습니다.
더 많은 에포크로 트레이닝하거나, 더 큰 트랜스포머 모델을 트레이닝하거나,
더 많은 레이블이 없는 텍스트에 대해 트레이닝하면 성능이 크게 향상될 것입니다.

KerasNLP의 주요 목표 중 하나는 NLP 모델 빌드에 있어 모듈화된 접근 방식을 제공하는 것입니다.
이 예시에서는 트랜스포머를 구축하는 하나의 방법을 보여주었지만,
KerasNLP는 텍스트 전처리와 모델 구축을 위한 다양한 구성 요소들을 계속해서 지원하고 있습니다.
여러분이 자연어 처리 문제에 대한 솔루션을 실험하는 데 있어 더 쉽게 접근할 수 있도록 돕기를 바랍니다.
