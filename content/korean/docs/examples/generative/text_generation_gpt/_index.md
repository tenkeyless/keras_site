---
title: KerasHub로 처음부터 GPT 텍스트 생성하기
linkTitle: KerasHub 처음부터 GPT 텍스트 생성
toc: true
weight: 21
type: docs
---

{{< keras/original checkedAt="2024-11-23" >}}

**{{< t f_author >}}** [Jesse Chan](https://github.com/jessechancy)  
**{{< t f_date_created >}}** 2022/07/25  
**{{< t f_last_modified >}}** 2022/07/25  
**{{< t f_description >}}** KerasHub를 사용하여 텍스트 생성을 위한 mini-GPT 모델 트레이닝하기.

{{< keras/version v=3 >}}

{{< cards cols="2" >}}
{{< card link="https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/generative/ipynb/text_generation_gpt.ipynb" title="Colab" tag="Colab" tagType="warning">}}
{{< card link="https://github.com/keras-team/keras-io/blob/master/examples/generative/text_generation_gpt.py" title="GitHub" tag="GitHub">}}
{{< /cards >}}

## 소개 {#introduction}

이 예시에서는, KerasHub를 사용하여 축소된 GPT(Generative Pre-Trained) 모델을 구축합니다.
GPT는 Transformer 기반 모델로, 프롬프트를 통해 정교한 텍스트를 생성할 수 있습니다.

우리는 [simplebooks-92](https://arxiv.org/abs/1911.12391) 코퍼스를 모델 트레이닝에 사용할 것입니다.
이 코퍼스는 몇몇 소설들로 구성되어 있으며,
작은 어휘 크기와 높은 단어 빈도가 있어, 적은 매개변수로 모델을 트레이닝하는 데 유리합니다.

이 예시는 [미니어처 GPT를 사용한 텍스트 생성]({{< relref "/docs/examples/generative/text_generation_with_miniature_gpt" >}})에서의 개념과
KerasHub 추상화를 결합합니다.
KerasHub의 토크나이제이션, 레이어, 그리고 메트릭을 사용하여 트레이닝 과정을 간소화하는 방법을 보여주고,
이후 KerasHub 샘플링 유틸리티를 사용하여, 출력 텍스트를 생성하는 방법을 설명합니다.

참고: 이 예시를 Colab에서 실행할 경우, 더 빠른 트레이닝을 위해 GPU 런타임을 활성화하십시오.

이 예시에서는 KerasHub가 필요합니다. 아래 명령을 통해 설치할 수 있습니다: `pip install keras-hub`

## Setup {#setup}

```python
!pip install -q --upgrade keras-hub
!pip install -q --upgrade keras  # Keras 3으로 업그레이드.
```

```python
import os
import keras_hub
import keras

import tensorflow.data as tf_data
import tensorflow.strings as tf_strings
```

## 세팅 & 하이퍼파라미터 {#settings-hyperparameters}

```python
# 데이터 설정
BATCH_SIZE = 64
MIN_STRING_LEN = 512  # 이보다 짧은 문자열은 버려집니다
SEQ_LEN = 128  # 트레이닝 시퀀스 길이(토큰 단위)

# 모델 설정
EMBED_DIM = 256
FEED_FORWARD_DIM = 128
NUM_HEADS = 3
NUM_LAYERS = 2
VOCAB_SIZE = 5000  # 모델 매개변수 제한

# 트레이닝 설정
EPOCHS = 5

# 추론 설정
NUM_TOKENS_TO_GENERATE = 80
```

## 데이터 로드 {#load-the-data}

이제 데이터셋을 다운로드해 보겠습니다!
SimpleBooks 데이터셋은 1,573개의 Gutenberg 책들로 구성되어 있으며,
단어 레벨 토큰에 대한 어휘 크기 비율이 가장 작은 데이터셋 중 하나입니다.
약 98,000개의 어휘 크기를 가지며, WikiText-103의 어휘 크기의 3분의 1에 해당합니다.
하지만 토큰 수는 약 1억 개로 비슷하여 작은 모델에 적합한 데이터셋입니다.

```python
keras.utils.get_file(
    origin="https://dldata-public.s3.us-east-2.amazonaws.com/simplebooks.zip",
    extract=True,
)
dir = os.path.expanduser("~/.keras/datasets/simplebooks/")

# simplebooks-92 트레이닝 세트를 로드하고, 짧은 줄은 필터링합니다.
raw_train_ds = (
    tf_data.TextLineDataset(dir + "simplebooks-92-raw/train.txt")
    .filter(lambda x: tf_strings.length(x) > MIN_STRING_LEN)
    .batch(BATCH_SIZE)
    .shuffle(buffer_size=256)
)

# simplebooks-92 검증 세트를 로드하고, 짧은 줄은 필터링합니다.
raw_val_ds = (
    tf_data.TextLineDataset(dir + "simplebooks-92-raw/valid.txt")
    .filter(lambda x: tf_strings.length(x) > MIN_STRING_LEN)
    .batch(BATCH_SIZE)
)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Downloading data from https://dldata-public.s3.us-east-2.amazonaws.com/simplebooks.zip
 282386239/282386239 ━━━━━━━━━━━━━━━━━━━━ 7s 0us/step
```

{{% /details %}}

## 토크나이저 트레이닝 {#train-the-tokenizer}

`VOCAB_SIZE`에 맞게 트레이닝 데이터셋에서 토크나이저를 트레이닝시킵니다.
이 값은 조정된 하이퍼파라미터입니다.
우리는 어휘 크기를 가능한 한 제한하려고 합니다. 이후에 보겠지만, 어휘 크기는 모델 파라미터 수에 큰 영향을 미칩니다.​
또한 _너무 적은_ 어휘를 사용하는 것은 원하지 않습니다. 그렇지 않으면, OOV(out-of-vocabulary) 하위 단어가 너무 많아질 수 있습니다.
또한, 어휘에는 세 개의 예약된 토큰이 포함됩니다:

- `"[PAD]"`:
  - `SEQ_LEN`에 맞추어 시퀀스를 패딩하는 데 사용됩니다.
  - 이 토큰은 `reserved_tokens`와 `vocab`에서 0번 인덱스를 차지하며,
    `WordPieceTokenizer`와 다른 레이어는 `0`이나 `vocab[0]`을 기본 패딩으로 간주합니다.
- `"[UNK]"`:
  - OOV 하위 단어에 대한 토큰이며, `WordPieceTokenizer`의 기본 `oov_token="[UNK]"`와 일치해야 합니다.
- `"[BOS]"`:
  - 문장의 시작을 의미하며, 여기에서는 트레이닝 데이터의 각 줄의 시작을 나타내는 토큰으로 사용됩니다.

```python
# 토크나이저 어휘 트레이닝
vocab = keras_hub.tokenizers.compute_word_piece_vocabulary(
    raw_train_ds,
    vocabulary_size=VOCAB_SIZE,
    lowercase=True,
    reserved_tokens=["[PAD]", "[UNK]", "[BOS]"],
)
```

## 토크나이저 로드 {#load-tokenizer}

어휘 데이터를 사용하여, [`keras_hub.tokenizers.WordPieceTokenizer`]({{< relref "/docs/api/keras_hub/tokenizers/word_piece_tokenizer#wordpiecetokenizer-class" >}})를 초기화합니다.
WordPieceTokenizer는 BERT 및 다른 모델에서 사용되는 WordPiece 알고리즘의 효율적인 구현입니다.
이 토크나이저는 공백 제거, 소문자 변환 등의 비가역적인 전처리 작업을 수행합니다.

```python
tokenizer = keras_hub.tokenizers.WordPieceTokenizer(
    vocabulary=vocab,
    sequence_length=SEQ_LEN,
    lowercase=True,
)
```

## 데이터 토크나이징 {#tokenize-data}

데이터셋을 토크나이징하고 이를 `features`와 `labels`로 분할하여 전처리합니다.

```python
# packer는 시작 토큰을 추가합니다.
start_packer = keras_hub.layers.StartEndPacker(
    sequence_length=SEQ_LEN,
    start_value=tokenizer.token_to_id("[BOS]"),
)


def preprocess(inputs):
    outputs = tokenizer(inputs)
    features = start_packer(outputs)
    labels = outputs
    return features, labels


# 토크나이징하고 트레이닝 및 레이블 시퀀스로 분할합니다.
train_ds = raw_train_ds.map(preprocess, num_parallel_calls=tf_data.AUTOTUNE).prefetch(
    tf_data.AUTOTUNE
)
val_ds = raw_val_ds.map(preprocess, num_parallel_calls=tf_data.AUTOTUNE).prefetch(
    tf_data.AUTOTUNE
)
```

## 모델 빌드 {#build-the-model}

다음 레이어들로 축소된 GPT 모델을 생성합니다:

- [`keras_hub.layers.TokenAndPositionEmbedding`]({{< relref "/docs/api/keras_hub/modeling_layers/token_and_position_embedding#tokenandpositionembedding-class" >}}) 레이어 하나, 이 레이어는 토큰과 위치의 임베딩을 결합합니다.
- 기본적인 causal 마스킹을 사용하는 여러 [`keras_hub.layers.TransformerDecoder`]({{< relref "/docs/api/keras_hub/modeling_layers/transformer_decoder#transformerdecoder-class" >}}) 레이어들. 이 레이어는 디코더 시퀀스만 사용할 때, 교차-어텐션을 포함하지 않습니다.
- 마지막으로 하나의 dense 선형 레이어가 있습니다.

```python
inputs = keras.layers.Input(shape=(None,), dtype="int32")
# 임베딩
embedding_layer = keras_hub.layers.TokenAndPositionEmbedding(
    vocabulary_size=VOCAB_SIZE,
    sequence_length=SEQ_LEN,
    embedding_dim=EMBED_DIM,
    mask_zero=True,
)
x = embedding_layer(inputs)
# Transformer 디코더들
for _ in range(NUM_LAYERS):
    decoder_layer = keras_hub.layers.TransformerDecoder(
        num_heads=NUM_HEADS,
        intermediate_dim=FEED_FORWARD_DIM,
    )
    x = decoder_layer(x)  # 하나의 인자만 넘기면 교차-어텐션을 스킵합니다.
# 출력
outputs = keras.layers.Dense(VOCAB_SIZE)(x)
model = keras.Model(inputs=inputs, outputs=outputs)
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
perplexity = keras_hub.metrics.Perplexity(from_logits=True, mask_token_id=0)
model.compile(optimizer="adam", loss=loss_fn, metrics=[perplexity])
```

모델 요약을 살펴보겠습니다. `token_and_position_embedding` 레이어와
출력 `dense` 레이어에서 대부분의 파라미터가 있습니다!
이는 어휘 크기(`VOCAB_SIZE`)가 모델 크기에 큰 영향을 미치고,
Transformer 디코더 레이어 수(`NUM_LAYERS`)는 상대적으로 영향을 덜 미친다는 것을 의미합니다.

```python
model.summary()
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Model: "functional_1"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape              ┃    Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ input_layer (InputLayer)        │ (None, None)              │          0 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ token_and_position_embedding    │ (None, None, 256)         │  1,312,768 │
│ (TokenAndPositionEmbedding)     │                           │            │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ transformer_decoder             │ (None, None, 256)         │    329,085 │
│ (TransformerDecoder)            │                           │            │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ transformer_decoder_1           │ (None, None, 256)         │    329,085 │
│ (TransformerDecoder)            │                           │            │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ dense (Dense)                   │ (None, None, 5000)        │  1,285,000 │
└─────────────────────────────────┴───────────────────────────┴────────────┘
 Total params: 3,255,938 (12.42 MB)
 Trainable params: 3,255,938 (12.42 MB)
 Non-trainable params: 0 (0.00 B)
```

{{% /details %}}

## 트레이닝 {#training}

이제 모델을 가졌으니, `fit()` 메서드를 사용해 트레이닝을 시작해 보겠습니다.

```python
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Epoch 1/5
 2445/2445 ━━━━━━━━━━━━━━━━━━━━ 216s 66ms/step - loss: 5.0008 - perplexity: 180.0715 - val_loss: 4.2176 - val_perplexity: 68.0438
Epoch 2/5
 2445/2445 ━━━━━━━━━━━━━━━━━━━━ 127s 48ms/step - loss: 4.1699 - perplexity: 64.7740 - val_loss: 4.0553 - val_perplexity: 57.7996
Epoch 3/5
 2445/2445 ━━━━━━━━━━━━━━━━━━━━ 126s 47ms/step - loss: 4.0286 - perplexity: 56.2138 - val_loss: 4.0134 - val_perplexity: 55.4446
Epoch 4/5
 2445/2445 ━━━━━━━━━━━━━━━━━━━━ 134s 50ms/step - loss: 3.9576 - perplexity: 52.3643 - val_loss: 3.9900 - val_perplexity: 54.1153
Epoch 5/5
 2445/2445 ━━━━━━━━━━━━━━━━━━━━ 135s 51ms/step - loss: 3.9080 - perplexity: 49.8242 - val_loss: 3.9500 - val_perplexity: 52.0006

<keras.src.callbacks.history.History at 0x7f7de0365ba0>
```

{{% /details %}}

## 추론 {#inference}

트레이닝된 모델로 성능을 확인해 봅시다.
이를 위해 모델에 `"[BOS]"` 토큰으로 시작하는 입력 시퀀스를 주고,
반복적으로 다음 토큰을 예측하여 시퀀스를 점진적으로 생성하는 샘플링 방법을 사용할 수 있습니다.

먼저 모델 입력과 같은 형태로 `"[BOS]"` 토큰만 포함된 프롬프트를 생성해 보겠습니다.

```python
# "packer" 레이어가 [BOS] 토큰을 추가해줍니다.
prompt_tokens = start_packer(tokenizer([""]))
prompt_tokens
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
<tf.Tensor: shape=(1, 128), dtype=int32, numpy=
array([[2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
      dtype=int32)>
```

{{% /details %}}

우리는 `keras_hub.samplers` 모듈을 사용하여 추론을 진행할 것이며,
이를 위해 방금 트레이닝한 모델을 감싸는 콜백 함수가 필요합니다.
이 래퍼 함수는 모델을 호출하여 현재 생성 중인 토큰에 대한 logit 예측을 반환합니다.

참고: 콜백을 정의할 때 사용할 수 있는 두 가지 고급 기능이 있습니다.
첫 번째는 이전 생성 단계에서 계산된 상태의 `cache`를 입력으로 받아들여, 생성 속도를 높일 수 있는 기능입니다.
두 번째는 각 생성된 토큰의 최종 "히든 상태"를 출력할 수 있는 기능입니다.
이는 반복을 피하기 위해 반복된 히든 상태를 패널티로 적용하는 [`keras_hub.samplers.ContrastiveSampler`]({{< relref "/docs/api/keras_hub/samplers/contrastive_sampler#contrastivesampler-class" >}})에서 사용됩니다.
두 기능 모두 선택 사항이며, 지금은 무시하겠습니다.

```python
def next(prompt, cache, index):
    logits = model(prompt)[:, index - 1, :]
    # 히든 상태는 지금 무시합니다; 대조 검색에서는 필요합니다.
    hidden_states = None
    return logits, hidden_states, cache
```

래퍼 함수를 만드는 것이 이러한 함수들을 사용하는 데 가장 복잡한 부분입니다.
이제 끝났으니, 다양한 유틸리티를 테스트해보겠습니다.
먼저 Greedy Search를 사용해 보겠습니다.

### Greedy search {#greedy-search}

우리는 각 시간 단계에서 가장 가능성이 높은 토큰을 선택합니다. 즉, 모델 출력의 argmax를 가져옵니다.

```python
sampler = keras_hub.samplers.GreedySampler()
output_tokens = sampler(
    next=next,
    prompt=prompt_tokens,
    index=1,  # [BOS] 토큰 이후 바로 샘플링 시작.
)
txt = tokenizer.detokenize(output_tokens)
print(f"Greedy search generated text: \n{txt}\n")
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Greedy search generated text:
[b'[BOS] " i \' m going to tell you , " said the boy , " i \' ll tell you , and you \' ll be a good friend , and you \' ll be a good friend , and you \' ll be a good friend , and you \' ll be a good friend , and you \' ll be a good friend , and you \' ll be a good friend , and you \' ll be a good friend , and you \' ll be a good friend , and you \' ll be a good friend , and you \' ll be a good friend , and you \' ll be a good friend , and you \' ll be a good']
```

{{% /details %}}

보시다시피 Greedy Search는 처음에는 약간의 의미가 있지만, 곧 스스로를 반복하기 시작합니다.
이는 텍스트 생성에서 흔히 발생하는 문제로,
이후에 다룰 몇 가지 확률 기반 텍스트 생성 유틸리티를 사용하여 해결할 수 있습니다!

### Beam search {#beam-search}

높은 레벨에서, Beam Search는 각 시간 단계에서 가장 가능성이 높은 `num_beams`개의 시퀀스를 추적하며,
모든 시퀀스에서 가장 좋은 다음 토큰을 예측합니다.
이는 여러 가능성을 저장하기 때문에 Greedy Search보다 개선된 방법이지만,
여러 잠재적인 시퀀스를 계산하고 저장해야 하므로 Greedy Search보다는 덜 효율적입니다.

**참고:** `num_beams=1`인 Beam Search는 Greedy Search와 동일합니다.

```python
sampler = keras_hub.samplers.BeamSampler(num_beams=10)
output_tokens = sampler(
    next=next,
    prompt=prompt_tokens,
    index=1,
)
txt = tokenizer.detokenize(output_tokens)
print(f"Beam search generated text: \n{txt}\n")
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Beam search generated text:
[b'[BOS] " i don \' t know anything about it , " she said . " i don \' t know anything about it . i don \' t know anything about it , but i don \' t know anything about it . i don \' t know anything about it , but i don \' t know anything about it . i don \' t know anything about it , but i don \' t know it . i don \' t know it , but i don \' t know it . i don \' t know it , but i don \' t know it . i don \' t know it , but i don \' t know it . i don \'']
```

{{% /details %}}

Beam Search 역시 Greedy Search처럼 반복을 시작하게 되며, 이는 여전히 결정론적 방법이기 때문입니다.

### Random search {#random-search}

Random Search는 우리의 첫 번째 확률적 방법입니다.
각 시간 단계에서, 모델이 제공한 softmax 확률을 사용하여 다음 토큰을 샘플링합니다.

```python
sampler = keras_hub.samplers.RandomSampler()
output_tokens = sampler(
    next=next,
    prompt=prompt_tokens,
    index=1,
)
txt = tokenizer.detokenize(output_tokens)
print(f"Random search generated text: \n{txt}\n")
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Random search generated text:
[b'[BOS] eleanor . like ice , not children would have suspicious forehead . they will see him , no goods in her plums . i have made a stump one , on the occasion , - - it is sacred , and one is unholy - plaything - - the partial consequences , and one refuge in a style of a boy , who was his grandmother . it was a young gentleman who bore off upon the middle of the day , rush and as he maltreated the female society , were growing at once . in and out of the craid little plays , stopping']
```

{{% /details %}}

Voilà, 반복이 없습니다! 하지만 Random Search에서는 어휘의 모든 단어가 등장할 가능성이 있기 때문에,
비합리적인 단어들이 나타날 수 있습니다.
이 문제는 다음 검색 유틸리티인 Top-K Search로 해결할 수 있습니다.

### Top-K search {#top-k-search}

Random Search와 유사하게, 우리는 모델이 제공한 확률 분포에서 다음 토큰을 샘플링합니다.
유일한 차이점은 여기에서 `k`개의 가장 가능성이 높은 토큰만을 선택하고,
샘플링하기 전에 이들에 대한 확률 질량을 분배한다는 점입니다.
이렇게 하면 낮은 확률의 토큰을 샘플링하지 않게 되어, 비합리적인 단어가 나타날 가능성이 줄어듭니다!

```python
sampler = keras_hub.samplers.TopKSampler(k=10)
output_tokens = sampler(
    next=next,
    prompt=prompt_tokens,
    index=1,
)
txt = tokenizer.detokenize(output_tokens)
print(f"Top-K search generated text: \n{txt}\n")
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Top-K search generated text:
[b'[BOS] " the young man was not the one , and the boy went away to the green forest . they were a little girl \' s wife , and the child loved him as much as he did , and he had often heard of a little girl who lived near the house . they were too tired to go , and when they went down to the barns and get into the barn , and they got the first of the barns that they had been taught to do so , and the little people went to their homes . she did , she told them that she had been a very clever , and they had made the first . she knew they']
```

{{% /details %}}

### Top-P search {#top-p-search}

Top-K Search에도 개선할 부분이 있습니다.
Top-K Search에서는 `k`가 고정되어 있어, 어떤 확률 분포든 동일한 수의 토큰을 선택합니다.
예를 들어, 확률 질량이 2개의 단어에 집중된 시나리오와 10개의 단어에 고르게 분포된 시나리오가 있다고 가정해 보겠습니다.
`k=2` 또는 `k=10`을 선택해야 할까요? 여기서 모든 `k`에 적합한 정답은 없습니다.

이때 등장하는 것이 Top-P Search입니다! `k`를 선택하는 대신,
상위 토큰의 확률 합이 `p`가 되도록 확률 `p`를 선택합니다.
이렇게 하면 확률 분포에 따라 동적으로 `k`를 조정할 수 있습니다.
예를 들어 `p=0.9`로 설정하면, 90%의 확률 질량이 상위 2개의 토큰에 집중되어 있는 경우,
상위 2개의 토큰을 필터링해 샘플링할 수 있습니다.
반대로 90%가 10개의 토큰에 분포되어 있으면, 상위 10개의 토큰을 필터링해 샘플링할 수 있습니다.

```python
sampler = keras_hub.samplers.TopPSampler(p=0.5)
output_tokens = sampler(
    next=next,
    prompt=prompt_tokens,
    index=1,
)
txt = tokenizer.detokenize(output_tokens)
print(f"Top-P search generated text: \n{txt}\n")
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Top-P search generated text:
[b'[BOS] the children were both born in the spring , and the youngest sister were very much like the other children , but they did not see them . they were very happy , and their mother was a beautiful one . the youngest was one of the youngest sister of the youngest , and the youngest baby was very fond of the children . when they came home , they would see a little girl in the house , and had the beautiful family , and the children of the children had to sit and look on their backs , and the eldest children were very long , and they were so bright and happy , as they were , they had never noticed their hair ,']
```

{{% /details %}}

### 콜백을 사용한 텍스트 생성 {#using-callbacks-for-text-generation}

콜백을 사용하여 각 에포크마다 모델의 예측 시퀀스를 출력할 수 있습니다!
아래는 Top-K Search를 사용하는 콜백의 예시입니다:

```python
class TopKTextGenerator(keras.callbacks.Callback):
    """Top-K를 사용하여 트레이닝된 모델에서 텍스트를 생성하는 콜백."""

    def __init__(self, k):
        self.sampler = keras_hub.samplers.TopKSampler(k)

    def on_epoch_end(self, epoch, logs=None):
        output_tokens = self.sampler(
            next=next,
            prompt=prompt_tokens,
            index=1,
        )
        txt = tokenizer.detokenize(output_tokens)
        print(f"Top-K search generated text: \n{txt}\n")


text_generation_callback = TopKTextGenerator(k=10)
# 콜백을 사용한 더미 트레이닝 루프 시연.
model.fit(train_ds.take(1), verbose=2, epochs=2, callbacks=[text_generation_callback])
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Epoch 1/2
Top-K search generated text:
[b"[BOS] the young man was in the middle of a month , and he was able to take the crotch , but a long time , for he felt very well for himself in the sepoys ' s hands were chalks . he was the only boy , and he had a few years before been married , and the man said he was a tall one . he was a very handsome , and he was a very handsome young fellow , and a handsome , noble young man , but a boy , and man . he was a very handsome man , and was tall and handsome , and he looked like a gentleman . he was an"]
1/1 - 16s - 16s/step - loss: 3.9454 - perplexity: 51.6987
```

```plain
Epoch 2/2
Top-K search generated text:
[b'[BOS] " well , it is true . it is true that i should go to the house of a collector , in the matter of prussia that there is no other way there . there is no chance of being in the habit of being in the way of an invasion . i know not what i have done , but i have seen the man in the middle of a day . the next morning i shall take him to my father , for i am not the very day of the town , which would have been a little more than the one \' s daughter , i think it over and the whole affair will be']
1/1 - 17s - 17s/step - loss: 3.7860 - perplexity: 44.0932

<keras.src.callbacks.history.History at 0x7f7de0325600>
```

{{% /details %}}

## 결론 {#conclusion}

요약하자면, 이번 예시에서는 KerasHub 레이어를 사용하여 서브-워드 단어 사전을 트레이닝하고,
트레이닝 데이터를 토큰화하며, 작은 GPT 모델을 생성하고,
텍스트 생성 라이브러리를 활용해 추론을 수행했습니다.

Transformer의 작동 원리나 전체 GPT 모델을 트레이닝하는 방법에 대해 더 알고 싶다면, 다음 읽을거리들을 추천드립니다:

- Attention Is All You Need [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)
- GPT-3 Paper [Brown et al., 2020](https://arxiv.org/abs/2005.14165)
