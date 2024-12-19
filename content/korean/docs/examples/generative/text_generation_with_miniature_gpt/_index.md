---
title: 미니어처 GPT로 텍스트 생성
linkTitle: 미니어처 GPT 텍스트 생성
toc: true
weight: 22
type: docs
---

{{< keras/original checkedAt="2024-11-23" >}}

**{{< t f_author >}}** [Apoorv Nandan](https://twitter.com/NandanApoorv)  
**{{< t f_date_created >}}** 2020/05/29  
**{{< t f_last_modified >}}** 2020/05/29  
**{{< t f_description >}}** 미니어처 GPT를 구현하고 텍스트 생성 모델을 트레이닝하기

{{< keras/version v=3 >}}

{{< cards cols="2" >}}
{{< card link="https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/generative/ipynb/text_generation_with_miniature_gpt.ipynb" title="Colab" tag="Colab" tagType="warning">}}
{{< card link="https://github.com/keras-team/keras-io/blob/master/examples/generative/text_generation_with_miniature_gpt.py" title="GitHub" tag="GitHub">}}
{{< /cards >}}

## 소개 {#introduction}

이 예시는 미니어처 버전의 GPT 모델을 사용하여 오토리그레시브(autoregressive) 언어 모델을 구현하는 방법을 보여줍니다.
이 모델은 어텐션 레이어에 인과 마스킹(causal masking)이 적용된 하나의 Transformer 블록으로 구성됩니다.
IMDB 감정 분류 데이터셋의 텍스트를 사용해 트레이닝하며, 주어진 프롬프트에 대한 새로운 영화 리뷰를 생성합니다.
이 스크립트를 당신만의 데이터셋에 사용할 때는, 최소 100만 개의 단어가 포함된 데이터셋을 사용해야 합니다.

이 예시는 `tf-nightly>=2.3.0-dev20200531` 또는 TensorFlow 2.3 이상 버전과 함께 실행해야 합니다.

**참조:**

- [GPT](https://www.semanticscholar.org/paper/Improving-Language-Understanding-by-Generative-Radford/cd18800a0fe0b668a1cc19f2ec95b5003d0a5035)
- [GPT-2](https://www.semanticscholar.org/paper/Language-Models-are-Unsupervised-Multitask-Learners-Radford-Wu/9405cc0d6169988371b2755e573cc28650d14dfe)
- [GPT-3](https://arxiv.org/abs/2005.14165)

## 셋업 {#setup}

```python
# 백엔드를 TensorFlow로 설정합니다.
# 이 코드는 `tensorflow`와 `torch`에서 작동하며, JAX에서는 작동하지 않습니다.
# 이는 `causal_attention_mask()`에서 사용된
# JAX의 `jax.numpy.tile` 함수가 jit 스코프 내에서 동적 `reps` 인자를 지원하지 않기 때문입니다.
# JAX에서 이 코드를 작동시키려면,
# `causal_attention_mask` 함수 내부를 jit 컴파일을 방지하는 데코레이터로 감싸서 처리할 수 있습니다:
# `with jax.ensure_compile_time_eval():`.
import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
from keras import layers
from keras import ops
from keras.layers import TextVectorization
import numpy as np
import os
import string
import random
import tensorflow
import tensorflow.data as tf_data
import tensorflow.strings as tf_strings
```

## Transformer 블록을 레이어로 구현하기 {#implement-a-transformer-block-as-a-layer}

```python
def causal_attention_mask(batch_size, n_dest, n_src, dtype):
    """
    셀프 어텐션에서 점곱 행렬(dot product matrix)의 상단 절반을 마스킹합니다.
    이는 미래의 토큰에서 현재 토큰으로 정보가 흐르는 것을 방지합니다.
    오른쪽 아래 모서리에서 시작하여, 하단 삼각형에 1을 채웁니다.
    """
    i = ops.arange(n_dest)[:, None]
    j = ops.arange(n_src)
    m = i >= j - n_src + n_dest
    mask = ops.cast(m, dtype)
    mask = ops.reshape(mask, [1, n_dest, n_src])
    mult = ops.concatenate(
        [ops.expand_dims(batch_size, -1), ops.convert_to_tensor([1, 1])], 0
    )
    return ops.tile(mask, mult)


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads, embed_dim)
        self.ffn = keras.Sequential(
            [
                layers.Dense(ff_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs):
        input_shape = ops.shape(inputs)
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        causal_mask = causal_attention_mask(batch_size, seq_len, seq_len, "bool")
        attention_output = self.att(inputs, inputs, attention_mask=causal_mask)
        attention_output = self.dropout1(attention_output)
        out1 = self.layernorm1(inputs + attention_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)
```

## 임베딩 레이어 구현 {#implement-an-embedding-layer}

토큰을 위한 임베딩 레이어와 토큰 인덱스(포지션)를 위한 임베딩 레이어를 각각 생성합니다.

```python
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = ops.shape(x)[-1]
        positions = ops.arange(0, maxlen, 1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions
```

## Miniature GPT 모델 구현 {#implement-the-miniature-gpt-model}

```python
vocab_size = 20000  # 상위 20,000개의 단어만 고려
maxlen = 80  # 최대 시퀀스 크기
embed_dim = 256  # 각 토큰에 대한 임베딩 크기
num_heads = 2  # 어텐션 헤드 수
feed_forward_dim = 256  # Transformer 내부 피드 포워드 네트워크의 히든 레이어 크기


def create_model():
    inputs = layers.Input(shape=(maxlen,), dtype="int32")
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, feed_forward_dim)
    x = transformer_block(x)
    outputs = layers.Dense(vocab_size)(x)
    model = keras.Model(inputs=inputs, outputs=[outputs, x])
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(
        "adam",
        loss=[loss_fn, None],
    )  # Transformer 블록에서의 워드 임베딩 기반으로 한 최적화 및 손실 없음
    return model
```

## 단어 레벨 언어 모델링을 위한 데이터 준비 {#prepare-the-data-for-word-level-language-modelling}

IMDB 데이터를 다운로드하고 텍스트 생성 작업을 위해 트레이닝 및 검증 세트를 결합합니다.

```python
!curl -O https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
!tar -xf aclImdb_v1.tar.gz
```

```python
batch_size = 128

# 데이터셋은 각 리뷰가 개별 텍스트 파일에 포함되어 있습니다.
# 텍스트 파일은 네 개의 다른 폴더에 존재합니다.
# 모든 파일의 목록을 생성합니다.
filenames = []
directories = [
    "aclImdb/train/pos",
    "aclImdb/train/neg",
    "aclImdb/test/pos",
    "aclImdb/test/neg",
]
for dir in directories:
    for f in os.listdir(dir):
        filenames.append(os.path.join(dir, f))

print(f"{len(filenames)} files")

# 텍스트 파일에서 데이터셋을 생성합니다.
random.shuffle(filenames)
text_ds = tf_data.TextLineDataset(filenames)
text_ds = text_ds.shuffle(buffer_size=256)
text_ds = text_ds.batch(batch_size)


def custom_standardization(input_string):
    """HTML 줄 바꿈 태그를 제거하고 구두점을 처리합니다."""
    lowercased = tf_strings.lower(input_string)
    stripped_html = tf_strings.regex_replace(lowercased, "<br />", " ")
    return tf_strings.regex_replace(stripped_html, f"([{string.punctuation}])", r" \1")


# 벡터화 레이어를 생성하고 텍스트에 맞춥니다.
vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=vocab_size - 1,
    output_mode="int",
    output_sequence_length=maxlen + 1,
)
vectorize_layer.adapt(text_ds)
vocab = vectorize_layer.get_vocabulary()  # 토큰 인덱스로부터 단어를 다시 얻기 위함


def prepare_lm_inputs_labels(text):
    """
    단어 시퀀스를 1 위치씩 이동시켜 위치 (i)의 대상이 위치 (i+1)에 있는 단어가 되도록 합니다.
    모델은 위치 (i)까지의 모든 단어를 사용하여 다음 단어를 예측할 것입니다.
    """
    text = tensorflow.expand_dims(text, -1)
    tokenized_sentences = vectorize_layer(text)
    x = tokenized_sentences[:, :-1]
    y = tokenized_sentences[:, 1:]
    return x, y


text_ds = text_ds.map(prepare_lm_inputs_labels, num_parallel_calls=tf_data.AUTOTUNE)
text_ds = text_ds.prefetch(tf_data.AUTOTUNE)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100 80.2M  100 80.2M    0     0  7926k      0  0:00:10  0:00:10 --:--:-- 7661k

50000 files
```

{{% /details %}}

## 텍스트 생성을 위한 Keras 콜백 구현 {#implement-a-keras-callback-for-generating-text}

```python
class TextGenerator(keras.callbacks.Callback):
    """트레이닝된 모델에서 텍스트를 생성하는 콜백.
    1. 모델에 시작 프롬프트를 제공
    2. 다음 토큰에 대한 확률 예측
    3. 다음 토큰을 샘플링하여 다음 입력에 추가

    Arguments:
        max_tokens: 정수, 프롬프트 이후에 생성할 토큰의 수.
        start_tokens: 정수 리스트, 시작 프롬프트의 토큰 인덱스.
        index_to_word: 문자열 리스트, TextVectorization 레이어에서 얻음.
        top_k: 정수, `top_k` 토큰 예측에서 샘플링.
        print_every: 정수, 이 만큼의 에포크 이후에 출력.
    """

    def __init__(
        self, max_tokens, start_tokens, index_to_word, top_k=10, print_every=1
    ):
        self.max_tokens = max_tokens
        self.start_tokens = start_tokens
        self.index_to_word = index_to_word
        self.print_every = print_every
        self.k = top_k

    def sample_from(self, logits):
        logits, indices = ops.top_k(logits, k=self.k, sorted=True)
        indices = np.asarray(indices).astype("int32")
        preds = keras.activations.softmax(ops.expand_dims(logits, 0))[0]
        preds = np.asarray(preds).astype("float32")
        return np.random.choice(indices, p=preds)

    def detokenize(self, number):
        return self.index_to_word[number]

    def on_epoch_end(self, epoch, logs=None):
        start_tokens = [_ for _ in self.start_tokens]
        if (epoch + 1) % self.print_every != 0:
            return
        num_tokens_generated = 0
        tokens_generated = []
        while num_tokens_generated <= self.max_tokens:
            pad_len = maxlen - len(start_tokens)
            sample_index = len(start_tokens) - 1
            if pad_len < 0:
                x = start_tokens[:maxlen]
                sample_index = maxlen - 1
            elif pad_len > 0:
                x = start_tokens + [0] * pad_len
            else:
                x = start_tokens
            x = np.array([x])
            y, _ = self.model.predict(x, verbose=0)
            sample_token = self.sample_from(y[0][sample_index])
            tokens_generated.append(sample_token)
            start_tokens.append(sample_token)
            num_tokens_generated = len(tokens_generated)
        txt = " ".join(
            [self.detokenize(_) for _ in self.start_tokens + tokens_generated]
        )
        print(f"generated text:\n{txt}\n")


# 시작 프롬프트 토큰화
word_to_index = {}
for index, word in enumerate(vocab):
    word_to_index[word] = index

start_prompt = "this movie is"
start_tokens = [word_to_index.get(_, 1) for _ in start_prompt.split()]
num_tokens_generated = 40
text_gen_callback = TextGenerator(num_tokens_generated, start_tokens, vocab)
```

## 모델 트레이닝 {#train-the-model}

**Note:** 이 코드는 가능하면 GPU에서 실행하는 것이 좋습니다.

```python
model = create_model()

model.fit(text_ds, verbose=2, epochs=25, callbacks=[text_gen_callback])
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Epoch 1/25

WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1699499022.078758  633491 device_compiler.h:187] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.
/home/mattdangerw/miniconda3/envs/keras-tensorflow/lib/python3.10/contextlib.py:153: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.
  self.gen.throw(typ, value, traceback)

generated text:
this movie is a good example of the [UNK] " movies , and the movie was pretty well written , i had to say that the movie made me of the [UNK] " and was very well done . i 've seen a few
391/391 - 33s - 84ms/step - loss: 5.4696
```

```plain
Epoch 2/25
generated text:
this movie is so far the worst movies i have ever seen . it is that it just a bit of a movie but i really don 't think it is a very bad movie . it is a lot and the characters in
391/391 - 16s - 42ms/step - loss: 4.7016
```

```plain
Epoch 3/25
generated text:
this movie is a classic and has a good cast in a good story . the movie itself is good at best . the acting is superb , but the story is a little bit slow , the music hall , and music is
391/391 - 16s - 42ms/step - loss: 4.4533
```

```plain
Epoch 4/25
generated text:
this movie is a good , and is not the greatest movie ever since , the director has a lot of [UNK] , but it 's just a bit of the original and the plot has some decent acting , it has a bit
391/391 - 16s - 42ms/step - loss: 4.2985
```

```plain
Epoch 5/25
generated text:
this movie is really bad , the acting in this movie is bad and bad . it 's not bad it . it 's a bad story about a bad film but i can 't imagine if it 's a bad ending . the
391/391 - 17s - 42ms/step - loss: 4.1787
```

```plain
Epoch 6/25
generated text:
this movie is so bad , the bad acting , everything is awful , the script is bad , and the only one that i just saw in the original [UNK] . i was hoping it could make up the sequel . it wasn
391/391 - 17s - 42ms/step - loss: 4.0807
```

```plain
Epoch 7/25
generated text:
this movie is one of the best kung fu movies i 've ever seen , i have seen in my life that 's not for anyone who has to think of it , or maybe even though , i can 't find it funny
391/391 - 16s - 42ms/step - loss: 3.9978
```

```plain
Epoch 8/25
generated text:
this movie is just plain boring . . . . . . . . . . . . . . . . . [UNK] , the movie [UNK] . . . [UNK] . . . . . . [UNK] is a bad , it
391/391 - 17s - 42ms/step - loss: 3.9236
```

```plain
Epoch 9/25
generated text:
this movie is the only good movie i think i 've never seen it again . but it 's the only thing i feel about it . the story was about the fact that it was a very good movie . the movie has
391/391 - 17s - 42ms/step - loss: 3.8586
```

```plain
Epoch 10/25
generated text:
this movie is very well written and directed . it contains some of the best [UNK] in the genre . the story is about a group of actors , especially jamie harris and danny glover who are the only good guys that is really
391/391 - 17s - 42ms/step - loss: 3.8002
```

```plain
Epoch 11/25
generated text:
this movie is so terrible . i think that the movie isn 't as bad as you should go and watch it again . there were so many clichés that it 's a very bad movie in itself . there is no story line
391/391 - 17s - 42ms/step - loss: 3.7478
```

```plain
Epoch 12/25
generated text:
this movie is a total waste of money and money . i am surprised to find it very funny , very enjoyable . the plot is totally unbelievable , the acting is horrible . the story is awful , it 's not scary at
391/391 - 17s - 42ms/step - loss: 3.6993
```

```plain
Epoch 13/25
generated text:
this movie is so bad and not very good as it goes . it 's a nice movie and it 's so bad that it takes you back on your tv . i don 't really know how bad this movie is . you
391/391 - 17s - 42ms/step - loss: 3.6546
```

```plain
Epoch 14/25
generated text:
this movie is a great fun story , with lots of action , and romance . if you like the action and the story is really bad . it doesn 't get the idea , but you have your heart of darkness . the
391/391 - 17s - 42ms/step - loss: 3.6147
```

```plain
Epoch 15/25
generated text:
this movie is a little more than a horror film . it 's not really a great deal , i can honestly say , a story about a group of teens that are all over the place . but this is still a fun
391/391 - 17s - 42ms/step - loss: 3.5769
```

```plain
Epoch 16/25
generated text:
this movie is just about a guy who is supposed to be a girl in the [UNK] of a movie that doesn 't make sense . the humor is not to watch it all the way the movie is . you can 't know
391/391 - 17s - 42ms/step - loss: 3.5425
```

```plain
Epoch 17/25
generated text:
this movie is one of the best movies i 've ever seen . i was really surprised when renting it and it wasn 't even better in it , it was not even funny and i really don 't really know what i was
391/391 - 17s - 42ms/step - loss: 3.5099
```

```plain
Epoch 18/25
generated text:
this movie is so bad . i think it 's a bit overrated . i have a lot of bad movies . i have to say that this movie was just bad . i was hoping the [UNK] . the [UNK] is good "
391/391 - 17s - 43ms/step - loss: 3.4800
```

```plain
Epoch 19/25
generated text:
this movie is one of the best kung fu movies i 've ever seen . it was a great movie , and for the music . the graphics are really cool . it 's like a lot more than the action scenes and action
391/391 - 17s - 42ms/step - loss: 3.4520
```

```plain
Epoch 20/25
generated text:
this movie is just plain awful and stupid .i cant get the movie . i cant believe people have ever spent money and money on the [UNK] . i swear i was so embarrassed to say that i had a few lines that are
391/391 - 17s - 42ms/step - loss: 3.4260
```

```plain
Epoch 21/25
generated text:
this movie is one of those movies that i 've ever seen , and you must know that i can say that i was not impressed with this one . i found it to be an interesting one . the story of the first
391/391 - 17s - 42ms/step - loss: 3.4014
```

```plain
Epoch 22/25
generated text:
this movie is about a man 's life and it is a very good film and it takes a look at some sort of movie . this movie is one of the most amazing movie you have ever had to go in , so
391/391 - 17s - 42ms/step - loss: 3.3783
```

```plain
Epoch 23/25
generated text:
this movie is a great , good thing about this movie , even the worst i 've ever seen ! it doesn 't mean anything terribly , the acting and the directing is terrible . the script is bad , the plot and the
391/391 - 17s - 42ms/step - loss: 3.3564
```

```plain
Epoch 24/25
generated text:
this movie is one of the best movies ever . [UNK] [UNK] ' is about the main character and a nobleman named fallon ; is stranded on an eccentric , falls in love when her island escaped . when , meanwhile , the escaped
391/391 - 17s - 42ms/step - loss: 3.3362
```

```plain
Epoch 25/25
generated text:
this movie is very good . the acting , especially the whole movie itself - a total of the worst . this movie had a lot to recommend it to anyone . it is not funny . the story is so lame ! the
391/391 - 17s - 42ms/step - loss: 3.3170

<keras.src.callbacks.history.History at 0x7f2166975f90>
```

{{% /details %}}
