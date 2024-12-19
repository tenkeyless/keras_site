---
title: KerasHub를 사용한 GPT2 텍스트 생성
linkTitle: GPT2 텍스트 생성 (KerasHub)
toc: true
weight: 20
type: docs
---

{{< keras/original checkedAt="2024-11-23" >}}

**{{< t f_author >}}** Chen Qian  
**{{< t f_date_created >}}** 2023/04/17  
**{{< t f_last_modified >}}** 2024/04/12  
**{{< t f_description >}}** KerasHub의 GPT2 모델과 `samplers`를 사용하여 텍스트 생성.

{{< keras/version v=3 >}}

{{< cards cols="2" >}}
{{< card link="https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/generative/ipynb/gpt2_text_generation_with_kerashub.ipynb" title="Colab" tag="Colab" tagType="warning">}}
{{< card link="https://github.com/keras-team/keras-io/blob/master/examples/generative/gpt2_text_generation_with_kerashub.py" title="GitHub" tag="GitHub">}}
{{< /cards >}}

이 튜토리얼에서는 [KerasHub]({{< relref "/docs/keras_hub" >}})를 사용하여,
사전 트레이닝된 대형 언어 모델(LLM)인
[GPT-2 모델](https://openai.com/research/better-language-models)(원래 OpenAI에서 개발)을 불러오고,
특정 텍스트 스타일에 맞게 미세 트레이닝(finetuning)을 진행한 후,
사용자 입력(프롬프트)에 기반한 텍스트를 생성하는 방법을 배웁니다.
또한, GPT2가 중국어와 같은 비영어권 언어에 빠르게 적응하는 방식을 배우게 됩니다.

## 시작하기 전에 {#before-we-begin}

Colab은 여러 가지 런타임을 제공합니다.
**Runtime -> Change runtime type**으로 이동하여,
GPU 하드웨어 가속기 런타임(12GB 이상의 호스트 RAM 및 약 15GB의 GPU RAM이 있어야 함)을 선택하세요.
GPT-2 모델을 미세 트레이닝할 예정이므로, CPU 런타임에서는 시간이 오래 걸립니다.

## KerasHub 설치, 백엔드 선택 및 종속성 import {#install-kerashub-choose-backend-and-import-dependencies}

이 예제에서는 [Keras 3]({{< relref "/docs/keras_3" >}})를 사용하여,
`"tensorflow"`, `"jax"`, 또는 `"torch"` 중 어느 것이든 사용할 수 있습니다.
KerasHub에는 Keras 3에 대한 지원이 내장되어 있으며,
사용하려는 백엔드를 선택하려면 `"KERAS_BACKEND"` 환경 변수를 변경하기만 하면 됩니다.
아래에서는 JAX 백엔드를 선택합니다.

```python
!pip install git+https://github.com/keras-team/keras-hub.git -q
```

```python
import os

os.environ["KERAS_BACKEND"] = "jax"  # 또는 "tensorflow" 또는 "torch"

import keras_hub
import keras
import tensorflow as tf
import time

keras.mixed_precision.set_global_policy("mixed_float16")
```

## 생성형 Large Language Models (LLMs) 소개 {#introduction-to-generative-large-language-models-llms}

대형 언어 모델(LLM)은 방대한 텍스트 데이터 코퍼스에 대해 트레이닝된 머신러닝 모델로,
텍스트 생성, 질문 응답, 기계 번역 등 다양한 자연어 처리(NLP) 작업에서 출력을 생성하는 데 사용됩니다.

생성형 LLM은 일반적으로 (Google이 2017년 개발한)
[Transformer 아키텍처](https://arxiv.org/abs/1706.03762)와 같은 딥러닝 신경망을 기반으로 하며,
수십억 개의 단어가 포함된 방대한 양의 텍스트 데이터를 사용하여 트레이닝됩니다.
Google [LaMDA](https://blog.google/technology/ai/lamda/)와
[PaLM](https://ai.googleblog.com/2022/04/pathways-language-model-palm-scaling-to.html) 같은 모델은
다양한 데이터 소스에서 수집한 대규모 데이터셋으로 트레이닝되며,
이를 통해 여러 작업에서 출력을 생성할 수 있습니다.
생성형 LLM의 핵심은 문장에서 다음 단어를 예측하는 방식으로,
이는 **Causal LM Pretraining**이라고 불립니다.
이러한 방식으로 LLM은 사용자 프롬프트에 따라 일관된 텍스트를 생성할 수 있습니다.
언어 모델에 대한 더 깊이 있는 논의는 [Stanford CS324 LLM 수업](https://stanford-cs324.github.io/winter2022/lectures/introduction/)을 참조하십시오.

## KerasHub 소개 {#introduction-to-kerashub}

대형 언어 모델을 처음부터 구축하고 트레이닝하는 것은 매우 복잡하고 비용이 많이 듭니다.
다행히도, 바로 사용할 수 있는 사전 트레이닝된 LLM들이 있습니다.
[KerasHub]({{< relref "/docs/keras_hub" >}})는 사전 트레이닝된 최신 모델을 제공하여,
별도의 트레이닝 없이도 실험할 수 있습니다.

KerasHub는 자연어 처리 작업의 전체 개발 주기를 지원하는 라이브러리로,
사전 트레이닝된 모델과 모듈화된 빌딩 블록을 모두 제공하여,
개발자가 사전 트레이닝된 모델을 재사용하거나 자신만의 LLM을 쉽게 구축할 수 있습니다.

요약하자면, KerasHub는 생성형 LLM을 위해 다음과 같은 기능을 제공합니다:

- `generate()` 메서드를 제공하는 사전 트레이닝된 모델,
  예: [`keras_hub.models.GPT2CausalLM`]({{< relref "/docs/api/keras_hub/models/gpt2/gpt2_causal_lm#gpt2causallm-class" >}}) 및 [`keras_hub.models.OPTCausalLM`]({{< relref "/docs/api/keras_hub/models/opt/opt_causal_lm#optcausallm-class" >}}).
- 텍스트 생성을 위한 Top-K, 빔 서치, 대조적 서치와 같은 샘플링 알고리즘을 구현하는 `Sampler` 클래스.
  이 샘플러들은 커스텀 모델과 함께 텍스트를 생성하는 데 사용할 수 있습니다.

## 사전 트레이닝된 GPT-2 모델 로드 및 텍스트 생성 {#load-a-pre-trained-gpt-2-model-and-generate-some-text}

KerasHub는 [Google Bert](https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html) 및 [GPT-2](https://openai.com/research/better-language-models)와 같은 여러 사전 트레이닝된 모델을 제공합니다.
사용 가능한 모델 목록은 [KerasHub 저장소](https://github.com/keras-team/keras-hub/tree/master/keras_hub/models)에서 확인할 수 있습니다.

아래와 같이 GPT-2 모델을 로드하는 것은 매우 간단합니다:

```python
# 트레이닝과 생성을 더 빠르게 하기 위해 전체 길이 1024 대신 길이 128의 전처리기를 사용합니다.
preprocessor = keras_hub.models.GPT2CausalLMPreprocessor.from_preset(
    "gpt2_base_en",
    sequence_length=128,
)
gpt2_lm = keras_hub.models.GPT2CausalLM.from_preset(
    "gpt2_base_en", preprocessor=preprocessor
)
```

모델이 로드되면 바로 텍스트 생성을 시작할 수 있습니다.
아래 셀을 실행하여 직접 시도해보세요.
단순히 _generate()_ 함수를 호출하기만 하면 됩니다.

```python
start = time.time()

output = gpt2_lm.generate("My trip to Yosemite was", max_length=200)
print("\nGPT-2 output:")
print(output)

end = time.time()
print(f"TOTAL TIME ELAPSED: {end - start:.2f}s")
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
GPT-2 output:
My trip to Yosemite was pretty awesome. The first time I went I didn't know how to go and it was pretty hard to get around. It was a bit like going on an adventure with a friend. The only things I could do were hike and climb the mountain. It's really cool to know you're not alone in this world. It's a lot of fun. I'm a little worried that I might not get to the top of the mountain in time to see the sunrise and sunset of the day. I think the weather is going to get a little warmer in the coming years.
```

```plain
This post is a little more in-depth on how to go on the trail. It covers how to hike on the Sierra Nevada, how to hike with the Sierra Nevada, how to hike in the Sierra Nevada, how to get to the top of the mountain, and how to get to the top with your own gear.
```

```plain
The Sierra Nevada is a very popular trail in Yosemite
TOTAL TIME ELAPSED: 25.36s
```

{{% /details %}}

또다른 예제를 시도해보세요:

```python
start = time.time()

output = gpt2_lm.generate("That Italian restaurant is", max_length=200)
print("\nGPT-2 output:")
print(output)

end = time.time()
print(f"TOTAL TIME ELAPSED: {end - start:.2f}s")
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
GPT-2 output:
That Italian restaurant is known for its delicious food, and the best part is that it has a full bar, with seating for a whole host of guests. And that's only because it's located at the heart of the neighborhood.
```

```plain
The menu at the Italian restaurant is pretty straightforward:
```

```plain
The menu consists of three main dishes:
```

```plain
Italian sausage
```

```plain
Bolognese
```

```plain
Sausage
```

```plain
Bolognese with cheese
```

```plain
Sauce with cream
```

```plain
Italian sausage with cheese
```

```plain
Bolognese with cheese
```

```plain
And the main menu consists of a few other things.
```

```plain
There are two tables: the one that serves a menu of sausage and bolognese with cheese (the one that serves the menu of sausage and bolognese with cheese) and the one that serves the menu of sausage and bolognese with cheese. The two tables are also open 24 hours a day, 7 days a week.
```

```plain
TOTAL TIME ELAPSED: 1.55s
```

{{% /details %}}

두 번째 호출이 훨씬 빠른 것을 확인할 수 있습니다.
이는 첫 번째 실행에서 계산 그래프가 [XLA 컴파일](https://www.tensorflow.org/xla)되었고,
그 이후로는 백그라운드(behind the scenes)에서 재사용되기 때문입니다.

생성된 텍스트의 품질이 괜찮아 보이지만, 이를 개선하기 위해 파인 튜닝을 할 수 있습니다.

## KerasHub의 GPT-2 모델에 대해 더 알아보기 {#more-on-the-gpt-2-model-from-kerashub}

이제 모델의 파라미터를 업데이트하기 위해 실제로 파인 튜닝을 진행할 예정이지만,
그 전에 GPT-2와 함께 작업할 수 있는 도구들을 살펴보겠습니다.

GPT-2 코드 전체는 [여기](https://github.com/keras-team/keras-hub/blob/master/keras_hub/models/gpt2/)에서 확인할 수 있습니다.
개념적으로 `GPT2CausalLM`은 KerasHub의 여러 모듈로 계층적으로 나누어질 수 있으며,
모두 _from_preset()_ 함수를 통해 사전 트레이닝된 모델을 로드할 수 있습니다:

- `keras_hub.models.GPT2Tokenizer`: GPT-2 모델에서 사용되는 토크나이저로, [byte-pair encoder](https://huggingface.co/course/chapter6/5?fw=pt)를 사용합니다.
- [`keras_hub.models.GPT2CausalLMPreprocessor`]({{< relref "/docs/api/keras_hub/models/gpt2/gpt2_causal_lm_preprocessor#gpt2causallmpreprocessor-class" >}}): GPT-2 Causal LM 트레이닝에 사용되는 전처리기입니다. 토크나이징을 비롯해 레이블 생성 및 종료 토큰 추가와 같은 작업을 수행합니다.
- [`keras_hub.models.GPT2Backbone`]({{< relref "/docs/api/keras_hub/models/gpt2/gpt2_backbone#gpt2backbone-class" >}}): GPT-2 모델로, [`keras_hub.layers.TransformerDecoder`]({{< relref "/docs/api/keras_hub/modeling_layers/transformer_decoder#transformerdecoder-class" >}})의 스택입니다. 이는 보통 `GPT-2`로 불립니다.
- [`keras_hub.models.GPT2CausalLM`]({{< relref "/docs/api/keras_hub/models/gpt2/gpt2_causal_lm#gpt2causallm-class" >}}): `GPT2Backbone`을 감싸며, `GPT2Backbone`의 출력을 임베딩 행렬과 곱하여 어휘 토큰에 대한 로그 확률을 생성합니다.

## Reddit 데이터셋으로 파인 튜닝하기 {#finetune-on-reddit-dataset}

이제 KerasHub의 GPT-2 모델에 대한 지식을 바탕으로,
모델을 파인 튜닝하여 특정 스타일로 텍스트를 생성하도록 만들 수 있습니다.
예를 들어, 짧거나 긴, 엄격하거나 캐주얼한 스타일로 텍스트를 생성하게 할 수 있습니다.
이 튜토리얼에서는 Reddit 데이터셋을 예시로 사용합니다.

```python
import tensorflow_datasets as tfds

reddit_ds = tfds.load("reddit_tifu", split="train", as_supervised=True)
```

Reddit TensorFlow Dataset의 샘플 데이터를 살펴보겠습니다. 두 가지 특징이 있습니다:

- **document**: 게시물의 텍스트.
- **title**: 제목.

```python
for document, title in reddit_ds:
    print(document.numpy())
    print(title.numpy())
    break
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
b"me and a friend decided to go to the beach last sunday. we loaded up and headed out. we were about half way there when i decided that i was not leaving till i had seafood. \n\nnow i'm not talking about red lobster. no friends i'm talking about a low country boil. i found the restaurant and got directions. i don't know if any of you have heard about the crab shack on tybee island but let me tell you it's worth it. \n\nwe arrived and was seated quickly. we decided to get a seafood sampler for two and split it. the waitress bought it out on separate platters for us. the amount of food was staggering. two types of crab, shrimp, mussels, crawfish, andouille sausage, red potatoes, and corn on the cob. i managed to finish it and some of my friends crawfish and mussels. it was a day to be a fat ass. we finished paid for our food and headed to the beach. \n\nfunny thing about seafood. it runs through me faster than a kenyan \n\nwe arrived and walked around a bit. it was about 45min since we arrived at the beach when i felt a rumble from the depths of my stomach. i ignored it i didn't want my stomach to ruin our fun. i pushed down the feeling and continued. about 15min later the feeling was back and stronger than before. again i ignored it and continued. 5min later it felt like a nuclear reactor had just exploded in my stomach. i started running. i yelled to my friend to hurry the fuck up. \n\nrunning in sand is extremely hard if you did not know this. we got in his car and i yelled at him to floor it. my stomach was screaming and if he didn't hurry i was gonna have this baby in his car and it wasn't gonna be pretty. after a few red lights and me screaming like a woman in labor we made it to the store. \n\ni practically tore his car door open and ran inside. i ran to the bathroom opened the door and barely got my pants down before the dam burst and a flood of shit poured from my ass. \n\ni finished up when i felt something wet on my ass. i rubbed it thinking it was back splash. no, mass was covered in the after math of me abusing the toilet. i grabbed all the paper towels i could and gave my self a whores bath right there. \n\ni sprayed the bathroom down with the air freshener and left. an elderly lady walked in quickly and closed the door. i was just about to walk away when i heard gag. instead of walking i ran. i got to the car and told him to get the hell out of there."
b'liking seafood'
```

{{% /details %}}

이 경우, 언어 모델에서 다음 단어 예측 작업을 수행하고 있으므로, 'document' 피처만 필요합니다.

```python
train_ds = (
    reddit_ds.map(lambda document, _: document)
    .batch(32)
    .cache()
    .prefetch(tf.data.AUTOTUNE)
)
```

이제 익숙한 _fit()_ 함수를 사용하여 모델을 파인 튜닝할 수 있습니다.
`GPT2CausalLM`이 [`keras_hub.models.Task`]({{< relref "/docs/api/keras_hub/base_classes/task#task-class" >}}) 인스턴스이기 때문에,
`fit` 메서드 내에서 `preprocessor`가 자동으로 호출됩니다.

이 단계는 GPU 메모리를 많이 사용하며, 전체 트레이닝을 완료하려면 시간이 꽤 걸립니다.
여기서는 데모 목적으로 데이터셋의 일부만 사용합니다.

```python
train_ds = train_ds.take(500)
num_epochs = 1

# 선형적으로 감소하는 학습률.
learning_rate = keras.optimizers.schedules.PolynomialDecay(
    5e-5,
    decay_steps=train_ds.cardinality() * num_epochs,
    end_learning_rate=0.0,
)
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
gpt2_lm.compile(
    optimizer=keras.optimizers.Adam(learning_rate),
    loss=loss,
    weighted_metrics=["accuracy"],
)

gpt2_lm.fit(train_ds, epochs=num_epochs)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
 500/500 ━━━━━━━━━━━━━━━━━━━━ 75s 120ms/step - accuracy: 0.3189 - loss: 3.3653

<keras.src.callbacks.history.History at 0x7f2af3fda410>
```

{{% /details %}}

파인 튜닝이 완료된 후에는, 동일한 _generate()_ 함수를 사용하여 텍스트를 다시 생성할 수 있습니다.
이번에는 텍스트가 Reddit 작성 스타일에 더 가까워지고, 생성되는 길이도 트레이닝 세트에서 설정한 길이에 가깝게 됩니다.

```python
start = time.time()

output = gpt2_lm.generate("I like basketball", max_length=200)
print("\nGPT-2 output:")
print(output)

end = time.time()
print(f"TOTAL TIME ELAPSED: {end - start:.2f}s")
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
GPT-2 output:
I like basketball. it has the greatest shot of all time and the best shot of all time. i have to play a little bit more and get some practice time.
```

```plain
today i got the opportunity to play in a tournament in a city that is very close to my school so i was excited to see how it would go. i had just been playing with a few other guys, so i thought i would go and play a couple games with them.
```

```plain
after a few games i was pretty confident and confident in myself. i had just gotten the opportunity and had to get some practice time.
```

```plain
so i go to the
TOTAL TIME ELAPSED: 21.13s
```

{{% /details %}}

## 샘플링 방법으로 들어가기 {#into-the-sampling-method}

KerasHub에서는 contrastive search, Top-K, beam sampling 등의 몇 가지 샘플링 방법을 제공합니다.
기본적으로 `GPT2CausalLM`은 Top-K 검색을 사용하지만, 사용자가 원하는 샘플링 방법을 선택할 수도 있습니다.

Optimizer와 activation 함수처럼, 커스텀 샘플러를 지정하는 방법에는 두 가지가 있습니다:

- 문자열 식별자를 사용합니다. 예를 들어 "greedy"와 같이 하면 기본 구성을 사용하게 됩니다.
- [`keras_hub.samplers.Sampler`]({{< relref "/docs/api/keras_hub/samplers/samplers#sampler-class" >}}) 인스턴스 전달을 통해,
  커스텀 구성을 사용할 수 있습니다.

```python
# 문자열 식별자를 사용합니다.
gpt2_lm.compile(sampler="top_k")
output = gpt2_lm.generate("I like basketball", max_length=200)
print("\nGPT-2 output:")
print(output)

# `Sampler` 인스턴스를 사용합니다.
# `GreedySampler`는 스스로 반복되는 경향이 있습니다.
greedy_sampler = keras_hub.samplers.GreedySampler()
gpt2_lm.compile(sampler=greedy_sampler)

output = gpt2_lm.generate("I like basketball", max_length=200)
print("\nGPT-2 output:")
print(output)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
GPT-2 output:
I like basketball, and this is a pretty good one.
```

```plain
first off, my wife is pretty good, she is a very good basketball player and she is really, really good at playing basketball.
```

```plain
she has an amazing game called basketball, it is a pretty fun game.
```

```plain
i play it on the couch.  i'm sitting there, watching the game on the couch.  my wife is playing with her phone.  she's playing on the phone with a bunch of people.
```

```plain
my wife is sitting there and watching basketball.  she's sitting there watching
```

```plain
GPT-2 output:
I like basketball, but i don't like to play it.
```

```plain
so i was playing basketball at my local high school, and i was playing with my friends.
```

```plain
i was playing with my friends, and i was playing with my brother, who was playing basketball with his brother.
```

```plain
so i was playing with my brother, and he was playing with his brother's brother.
```

```plain
so i was playing with my brother, and he was playing with his brother's brother.
```

```plain
so i was playing with my brother, and he was playing with his brother's brother.
```

```plain
so i was playing with my brother, and he was playing with his brother's brother.
```

```plain
so i was playing with my brother, and he was playing with his brother's brother.
```

```plain
so i was playing with my brother, and he was playing with his brother
```

{{% /details %}}

KerasHub `Sampler` 클래스에 대한 자세한 내용은,
[여기](https://github.com/keras-team/keras-hub/tree/master/keras_hub/samplers)에서 코드를 확인할 수 있습니다.

## 중국 시 데이터셋으로 파인 튜닝하기 {#finetune-on-chinese-poem-dataset}

GPT-2를 비영어권 데이터셋에서도 파인 튜닝할 수 있습니다.
중국어를 아는 독자들을 위해, 이 섹션에서는 GPT-2를 중국 시 데이터셋으로 파인 튜닝하여 모델을 시인으로 만드는 방법을 설명합니다!

GPT-2는 byte-pair encoder를 사용하며,
원래의 사전 트레이닝 데이터셋에는 일부 중국어 문자가 포함되어 있기 때문에,
원래의 vocab을 사용하여 중국어 데이터셋에서 파인 튜닝을 할 수 있습니다.

```python
!# 중국 시 데이터셋 로드
!git clone https://github.com/chinese-poetry/chinese-poetry.git
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Cloning into 'chinese-poetry'...
```

{{% /details %}}

JSON 파일에서 텍스트를 불러옵니다. 데모 목적을 위해 《全唐诗》만 사용합니다.

```python
import os
import json

poem_collection = []
for file in os.listdir("chinese-poetry/全唐诗"):
    if ".json" not in file or "poet" not in file:
        continue
    full_filename = "%s/%s" % ("chinese-poetry/全唐诗", file)
    with open(full_filename, "r") as f:
        content = json.load(f)
        poem_collection.extend(content)

paragraphs = ["".join(data["paragraphs"]) for data in poem_collection]
```

샘플 데이터를 확인해 보겠습니다.

```python
print(paragraphs[0])
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
毋謂支山險，此山能幾何。崎嶔十年夢，知歷幾蹉跎。
```

{{% /details %}}

Reddit 예제와 유사하게, TF 데이터셋으로 변환한 후, 일부 데이터만 사용하여 트레이닝합니다.

```python
train_ds = (
    tf.data.Dataset.from_tensor_slices(paragraphs)
    .batch(16)
    .cache()
    .prefetch(tf.data.AUTOTUNE)
)

# 전체 데이터셋을 처리하는 데 시간이 오래 걸리므로,
# 데모 목적을 위해 `500`개의 데이터를 사용하고, 1번의 에포크를 실행합니다.
train_ds = train_ds.take(500)
num_epochs = 1

learning_rate = keras.optimizers.schedules.PolynomialDecay(
    5e-4,
    decay_steps=train_ds.cardinality() * num_epochs,
    end_learning_rate=0.0,
)
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
gpt2_lm.compile(
    optimizer=keras.optimizers.Adam(learning_rate),
    loss=loss,
    weighted_metrics=["accuracy"],
)

gpt2_lm.fit(train_ds, epochs=num_epochs)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
 500/500 ━━━━━━━━━━━━━━━━━━━━ 49s 71ms/step - accuracy: 0.2357 - loss: 2.8196

<keras.src.callbacks.history.History at 0x7f2b2c192bc0>
```

{{% /details %}}

결과를 확인해 봅시다!

```python
output = gpt2_lm.generate("昨夜雨疏风骤", max_length=200)
print(output)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
昨夜雨疏风骤，爲臨江山院短靜。石淡山陵長爲羣，臨石山非處臨羣。美陪河埃聲爲羣，漏漏漏邊陵塘
```

{{% /details %}}

나쁘지 않네요 😀
