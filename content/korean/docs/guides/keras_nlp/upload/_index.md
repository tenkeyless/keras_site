---
title: KerasNLP로 모델 업로드
linkTitle: KerasNLP로 모델 업로드
toc: true
weight: 3
type: docs
---

{{< keras/original checkedAt="2024-11-19" >}}

**{{< t f_author >}}** [Samaneh Saadat](https://github.com/SamanehSaadat/), [Matthew Watson](https://github.com/mattdangerw/)  
**{{< t f_date_created >}}** 2024/04/29  
**{{< t f_last_modified >}}** 2024/04/29  
**{{< t f_description >}}** 미세 조정된 KerasNLP 모델을 모델 허브에 업로드하는 방법에 대한 소개.

{{< cards cols="2" >}}
{{< card link="https://colab.research.google.com/github/keras-team/keras-io/blob/master/guides/ipynb/keras_nlp/upload.ipynb" title="Colab" tag="Colab" tagType="warning">}}
{{< card link="https://github.com/keras-team/keras-io/blob/master/guides/keras_nlp/upload.py" title="GitHub" tag="GitHub">}}
{{< /cards >}}

## 소개 {#introduction}

머신 러닝 모델을 특정 작업에 맞게 파인 튜닝하면 인상적인 결과를 얻을 수 있습니다.
파인 튜닝된 모델을 모델 허브에 업로드하면 더 넓은 커뮤니티와 공유할 수 있으며,
이를 통해 다른 연구자들과 개발자들이 접근성을 높일 수 있습니다.
또한, 이를 통해 실세계 애플리케이션에 모델을 통합하는 과정이 간소화될 수 있습니다.

이 가이드에서는 파인 튜닝된 모델을 [Kaggle Models](https://www.kaggle.com/models)와
[Hugging Face Hub](https://huggingface.co/models)와 같은 인기 있는 모델 허브에 업로드하는 방법을 설명합니다.

## 셋업 {#setup}

우선 필요한 라이브러리를 설치하고 import 합니다.
이 가이드에서는 KerasNLP를 사용합니다.

```python
!pip install -q --upgrade keras-nlp huggingface-hub kagglehub
```

```python
import os

os.environ["KERAS_BACKEND"] = "jax"

import keras_nlp
```

## 데이터 {#data}

이 가이드에서는 IMDB 리뷰 데이터셋을 사용합니다. `tensorflow_dataset`에서 데이터셋을 불러옵니다.

```python
import tensorflow_datasets as tfds

imdb_train, imdb_test = tfds.load(
    "imdb_reviews",
    split=["train", "test"],
    as_supervised=True,
    batch_size=4,
)
```

가이드 실행 속도를 높이기 위해 트레이닝 샘플 중 일부만 사용합니다.
하지만, 더 높은 품질의 모델이 필요하다면, 더 많은 트레이닝 샘플을 사용하는 것이 좋습니다.

```python
imdb_train = imdb_train.take(100)
```

## 작업 업로드 {#task-upload}

[`keras_nlp.models.Task`]({{< relref "/docs/api/keras_nlp/base_classes/task#task-class" >}})는 [`keras_nlp.models.Backbone`]({{< relref "/docs/api/keras_nlp/base_classes/backbone#backbone-class" >}})과 [`keras_nlp.models.Preprocessor`]({{< relref "/docs/api/keras_nlp/base_classes/preprocessor#preprocessor-class" >}})를 결합하여,
텍스트 문제에 대해 직접 트레이닝, 파인 튜닝 및 예측에 사용할 수 있는 모델을 생성합니다.
이 섹션에서는, `Task`를 생성하고, 파인 튜닝하여 모델 허브에 업로드하는 방법을 설명합니다.

### 모델 로드 {#load-model}

베이스 모델을 기반으로 Causal LM을 구축하려면,
[`keras_nlp.models.CausalLM.from_preset`]({{< relref "/docs/api/keras_nlp/base_classes/causal_lm#from_preset-method" >}})를 호출하고,
미리 설정된 식별자(built-in preset identifier)를 전달하면 됩니다.

```python
causal_lm = keras_nlp.models.CausalLM.from_preset("gpt2_base_en")
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Downloading from https://www.kaggle.com/api/v1/models/keras/gpt2/keras/gpt2_base_en/2/download/task.json...

Downloading from https://www.kaggle.com/api/v1/models/keras/gpt2/keras/gpt2_base_en/2/download/preprocessor.json...
```

{{% /details %}}

### 모델 미세 조정 {#fine-tune-model}

모델을 불러온 후, `.fit()`을 호출하여 파인 튜닝할 수 있습니다.
여기에서는 IMDB 리뷰에 모델을 파인 튜닝하여 모델을 영화 도메인에 맞게 최적화합니다.

```python
# Causal LM을 위해 레이블을 제거하고 리뷰 텍스트만 유지합니다.
imdb_train_reviews = imdb_train.map(lambda x, y: x)

# Causal LM을 파인 튜닝합니다.
causal_lm.fit(imdb_train_reviews)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
100/100 ━━━━━━━━━━━━━━━━━━━━ 151s 1s/step - loss: 1.0198 - sparse_categorical_accuracy: 0.3271
```

{{% /details %}}

### 모델을 로컬에 저장하기 {#save-the-model-locally}

모델을 업로드하려면, 먼저 `save_to_preset`을 사용하여 모델을 로컬에 저장해야 합니다.

```python
preset_dir = "./gpt2_imdb"
causal_lm.save_to_preset(preset_dir)
```

저장된 파일을 확인해 봅시다.

```python
os.listdir(preset_dir)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
['preprocessor.json',
 'tokenizer.json',
 'task.json',
 'model.weights.h5',
 'config.json',
 'metadata.json',
 'assets']
```

{{% /details %}}

#### 로컬에 저장된 모델 불러오기 {#load-a-locally-saved-model}

로컬에 저장된 프리셋 모델은 `from_preset`을 사용하여 불러올 수 있습니다.
저장된 그대로 불러올 수 있습니다.

```python
causal_lm = keras_nlp.models.CausalLM.from_preset(preset_dir)
```

또한 [`keras_nlp.models.Backbone`]({{< relref "/docs/api/keras_nlp/base_classes/backbone#backbone-class" >}}) 및 `keras_nlp.models.Tokenizer` 객체를
프리셋 디렉토리에서 불러올 수 있습니다.
이 객체들은 위에서 사용한 `causal_lm.backbone` 및
`causal_lm.preprocessor.tokenizer`와 동일합니다.

```python
backbone = keras_nlp.models.Backbone.from_preset(preset_dir)
tokenizer = keras_nlp.models.Tokenizer.from_preset(preset_dir)
```

### 모델 허브에 모델 업로드 {#upload-the-model-to-a-model-hub}

프리셋을 디렉토리에 저장한 후,
이 디렉토리는 KerasNLP 라이브러리를 통해
Kaggle 또는 Hugging Face와 같은 모델 허브에 직접 업로드할 수 있습니다.
Kaggle에 모델을 업로드하려면 URI가 `kaggle://`로 시작해야 하고,
Hugging Face에 업로드하려면 `hf://`로 시작해야 합니다.

#### Kaggle에 업로드 {#upload-to-kaggle}

Kaggle에 모델을 업로드하려면 먼저 Kaggle에 인증해야 합니다.
다음 중 하나의 방법을 사용할 수 있습니다:

1. 환경 변수 `KAGGLE_USERNAME`과 `KAGGLE_KEY`를 설정합니다.
2. 로컬에 `~/.kaggle/kaggle.json` 파일을 제공합니다.
3. `kagglehub.login()`을 호출합니다.

계속하기 전에 로그인 상태를 확인해 봅시다.

```python
import kagglehub

if "KAGGLE_USERNAME" not in os.environ or "KAGGLE_KEY" not in os.environ:
    kagglehub.login()
```

모델을 업로드하려면 `keras_nlp.upload_preset(uri, preset_dir)` API를 사용할 수 있으며,
Kaggle에 업로드할 경우 `uri`의 형식은
`kaggle://<KAGGLE_USERNAME>/<MODEL>/Keras/<VARIATION>`입니다.
`preset_dir`은 모델이 저장된 디렉토리입니다.

다음 명령을 실행하면 `preset_dir`에 저장된 모델을 Kaggle에 업로드할 수 있습니다:

```python
kaggle_username = kagglehub.whoami()["username"]
kaggle_uri = f"kaggle://{kaggle_username}/gpt2/keras/gpt2_imdb"
keras_nlp.upload_preset(kaggle_uri, preset_dir)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Upload successful: preprocessor.json (834B)
Upload successful: tokenizer.json (322B)
Upload successful: task.json (2KB)
Upload successful: model.weights.h5 (475MB)
Upload successful: config.json (431B)
Upload successful: metadata.json (142B)
Upload successful: merges.txt (446KB)
Upload successful: vocabulary.json (1018KB)

Your model instance version has been created.
```

{{% /details %}}

#### Hugging Face에 업로드 {#upload-to-hugging-face}

Hugging Face에 모델을 업로드하려면 먼저 Hugging Face에 인증해야 합니다.
다음 중 하나의 방법을 사용할 수 있습니다:

1. 환경 변수 `HF_USERNAME`과 `HF_TOKEN`을 설정합니다.
2. `huggingface_hub.notebook_login()`을 호출합니다.

계속하기 전에 로그인 상태를 확인해 봅시다.

```python
import huggingface_hub

if "HF_USERNAME" not in os.environ or "HF_TOKEN" not in os.environ:
    huggingface_hub.notebook_login()
```

`keras_nlp.upload_preset(uri, preset_dir)`는
Hugging Face에 모델을 업로드할 때 사용할 수 있으며,
이 경우 `uri` 형식은 `hf://<HF_USERNAME>/<MODEL>`입니다.

다음 명령을 실행하면 `preset_dir`에 저장된 모델을 Hugging Face에 업로드할 수 있습니다:

```python
hf_username = huggingface_hub.whoami()["name"]
hf_uri = f"hf://{hf_username}/gpt2_imdb"
keras_nlp.upload_preset(hf_uri, preset_dir)
```

### 사용자가 업로드한 모델 불러오기 {#load-a-user-uploaded-model}

Kaggle에 모델이 업로드된 것을 확인한 후, `from_preset`을 호출하여 모델을 불러올 수 있습니다.

```python
causal_lm = keras_nlp.models.CausalLM.from_preset(
    f"kaggle://{kaggle_username}/gpt2/keras/gpt2_imdb"
)
```

Hugging Face에 업로드된 모델도 `from_preset`을 호출하여 불러올 수 있습니다.

```python
causal_lm = keras_nlp.models.CausalLM.from_preset(f"hf://{hf_username}/gpt2_imdb")
```

## 분류기 업로드 {#classifier-upload}

분류기 모델 업로드는 Causal LM 업로드와 유사합니다.
미세 조정된 모델을 업로드하려면, 먼저 `save_to_preset` API를 사용하여 모델을 로컬 디렉토리에 저장한 후,
[`keras_nlp.upload_preset`]({{< relref "/docs/api/keras_nlp/base_classes/upload_preset#upload_preset-function" >}})를 통해 업로드할 수 있습니다.

```python
# 베이스 모델 로드.
classifier = keras_nlp.models.Classifier.from_preset(
    "bert_tiny_en_uncased", num_classes=2
)

# 분류기 미세 조정.
classifier.fit(imdb_train)

# 로컬 프리셋 디렉토리에 모델 저장.
preset_dir = "./bert_tiny_imdb"
classifier.save_to_preset(preset_dir)

# Kaggle에 업로드.
keras_nlp.upload_preset(
    f"kaggle://{kaggle_username}/bert/keras/bert_tiny_imdb", preset_dir
)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
100/100 ━━━━━━━━━━━━━━━━━━━━ 7s 31ms/step - loss: 0.6975 - sparse_categorical_accuracy: 0.5164
```

```plain
Upload successful: preprocessor.json (947B)
Upload successful: tokenizer.json (461B)
Upload successful: task.json (2KB)
Upload successful: task.weights.h5 (50MB)
Upload successful: model.weights.h5 (17MB)
Upload successful: config.json (454B)
Upload successful: metadata.json (140B)
Upload successful: vocabulary.txt (226KB)

Your model instance version has been created.
```

{{% /details %}}

모델이 Kaggle에 업로드된 것을 확인한 후, `from_preset`을 호출하여 모델을 불러올 수 있습니다.

```python
classifier = keras_nlp.models.Classifier.from_preset(
    f"kaggle://{kaggle_username}/bert/keras/bert_tiny_imdb"
)
```
