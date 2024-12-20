---
title: LoRA 및 QLoRA를 사용한 Gemma의 매개변수 효율적인 미세 조정
linkTitle: LoRA 및 QLoRA로 Gemma의 매개변수 효율적 미세 조정
toc: true
weight: 1
type: docs
---

{{< keras/original checkedAt="2024-11-24" >}}

**{{< t f_author >}}** [Hongyu Chiu](https://github.com/james77777778), [Abheesht Sharma](https://github.com/abheesht17/), [Matthew Watson](https://github.com/mattdangerw/)  
**{{< t f_date_created >}}** 2024/08/06  
**{{< t f_last_modified >}}** 2024/08/06  
**{{< t f_description >}}** Use KerasHub to fine-tune a Gemma LLM with LoRA and QLoRA.

{{< keras/version v=3 >}}

{{< cards cols="2" >}}
{{< card link="https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/keras_recipes/ipynb/parameter_efficient_finetuning_of_gemma_with_lora_and_qlora.ipynb" title="Colab" tag="Colab" tagType="warning">}}
{{< card link="https://github.com/keras-team/keras-io/blob/master/examples/keras_recipes/parameter_efficient_finetuning_of_gemma_with_lora_and_qlora.py" title="GitHub" tag="GitHub">}}
{{< /cards >}}

## Introduction {#introduction}

Large Language Models (LLMs) have been shown to be effective at a variety of NLP tasks. An LLM is first pre-trained on a large corpus of text in a self-supervised fashion. Pre-training helps LLMs learn general-purpose knowledge, such as statistical relationships between words. An LLM can then be fine-tuned on a downstream task of interest (such as sentiment analysis).

However, LLMs are extremely large in size, and we don't need to train all the parameters in the model while fine-tuning, especially because datasets on which the model is fine-tuned are relatively small. Another way of saying this is that LLMs are over-parametrized for fine-tuning. This is where [Low-Rank Adaptation (LoRA)](https://arxiv.org/abs/2106.09685) comes in; it significantly reduces the number of trainable parameters. This results in a decrease in training time and GPU memory usage, while maintaining the quality of the outputs.

Furthermore, [Quantized Low-Rank Adaptation (QLoRA)](https://arxiv.org/abs/2305.14314) extends LoRA to enhance efficiency through quantization techniques without performance degradation.

In this example, we will fine-tune KerasHub's [Gemma model](https://keras.io/api/keras_hub/models/gemma/) on the next token prediction task using LoRA and QLoRA.

Note that this example runs on all backends supported by Keras. TensorFlow is only used for data preprocessing.

## Setup {#setup}

Before we start implementing the pipeline, let's install and import all the libraries we need. We'll be using the KerasHub library.

Secondly, let's set the precision to bfloat16. This will help us reduce the memory usage and training time.

Also, ensure that `KAGGLE_USERNAME` and `KAGGLE_KEY` have been correctly configured to access the Gemma model.

```python
# We might need the latest code from Keras and KerasHub
!pip install -q git+https://github.com/keras-team/keras.git git+https://github.com/keras-team/keras-hub.git
```

```python
import gc
import os

os.environ["KERAS_BACKEND"] = "jax"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress verbose logging from TF

# os.environ["KAGGLE_USERNAME"] = "..."
# os.environ["KAGGLE_KEY"] = "..."

import keras
import keras_hub
import tensorflow as tf
import tensorflow_datasets as tfds

keras.config.set_dtype_policy("bfloat16")
```

## Dataset {#dataset}

We will use the MTNT (Machine Translation of Noisy Text) dataset, which is available from TensorFlow Datasets. In this example, we will use the French-to-English portion of the dataset.

```python
train_ds = tfds.load("mtnt/fr-en", split="train")
```

We can print some samples. Each sample in the dataset contains two entries:

- src: the original French sentence.
- dst: the corresponding English translation.

```python
examples = train_ds.take(3)
examples = examples.as_numpy_iterator()

for idx, example in enumerate(examples):
    print(f"Example {idx}:")
    for key, val in example.items():
        print(f"{key}: {val}")
    print()
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Example 0:
dst: b'Yep, serious...'
src: b"Le journal l'est peut-\xc3\xaatre, mais m\xc3\xaame moi qui suit de droite je les trouve limite de temps en temps..."
```

```plain
Example 1:
dst: b'Finally, I explained to you in what context this copy-pasting is relevant: when we are told padamalgame etc.'
src: b"Enfin je t'ai expliqu\xc3\xa9 dans quel cadre ce copypasta est pertinent : quand on nous dit padamalgame etc."
```

```plain
Example 2:
dst: b'Gift of Ubiquity: Fran\xc3\xa7ois Baroin is now advisor to the Barclays Bank, mayor, president of the agglomeration, professor at HEC Paris, president of the Association of Mayors of France and Advocate Counselor, it must take him half a day each month.'
src: b"Don d'Ubiquit\xc3\xa9 : Fran\xc3\xa7ois Baroin est d\xc3\xa9sormais conseiller \xc3\xa0 la Banque Barclays, maire, pr\xc3\xa9sident d'agglom\xc3\xa9ration, professeur \xc3\xa0 HEC Paris, pr\xc3\xa9sident de l'association des maires de France et avocat  Conseiller, \xc3\xa7a doit lui prendre une demi journ\xc3\xa9e par mois."
```

{{% /details %}}

Since we will fine-tune our model to perform a French-to-English translation task, we should format the inputs for instruction tuning. For example, we could format the translation task in this example like:

```plain
<start_of_turn>user
Translate French into English:
{src}<end_of_turn>
<start_of_turn>model
{dst}<end_of_turn>
```

The special tokens such as `<start_of_turn>user`, `<start_of_turn>model` and `<end_of_turn>` are used for Gemma models. You can learn more from https://ai.google.dev/gemma/docs/formatting

```python
train_ds = train_ds.map(
    lambda x: tf.strings.join(
        [
            "<start_of_turn>user\n",
            "Translate French into English:\n",
            x["src"],
            "<end_of_turn>\n",
            "<start_of_turn>model\n",
            "Translation:\n",
            x["dst"],
            "<end_of_turn>",
        ]
    )
)
examples = train_ds.take(3)
examples = examples.as_numpy_iterator()

for idx, example in enumerate(examples):
    print(f"Example {idx}:")
    print(example)
    print()
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Example 0:
b"<start_of_turn>user\nTranslate French into English:\nLe journal l'est peut-\xc3\xaatre, mais m\xc3\xaame moi qui suit de droite je les trouve limite de temps en temps...<end_of_turn>\n<start_of_turn>model\nTranslation:\nYep, serious...<end_of_turn>"
```

```plain
Example 1:
b"<start_of_turn>user\nTranslate French into English:\nEnfin je t'ai expliqu\xc3\xa9 dans quel cadre ce copypasta est pertinent : quand on nous dit padamalgame etc.<end_of_turn>\n<start_of_turn>model\nTranslation:\nFinally, I explained to you in what context this copy-pasting is relevant: when we are told padamalgame etc.<end_of_turn>"
```

```plain
Example 2:
b"<start_of_turn>user\nTranslate French into English:\nDon d'Ubiquit\xc3\xa9 : Fran\xc3\xa7ois Baroin est d\xc3\xa9sormais conseiller \xc3\xa0 la Banque Barclays, maire, pr\xc3\xa9sident d'agglom\xc3\xa9ration, professeur \xc3\xa0 HEC Paris, pr\xc3\xa9sident de l'association des maires de France et avocat  Conseiller, \xc3\xa7a doit lui prendre une demi journ\xc3\xa9e par mois.<end_of_turn>\n<start_of_turn>model\nTranslation:\nGift of Ubiquity: Fran\xc3\xa7ois Baroin is now advisor to the Barclays Bank, mayor, president of the agglomeration, professor at HEC Paris, president of the Association of Mayors of France and Advocate Counselor, it must take him half a day each month.<end_of_turn>"
```

{{% /details %}}

We will take a subset of the dataset for the purpose of this example.

```python
train_ds = train_ds.batch(1).take(100)
```

## Model {#model}

KerasHub provides implementations of many popular model architectures. In this example, we will use `GemmaCausalLM`, an end-to-end Gemma model for causal language modeling. A causal language model predicts the next token based on previous tokens.

Note that `sequence_length` is set to `256` to speed up the fitting.

```python
preprocessor = keras_hub.models.GemmaCausalLMPreprocessor.from_preset(
    "gemma_1.1_instruct_2b_en", sequence_length=256
)
gemma_lm = keras_hub.models.GemmaCausalLM.from_preset(
    "gemma_1.1_instruct_2b_en", preprocessor=preprocessor
)
gemma_lm.summary()
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Preprocessor: "gemma_causal_lm_preprocessor"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Tokenizer (type)                                   ┃                                             Vocab # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ gemma_tokenizer (GemmaTokenizer)                   │                                             256,000 │
└────────────────────────────────────────────────────┴─────────────────────────────────────────────────────┘
Model: "gemma_causal_lm"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                  ┃ Output Shape              ┃         Param # ┃ Connected to               ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ padding_mask (InputLayer)     │ (None, None)              │               0 │ -                          │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ token_ids (InputLayer)        │ (None, None)              │               0 │ -                          │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ gemma_backbone                │ (None, None, 2048)        │   2,506,172,416 │ padding_mask[0][0],        │
│ (GemmaBackbone)               │                           │                 │ token_ids[0][0]            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ token_embedding               │ (None, None, 256000)      │     524,288,000 │ gemma_backbone[0][0]       │
│ (ReversibleEmbedding)         │                           │                 │                            │
└───────────────────────────────┴───────────────────────────┴─────────────────┴────────────────────────────┘
 Total params: 2,506,172,416 (4.67 GB)
 Trainable params: 2,506,172,416 (4.67 GB)
 Non-trainable params: 0 (0.00 B)
```

{{% /details %}}

## LoRA Fine-tuning {#lora-fine-tuning}

### What exactly is LoRA? {#what-exactly-is-lora}

Low-rank adaptation (LoRA) is a parameter-efficient fine-tuning technique for LLMs. It freezes the weights of the LLM, and injects trainable rank-decomposition matrices. Let's understand this more clearly.

Assume we have an `n x n` pre-trained dense layer (or weight matrix), `W0`. We initialize two dense layers, `A` and `B`, of shapes `n x rank`, and `rank x n`, respectively. `rank` is much smaller than `n`. In the paper, values between 1 and 4 are shown to work well.

### LoRA equation {#lora-equation}

The original equation is `output = W0x + b0`, where `x` is the input, `W0` and `b0` are the weight matrix and bias terms of the original dense layer (frozen). The LoRA equation is: `output = W0x + b0 + BAx`, where `A` and `B` are the rank-decomposition matrices.

LoRA is based on the idea that updates to the weights of the pre-trained language model have a low "intrinsic rank" since pre-trained language models are over-parametrized. Predictive performance of full fine-tuning can be replicated even by constraining `W0`'s updates to low-rank decomposition matrices.

### Number of trainable parameters {#number-of-trainable-parameters}

Let's do some quick math. Suppose `n` is 768, and `rank` is 4. `W0` has `768 x 768 = 589,824` parameters, whereas the LoRA layers, `A` and `B` together have `768 x 4 + 4 x 768 = 6,144` parameters. So, for the dense layer, we go from `589,824` trainable parameters to `6,144` trainable parameters!

### Why does LoRA reduce memory footprint? {#why-does-lora-reduce-memory-footprint}

Even though the total number of parameters increase (since we are adding LoRA layers), the memory footprint reduces, because the number of trainable parameters reduces. Let's dive deeper into this.

The memory usage of a model can be split into four parts:

- Model memory: This is the memory required to store the model weights. This will be slightly higher for LoRA than the original model.
- Forward pass memory: This mostly depends on batch size, sequence length, etc. We keep this constant for both models for a fair comparison.
- Backward pass memory: This is the memory required to store the gradients. Note that the gradients are computed only for the trainable parameters.
- Optimizer memory: This is the memory required to store the optimizer state. For example, the Adam optimizer stores the "1st moment vectors" and "2nd moment vectors" for the trainable parameters.

Since, with LoRA, there is a huge reduction in the number of trainable parameters, the optimizer memory and the memory required to store the gradients for LoRA is much less than the original model. This is where most of the memory savings happen.

### Why is LoRA so popular? {#why-is-lora-so-popular}

- Reduces GPU memory usage;
- Faster training; and
- No additional inference latency.

When using KerasHub, we can enable LoRA with an one-line API: `enable_lora(rank=4)`

From `gemma_lm.summary()`, we can see enabling LoRA reduces the number of trainable parameters significantly (from 2.5 billion to 1.3 million).

```python
gemma_lm.backbone.enable_lora(rank=4)
gemma_lm.summary()
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Preprocessor: "gemma_causal_lm_preprocessor"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Tokenizer (type)                                   ┃                                             Vocab # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ gemma_tokenizer (GemmaTokenizer)                   │                                             256,000 │
└────────────────────────────────────────────────────┴─────────────────────────────────────────────────────┘
Model: "gemma_causal_lm"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                  ┃ Output Shape              ┃         Param # ┃ Connected to               ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ padding_mask (InputLayer)     │ (None, None)              │               0 │ -                          │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ token_ids (InputLayer)        │ (None, None)              │               0 │ -                          │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ gemma_backbone                │ (None, None, 2048)        │   2,507,536,384 │ padding_mask[0][0],        │
│ (GemmaBackbone)               │                           │                 │ token_ids[0][0]            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ token_embedding               │ (None, None, 256000)      │     524,288,000 │ gemma_backbone[0][0]       │
│ (ReversibleEmbedding)         │                           │                 │                            │
└───────────────────────────────┴───────────────────────────┴─────────────────┴────────────────────────────┘
 Total params: 2,507,536,384 (4.67 GB)
 Trainable params: 1,363,968 (2.60 MB)
 Non-trainable params: 2,506,172,416 (4.67 GB)
```

{{% /details %}}

Let's fine-tune the LoRA model.

```python
# To save memory, use the SGD optimizer instead of the usual AdamW optimizer.
# For this specific example, SGD is more than enough.
optimizer = keras.optimizers.SGD(learning_rate=1e-4)
gemma_lm.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=optimizer,
    weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
)
gemma_lm.fit(train_ds, epochs=1)
```

After fine-tuning, responses will follow the instructions provided in the prompt.

```python
template = (
    "<start_of_turn>user\n"
    "Translate French into English:\n"
    "{inputs}"
    "<end_of_turn>\n"
    "<start_of_turn>model\n"
    "Translation:\n"
)
prompt = template.format(inputs="Bonjour, je m'appelle Morgane.")
outputs = gemma_lm.generate(prompt, max_length=256)
print("Translation:\n", outputs.replace(prompt, ""))
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Translation:
 Hello, my name is Morgane.
```

{{% /details %}}

Release memory.

```python
del preprocessor
del gemma_lm
del optimizer
gc.collect()
```

## QLoRA Fine-tuning {#qlora-fine-tuning}

Quantized Low-Rank Adaptation (QLoRA) extends LoRA to enhance efficiency by quantizing the model weights from high precision data types, such as float32, to lower precision data types like int8. This leads to reduced memory usage and faster computation. The saved model weights are also much smaller.

Note that the QLoRA implementation here is a simplified version compared to the original. The differences are:

- The 4-bit NormalFloat format is not used because no backend supports it.
- No double quantization.
- No Paged optimizer.

To enable QLoRA in KerasHub, follow these steps:

1.  Instantiate the model.
2.  Quantize the weights using dynamic int8 quantization.
3.  Enable LoRA.

Steps 2 and 3 are achieved with one-line APIs:

- `quantize("int8")`
- `enable_lora(...)`

```python
preprocessor = keras_hub.models.GemmaCausalLMPreprocessor.from_preset(
    "gemma_1.1_instruct_2b_en", sequence_length=256
)
gemma_lm = keras_hub.models.GemmaCausalLM.from_preset(
    "gemma_1.1_instruct_2b_en", preprocessor=preprocessor
)
gemma_lm.quantize("int8")
gemma_lm.backbone.enable_lora(rank=4)
gemma_lm.summary()
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Preprocessor: "gemma_causal_lm_preprocessor_1"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Tokenizer (type)                                   ┃                                             Vocab # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ gemma_tokenizer (GemmaTokenizer)                   │                                             256,000 │
└────────────────────────────────────────────────────┴─────────────────────────────────────────────────────┘
Model: "gemma_causal_lm_1"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                  ┃ Output Shape              ┃         Param # ┃ Connected to               ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ padding_mask (InputLayer)     │ (None, None)              │               0 │ -                          │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ token_ids (InputLayer)        │ (None, None)              │               0 │ -                          │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ gemma_backbone                │ (None, None, 2048)        │   2,508,502,016 │ padding_mask[0][0],        │
│ (GemmaBackbone)               │                           │                 │ token_ids[0][0]            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ token_embedding               │ (None, None, 256000)      │     524,544,000 │ gemma_backbone[0][0]       │
│ (ReversibleEmbedding)         │                           │                 │                            │
└───────────────────────────────┴───────────────────────────┴─────────────────┴────────────────────────────┘
 Total params: 2,508,502,016 (2.34 GB)
 Trainable params: 1,363,968 (2.60 MB)
 Non-trainable params: 2,507,138,048 (2.34 GB)
```

{{% /details %}}

Let's fine-tune the QLoRA model.

If you are using a device with int8 acceleration support, you should see an improvement in the training speed.

```python
optimizer = keras.optimizers.SGD(learning_rate=1e-4)
gemma_lm.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=optimizer,
    weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
)
gemma_lm.fit(train_ds, epochs=1)
```

You should get a similar output with QLoRA fine-tuning.

```python
prompt = template.format(inputs="Bonjour, je m'appelle Morgane.")
outputs = gemma_lm.generate(prompt, max_length=256)
print("Translation:\n", outputs.replace(prompt, ""))
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Translation:
 Hello, my name is Morgane.
```

{{% /details %}}

And we're all done!

Note that for demonstration purposes, this example fine-tunes the model on a small subset of the dataset for just one epoch and with a low LoRA rank value. To get better responses from the fine-tuned model, you can experiment with:

- Increasing the size of the fine-tuning dataset.
- Training for more steps (epochs).
- Setting a higher LoRA rank.
- Modifying the hyperparameter values such as `learning_rate` and `weight_decay`.
