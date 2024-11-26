---
title: Abstractive Text Summarization with BART
toc: true
weight: 22
type: docs
---

{{< keras/original checkedAt="2024-11-21" >}}

**Author:** [Abheesht Sharma](https://github.com/abheesht17/)  
**Date created:** 2023/07/08  
**Last modified:** 2024/03/20  
**Description:** Use KerasHub to fine-tune BART on the abstractive summarization task.

{{< keras/version v=3 >}}

{{< cards cols="2" >}}
{{< card link="https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/nlp/ipynb/abstractive_summarization_with_bart.ipynb" title="Colab" tag="Colab" tagType="warning">}}
{{< card link="https://github.com/keras-team/keras-io/blob/master/examples/nlp/abstractive_summarization_with_bart.py" title="GitHub" tag="GitHub">}}
{{< /cards >}}

## Introduction

In the era of information overload, it has become crucial to extract the crux of a long document or a conversation and express it in a few sentences. Owing to the fact that summarization has widespread applications in different domains, it has become a key, well-studied NLP task in recent years.

[Bidirectional Autoregressive Transformer (BART)](https://arxiv.org/abs/1910.13461) is a Transformer-based encoder-decoder model, often used for sequence-to-sequence tasks like summarization and neural machine translation. BART is pre-trained in a self-supervised fashion on a large text corpus. During pre-training, the text is corrupted and BART is trained to reconstruct the original text (hence called a "denoising autoencoder"). Some pre-training tasks include token masking, token deletion, sentence permutation (shuffle sentences and train BART to fix the order), etc.

In this example, we will demonstrate how to fine-tune BART on the abstractive summarization task (on conversations!) using KerasHub, and generate summaries using the fine-tuned model.

## Setup

Before we start implementing the pipeline, let's install and import all the libraries we need. We'll be using the KerasHub library. We will also need a couple of utility libraries.

```python
!pip install git+https://github.com/keras-team/keras-hub.git py7zr -q
```

{{% details title="Result" closed="true" %}}

```plain
  Installing build dependencies ... [?25l[?25hdone
  Getting requirements to build wheel ... [?25l[?25hdone
  Preparing metadata (pyproject.toml) ... [?25l[?25hdone
[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 66.4/66.4 kB [31m1.4 MB/s eta [36m0:00:00
[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.1/2.1 MB [31m34.8 MB/s eta [36m0:00:00
[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 412.3/412.3 kB [31m30.4 MB/s eta [36m0:00:00
[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 138.8/138.8 kB [31m15.1 MB/s eta [36m0:00:00
[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 49.8/49.8 kB [31m5.8 MB/s eta [36m0:00:00
[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.7/2.7 MB [31m61.4 MB/s eta [36m0:00:00
[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 93.1/93.1 kB [31m10.1 MB/s eta [36m0:00:00
[?25h  Building wheel for keras-hub (pyproject.toml) ... [?25l[?25hdone
```

{{% /details %}}

This examples uses [Keras 3]({{< relref "/docs/keras_3" >}}) to work in any of `"tensorflow"`, `"jax"` or `"torch"`. Support for Keras 3 is baked into KerasHub, simply change the `"KERAS_BACKEND"` environment variable to select the backend of your choice. We select the JAX backend below.

```python
import os

os.environ["KERAS_BACKEND"] = "jax"
```

Import all necessary libraries.

```python
import py7zr
import time

import keras_hub
import keras
import tensorflow as tf
import tensorflow_datasets as tfds
```

{{% details title="Result" closed="true" %}}

```plain
Using JAX backend.
```

{{% /details %}}

Let's also define our hyperparameters.

```python
BATCH_SIZE = 8
NUM_BATCHES = 600
EPOCHS = 1  # Can be set to a higher value for better results
MAX_ENCODER_SEQUENCE_LENGTH = 512
MAX_DECODER_SEQUENCE_LENGTH = 128
MAX_GENERATION_LENGTH = 40
```

## Dataset

Let's load the [SAMSum dataset](https://arxiv.org/abs/1911.12237). This dataset contains around 15,000 pairs of conversations/dialogues and summaries.

```python
# Download the dataset.
filename = keras.utils.get_file(
    "corpus.7z",
    origin="https://huggingface.co/datasets/samsum/resolve/main/data/corpus.7z",
)

# Extract the `.7z` file.
with py7zr.SevenZipFile(filename, mode="r") as z:
    z.extractall(path="/root/tensorflow_datasets/downloads/manual")

# Load data using TFDS.
samsum_ds = tfds.load("samsum", split="train", as_supervised=True)
```

{{% details title="Result" closed="true" %}}

```plain
Downloading data from https://huggingface.co/datasets/samsum/resolve/main/data/corpus.7z
 2944100/2944100 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step
Downloading and preparing dataset Unknown size (download: Unknown size, generated: 10.71 MiB, total: 10.71 MiB) to /root/tensorflow_datasets/samsum/1.0.0...

Generating splits...:   0%|          | 0/3 [00:00<?, ? splits/s]

Generating train examples...:   0%|          | 0/14732 [00:00<?, ? examples/s]

Shuffling /root/tensorflow_datasets/samsum/1.0.0.incompleteYA9MAV/samsum-train.tfrecord*...:   0%|          | …

Generating validation examples...:   0%|          | 0/818 [00:00<?, ? examples/s]

Shuffling /root/tensorflow_datasets/samsum/1.0.0.incompleteYA9MAV/samsum-validation.tfrecord*...:   0%|       …

Generating test examples...:   0%|          | 0/819 [00:00<?, ? examples/s]

Shuffling /root/tensorflow_datasets/samsum/1.0.0.incompleteYA9MAV/samsum-test.tfrecord*...:   0%|          | 0…

Dataset samsum downloaded and prepared to /root/tensorflow_datasets/samsum/1.0.0. Subsequent calls will reuse this data.
```

{{% /details %}}

The dataset has two fields: `dialogue` and `summary`. Let's see a sample.

```python
for dialogue, summary in samsum_ds:
    print(dialogue.numpy())
    print(summary.numpy())
    break
```

{{% details title="Result" closed="true" %}}

```plain
b"Carter: Hey Alexis, I just wanted to let you know that I had a really nice time with you tonight. \r\nAlexis: Thanks Carter. Yeah, I really enjoyed myself as well. \r\nCarter: If you are up for it, I would really like to see you again soon.\r\nAlexis: Thanks Carter, I'm flattered. But I have a really busy week coming up.\r\nCarter: Yeah, no worries. I totally understand. But if you ever want to go grab dinner again, just let me know. \r\nAlexis: Yeah of course. Thanks again for tonight. \r\nCarter: Sure. Have a great night. "
b'Alexis and Carter met tonight. Carter would like to meet again, but Alexis is busy.'
```

{{% /details %}}

We'll now batch the dataset and retain only a subset of the dataset for the purpose of this example. The dialogue is fed to the encoder, and the corresponding summary serves as input to the decoder. We will, therefore, change the format of the dataset to a dictionary having two keys: `"encoder_text"` and `"decoder_text"`.This is how [`keras_hub.models.BartSeq2SeqLMPreprocessor`]({{< relref "/docs/api/keras_hub/models/bart/bart_seq_2_seq_lm_preprocessor#bartseq2seqlmpreprocessor-class" >}}) expects the input format to be.

```python
train_ds = (
    samsum_ds.map(
        lambda dialogue, summary: {"encoder_text": dialogue, "decoder_text": summary}
    )
    .batch(BATCH_SIZE)
    .cache()
)
train_ds = train_ds.take(NUM_BATCHES)
```

## Fine-tune BART

Let's load the model and preprocessor first. We use sequence lengths of 512 and 128 for the encoder and decoder, respectively, instead of 1024 (which is the default sequence length). This will allow us to run this example quickly on Colab.

If you observe carefully, the preprocessor is attached to the model. What this means is that we don't have to worry about preprocessing the text inputs; everything will be done internally. The preprocessor tokenizes the encoder text and the decoder text, adds special tokens and pads them. To generate labels for auto-regressive training, the preprocessor shifts the decoder text one position to the right. This is done because at every timestep, the model is trained to predict the next token.

```python
preprocessor = keras_hub.models.BartSeq2SeqLMPreprocessor.from_preset(
    "bart_base_en",
    encoder_sequence_length=MAX_ENCODER_SEQUENCE_LENGTH,
    decoder_sequence_length=MAX_DECODER_SEQUENCE_LENGTH,
)
bart_lm = keras_hub.models.BartSeq2SeqLM.from_preset(
    "bart_base_en", preprocessor=preprocessor
)

bart_lm.summary()
```

{{% details title="Result" closed="true" %}}

```plain
Downloading data from https://storage.googleapis.com/keras-hub/models/bart_base_en/v1/vocab.json
 898823/898823 ━━━━━━━━━━━━━━━━━━━━ 1s 1us/step
Downloading data from https://storage.googleapis.com/keras-hub/models/bart_base_en/v1/merges.txt
 456318/456318 ━━━━━━━━━━━━━━━━━━━━ 1s 1us/step
Downloading data from https://storage.googleapis.com/keras-hub/models/bart_base_en/v1/model.h5
 557969120/557969120 ━━━━━━━━━━━━━━━━━━━━ 29s 0us/step
```

```plain
Preprocessor: "bart_seq2_seq_lm_preprocessor"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Tokenizer (type)                                   ┃                                             Vocab # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ bart_tokenizer (BartTokenizer)                     │                                              50,265 │
└────────────────────────────────────────────────────┴─────────────────────────────────────────────────────┘
```

```plain
Model: "bart_seq2_seq_lm"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                  ┃ Output Shape              ┃     Param # ┃ Connected to                   ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ decoder_padding_mask          │ (None, None)              │           0 │ -                              │
│ (InputLayer)                  │                           │             │                                │
├───────────────────────────────┼───────────────────────────┼─────────────┼────────────────────────────────┤
│ decoder_token_ids             │ (None, None)              │           0 │ -                              │
│ (InputLayer)                  │                           │             │                                │
├───────────────────────────────┼───────────────────────────┼─────────────┼────────────────────────────────┤
│ encoder_padding_mask          │ (None, None)              │           0 │ -                              │
│ (InputLayer)                  │                           │             │                                │
├───────────────────────────────┼───────────────────────────┼─────────────┼────────────────────────────────┤
│ encoder_token_ids             │ (None, None)              │           0 │ -                              │
│ (InputLayer)                  │                           │             │                                │
├───────────────────────────────┼───────────────────────────┼─────────────┼────────────────────────────────┤
│ bart_backbone (BartBackbone)  │ [(None, None, 768),       │ 139,417,344 │ decoder_padding_mask[0][0],    │
│                               │ (None, None, 768)]        │             │ decoder_token_ids[0][0],       │
│                               │                           │             │ encoder_padding_mask[0][0],    │
│                               │                           │             │ encoder_token_ids[0][0]        │
├───────────────────────────────┼───────────────────────────┼─────────────┼────────────────────────────────┤
│ reverse_embedding             │ (None, 50265)             │  38,603,520 │ bart_backbone[0][0]            │
│ (ReverseEmbedding)            │                           │             │                                │
└───────────────────────────────┴───────────────────────────┴─────────────┴────────────────────────────────┘
 Total params: 139,417,344 (4.15 GB)
 Trainable params: 139,417,344 (4.15 GB)
 Non-trainable params: 0 (0.00 B)
```

{{% /details %}}

Define the optimizer and loss. We use the Adam optimizer with a linearly decaying learning rate. Compile the model.

```python
optimizer = keras.optimizers.AdamW(
    learning_rate=5e-5,
    weight_decay=0.01,
    epsilon=1e-6,
    global_clipnorm=1.0,  # Gradient clipping.
)
# Exclude layernorm and bias terms from weight decay.
optimizer.exclude_from_weight_decay(var_names=["bias"])
optimizer.exclude_from_weight_decay(var_names=["gamma"])
optimizer.exclude_from_weight_decay(var_names=["beta"])

loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

bart_lm.compile(
    optimizer=optimizer,
    loss=loss,
    weighted_metrics=["accuracy"],
)
```

Let's train the model!

```python
bart_lm.fit(train_ds, epochs=EPOCHS)
```

{{% details title="Result" closed="true" %}}

```plain
 600/600 ━━━━━━━━━━━━━━━━━━━━ 398s 586ms/step - loss: 0.4330

<keras_core.src.callbacks.history.History at 0x7ae2faf3e110>
```

{{% /details %}}

## Generate summaries and evaluate them!

Now that the model has been trained, let's get to the fun part - actually generating summaries! Let's pick the first 100 samples from the validation set and generate summaries for them. We will use the default decoding strategy, i.e., greedy search.

Generation in KerasHub is highly optimized. It is backed by the power of XLA. Secondly, key/value tensors in the self-attention layer and cross-attention layer in the decoder are cached to avoid recomputation at every timestep.

```python
def generate_text(model, input_text, max_length=200, print_time_taken=False):
    start = time.time()
    output = model.generate(input_text, max_length=max_length)
    end = time.time()
    print(f"Total Time Elapsed: {end - start:.2f}s")
    return output


# Load the dataset.
val_ds = tfds.load("samsum", split="validation", as_supervised=True)
val_ds = val_ds.take(100)

dialogues = []
ground_truth_summaries = []
for dialogue, summary in val_ds:
    dialogues.append(dialogue.numpy())
    ground_truth_summaries.append(summary.numpy())

# Let's make a dummy call - the first call to XLA generally takes a bit longer.
_ = generate_text(bart_lm, "sample text", max_length=MAX_GENERATION_LENGTH)

# Generate summaries.
generated_summaries = generate_text(
    bart_lm,
    val_ds.map(lambda dialogue, _: dialogue).batch(8),
    max_length=MAX_GENERATION_LENGTH,
    print_time_taken=True,
)
```

{{% details title="Result" closed="true" %}}

```plain
Total Time Elapsed: 21.22s
Total Time Elapsed: 49.00s
```

{{% /details %}}

Let's see some of the summaries.

```python
for dialogue, generated_summary, ground_truth_summary in zip(
    dialogues[:5], generated_summaries[:5], ground_truth_summaries[:5]
):
    print("Dialogue:", dialogue)
    print("Generated Summary:", generated_summary)
    print("Ground Truth Summary:", ground_truth_summary)
    print("=============================")
```

{{% details title="Result" closed="true" %}}

```plain
Dialogue: b'Tony: Is the boss in?\r\nClaire: Not yet.\r\nTony: Could let me know when he comes, please? \r\nClaire: Of course.\r\nTony: Thank you.'
Generated Summary: Tony will let Claire know when her boss comes.
Ground Truth Summary: b"The boss isn't in yet. Claire will let Tony know when he comes."
=============================
Dialogue: b"James: What shouldl I get her?\r\nTim: who?\r\nJames: gees Mary my girlfirend\r\nTim: Am I really the person you should be asking?\r\nJames: oh come on it's her birthday on Sat\r\nTim: ask Sandy\r\nTim: I honestly am not the right person to ask this\r\nJames: ugh fine!"
Generated Summary: Mary's girlfriend is birthday. James and Tim are going to ask Sandy to buy her.
Ground Truth Summary: b"Mary's birthday is on Saturday. Her boyfriend, James, is looking for gift ideas. Tim suggests that he ask Sandy."
=============================
Dialogue: b"Mary: So, how's Israel? Have you been on the beach?\r\nKate: It's so expensive! But they say, it's Tel Aviv... Tomorrow we are going to Jerusalem.\r\nMary: I've heard Israel is expensive, Monica was there on vacation last year, she complained about how pricey it is. Are you going to the Dead Sea before it dies? ahahahha\r\nKate: ahahhaha yup, in few days."
Generated Summary: Kate is on vacation in Tel Aviv. Mary will visit the Dead Sea in a few days.
Ground Truth Summary: b'Mary and Kate discuss how expensive Israel is. Kate is in Tel Aviv now, planning to travel to Jerusalem tomorrow, and to the Dead Sea few days later.'
=============================
Dialogue: b"Giny: do we have rice?\r\nRiley: nope, it's finished\r\nGiny: fuck!\r\nGiny: ok, I'll buy"
Generated Summary: Giny wants to buy rice from Riley.
Ground Truth Summary: b"Giny and Riley don't have any rice left. Giny will buy some."
=============================
Dialogue: b"Jude: i'll be in warsaw at the beginning of december so we could meet again\r\nLeon: !!!\r\nLeon: at the beginning means...?\r\nLeon: cuz I won't be here during the first weekend\r\nJude: 10\r\nJude: but i think it's a monday, so never mind i guess :D\r\nLeon: yeah monday doesn't really work for me :D\r\nLeon: :<\r\nJude: oh well next time :d\r\nLeon: yeah...!"
Generated Summary: Jude and Leon will meet again this weekend at 10 am.
Ground Truth Summary: b'Jude is coming to Warsaw on the 10th of December and wants to see Leon. Leon has no time.'
=============================
```

{{% /details %}}

The generated summaries look awesome! Not bad for a model trained only for 1 epoch and on 5000 examples :)
