---
title: Sentence embeddings using Siamese RoBERTa-networks
toc: true
weight: 20
type: docs
---

{{< keras/original checkedAt="2024-11-21" >}}

**Author:** [Mohammed Abu El-Nasr](https://github.com/abuelnasr0)  
**Date created:** 2023/07/14  
**Last modified:** 2023/07/14  
**Description:** Fine-tune a RoBERTa model to generate sentence embeddings using KerasHub.

{{< keras/version v=3 >}}

{{< cards cols="2" >}}
{{< card link="https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/nlp/ipynb/sentence_embeddings_with_sbert.ipynb" title="Colab" tag="Colab" tagType="warning">}}
{{< card link="https://github.com/keras-team/keras-io/blob/master/examples/nlp/sentence_embeddings_with_sbert.py" title="GitHub" tag="GitHub">}}
{{< /cards >}}

## Introduction

BERT and RoBERTa can be used for semantic textual similarity tasks, where two sentences are passed to the model and the network predicts whether they are similar or not. But what if we have a large collection of sentences and want to find the most similar pairs in that collection? That will take n\*(n-1)/2 inference computations, where n is the number of sentences in the collection. For example, if n = 10000, the required time will be 65 hours on a V100 GPU.

A common method to overcome the time overhead issue is to pass one sentence to the model, then average the output of the model, or take the first token (the \[CLS\] token) and use them as a [sentence embedding](https://en.wikipedia.org/wiki/Sentence_embedding), then use a vector similarity measure like cosine similarity or Manhatten / Euclidean distance to find close sentences (semantically similar sentences). That will reduce the time to find the most similar pairs in a collection of 10,000 sentences from 65 hours to 5 seconds!

If we use RoBERTa directly, that will yield rather bad sentence embeddings. But if we fine-tune RoBERTa using a Siamese network, that will generate semantically meaningful sentence embeddings. This will enable RoBERTa to be used for new tasks. These tasks include:

- Large-scale semantic similarity comparison.
- Clustering.
- Information retrieval via semantic search.

In this example, we will show how to fine-tune a RoBERTa model using a Siamese network such that it will be able to produce semantically meaningful sentence embeddings and use them in a semantic search and clustering example. This method of fine-tuning was introduced in [Sentence-BERT](https://arxiv.org/abs/1908.10084)

## Setup

Let's install and import the libraries we need. We'll be using the KerasHub library in this example.

We will also enable [mixed precision](https://www.tensorflow.org/guide/mixed_precision) training. This will help us reduce the training time.

```python
!pip install -q --upgrade keras-hub
!pip install -q --upgrade keras  # Upgrade to Keras 3.
```

```python
import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import keras_hub
import tensorflow as tf
import tensorflow_datasets as tfds
import sklearn.cluster as cluster

keras.mixed_precision.set_global_policy("mixed_float16")
```

## Fine-tune the model using siamese networks

[Siamese network](https://en.wikipedia.org/wiki/Siamese_neural_network) is a neural network architecture that contains two or more subnetworks. The subnetworks share the same weights. It is used to generate feature vectors for each input and then compare them for similarity.

For our example, the subnetwork will be a RoBERTa model that has a pooling layer on top of it to produce the embeddings of the input sentences. These embeddings will then be compared to each other to learn to produce semantically meaningful embeddings.

The pooling strategies used are mean, max, and CLS pooling. Mean pooling produces the best results. We will use it in our examples.

### Fine-tune using the regression objective function

For building the siamese network with the regression objective function, the siamese network is asked to predict the cosine similarity between the embeddings of the two input sentences.

Cosine similarity indicates the angle between the sentence embeddings. If the cosine similarity is high, that means there is a small angle between the embeddings; hence, they are semantically similar.

#### Load the dataset

We will use the STSB dataset to fine-tune the model for the regression objective. STSB consists of a collection of sentence pairs that are labelled in the range \[0, 5\]. 0 indicates the least semantic similarity between the two sentences, and 5 indicates the most semantic similarity between the two sentences.

The range of the cosine similarity is \[-1, 1\] and it's the output of the siamese network, but the range of the labels in the dataset is \[0, 5\]. We need to unify the range between the cosine similarity and the dataset labels, so while preparing the dataset, we will divide the labels by 2.5 and subtract 1.

```python
TRAIN_BATCH_SIZE = 6
VALIDATION_BATCH_SIZE = 8

TRAIN_NUM_BATCHES = 300
VALIDATION_NUM_BATCHES = 40

AUTOTUNE = tf.data.experimental.AUTOTUNE


def change_range(x):
    return (x / 2.5) - 1


def prepare_dataset(dataset, num_batches, batch_size):
    dataset = dataset.map(
        lambda z: (
            [z["sentence1"], z["sentence2"]],
            [tf.cast(change_range(z["label"]), tf.float32)],
        ),
        num_parallel_calls=AUTOTUNE,
    )
    dataset = dataset.batch(batch_size)
    dataset = dataset.take(num_batches)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset


stsb_ds = tfds.load(
    "glue/stsb",
)
stsb_train, stsb_valid = stsb_ds["train"], stsb_ds["validation"]

stsb_train = prepare_dataset(stsb_train, TRAIN_NUM_BATCHES, TRAIN_BATCH_SIZE)
stsb_valid = prepare_dataset(stsb_valid, VALIDATION_NUM_BATCHES, VALIDATION_BATCH_SIZE)
```

Let's see examples from the dataset of two sentenses and their similarity.

```python
for x, y in stsb_train:
    for i, example in enumerate(x):
        print(f"sentence 1 : {example[0]} ")
        print(f"sentence 2 : {example[1]} ")
        print(f"similarity : {y[i]} \n")
    break
```

{{% details title="Result" closed="true" %}}

```plain
sentence 1 : b"A young girl is sitting on Santa's lap."
sentence 2 : b"A little girl is sitting on Santa's lap"
similarity : [0.9200001]
```

```plain
sentence 1 : b'A women sitting at a table drinking with a basketball picture in the background.'
sentence 2 : b'A woman in a sari drinks something while sitting at a table.'
similarity : [0.03999996]
```

```plain
sentence 1 : b'Norway marks anniversary of massacre'
sentence 2 : b"Norway Marks Anniversary of Breivik's Massacre"
similarity : [0.52]
```

```plain
sentence 1 : b'US drone kills six militants in Pakistan: officials'
sentence 2 : b'US missiles kill 15 in Pakistan: officials'
similarity : [-0.03999996]
```

```plain
sentence 1 : b'On Tuesday, the central bank left interest rates steady, as expected, but also declared that overall risks were weighted toward weakness and warned of deflation risks.'
sentence 2 : b"The central bank's policy board left rates steady for now, as widely expected, but surprised the market by declaring that overall risks were weighted toward weakness."
similarity : [0.6]
```

```plain
sentence 1 : b'At one of the three sampling sites at Huntington Beach, the bacteria reading came back at 160 on June 16 and at 120 on June 23.'
sentence 2 : b'The readings came back at 160 on June 16 and 120 at June 23 at one of three sampling sites at Huntington Beach.'
similarity : [0.29999995]
```

{{% /details %}}

#### Build the encoder model.

Now, we'll build the encoder model that will produce the sentence embeddings. It consists of:

- A preprocessor layer to tokenize and generate padding masks for the sentences.
- A backbone model that will generate the contextual representation of each token in the sentence.
- A mean pooling layer to produce the embeddings. We will use [`keras.layers.GlobalAveragePooling1D`]({{< relref "/docs/api/layers/pooling_layers/global_average_pooling1d#globalaveragepooling1d-class" >}}) to apply the mean pooling to the backbone outputs. We will pass the padding mask to the layer to exclude padded tokens from being averaged.
- A normalization layer to normalize the embeddings as we are using the cosine similarity.

```python
preprocessor = keras_hub.models.RobertaPreprocessor.from_preset("roberta_base_en")
backbone = keras_hub.models.RobertaBackbone.from_preset("roberta_base_en")
inputs = keras.Input(shape=(1,), dtype="string", name="sentence")
x = preprocessor(inputs)
h = backbone(x)
embedding = keras.layers.GlobalAveragePooling1D(name="pooling_layer")(
    h, x["padding_mask"]
)
n_embedding = keras.layers.UnitNormalization(axis=1)(embedding)
roberta_normal_encoder = keras.Model(inputs=inputs, outputs=n_embedding)

roberta_normal_encoder.summary()
```

{{% details title="Result" closed="true" %}}

```plain
Model: "functional_1"
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
┃ Layer (type)        ┃ Output Shape      ┃ Param # ┃ Connected to         ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
│ sentence            │ (None, 1)         │       0 │ -                    │
│ (InputLayer)        │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ roberta_preprocess… │ [(None, 512),     │       0 │ sentence[0][0]       │
│ (RobertaPreprocess… │ (None, 512)]      │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ roberta_backbone    │ (None, 512, 768)  │ 124,05… │ roberta_preprocesso… │
│ (RobertaBackbone)   │                   │         │ roberta_preprocesso… │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ pooling_layer       │ (None, 768)       │       0 │ roberta_backbone[0]… │
│ (GlobalAveragePool… │                   │         │ roberta_preprocesso… │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ unit_normalization  │ (None, 768)       │       0 │ pooling_layer[0][0]  │
│ (UnitNormalization) │                   │         │                      │
└─────────────────────┴───────────────────┴─────────┴──────────────────────┘
 Total params: 124,052,736 (473.22 MB)
 Trainable params: 124,052,736 (473.22 MB)
 Non-trainable params: 0 (0.00 B)
```

{{% /details %}}

#### Build the Siamese network with the regression objective function.

It's described above that the Siamese network has two or more subnetworks, and for this Siamese model, we need two encoders. But we don't have two encoders; we have only one encoder, but we will pass the two sentences through it. That way, we can have two paths to get the embeddings and also shared weights between the two paths.

After passing the two sentences to the model and getting the normalized embeddings, we will multiply the two normalized embeddings to get the cosine similarity between the two sentences.

```python
class RegressionSiamese(keras.Model):
    def __init__(self, encoder, **kwargs):
        inputs = keras.Input(shape=(2,), dtype="string", name="sentences")
        sen1, sen2 = keras.ops.split(inputs, 2, axis=1)
        u = encoder(sen1)
        v = encoder(sen2)
        cosine_similarity_scores = keras.ops.matmul(u, keras.ops.transpose(v))

        super().__init__(
            inputs=inputs,
            outputs=cosine_similarity_scores,
            **kwargs,
        )

        self.encoder = encoder

    def get_encoder(self):
        return self.encoder
```

#### Fit the model

Let's try this example before training and compare it to the output after training.

```python
sentences = [
    "Today is a very sunny day.",
    "I am hungry, I will get my meal.",
    "The dog is eating his food.",
]
query = ["The dog is enjoying his meal."]

encoder = roberta_normal_encoder

sentence_embeddings = encoder(tf.constant(sentences))
query_embedding = encoder(tf.constant(query))

cosine_similarity_scores = tf.matmul(query_embedding, tf.transpose(sentence_embeddings))
for i, sim in enumerate(cosine_similarity_scores[0]):
    print(f"cosine similarity score between sentence {i+1} and the query = {sim} ")
```

{{% details title="Result" closed="true" %}}

```plain
cosine similarity score between sentence 1 and the query = 0.96630859375
cosine similarity score between sentence 2 and the query = 0.97607421875
cosine similarity score between sentence 3 and the query = 0.99365234375
```

{{% /details %}}

For the training we will use `MeanSquaredError()` as loss function, and `Adam()` optimizer with learning rate = 2e-5.

```python
roberta_regression_siamese = RegressionSiamese(roberta_normal_encoder)

roberta_regression_siamese.compile(
    loss=keras.losses.MeanSquaredError(),
    optimizer=keras.optimizers.Adam(2e-5),
    jit_compile=False,
)

roberta_regression_siamese.fit(stsb_train, validation_data=stsb_valid, epochs=1)
```

{{% details title="Result" closed="true" %}}

```plain
 300/300 ━━━━━━━━━━━━━━━━━━━━ 115s 297ms/step - loss: 0.4751 - val_loss: 0.4025

<keras.src.callbacks.history.History at 0x7f5a78392140>
```

{{% /details %}}

Let's try the model after training, we will notice a huge difference in the output. That means that the model after fine-tuning is capable of producing semantically meaningful embeddings. where the semantically similar sentences have a small angle between them. and semantically dissimilar sentences have a large angle between them.

```python
sentences = [
    "Today is a very sunny day.",
    "I am hungry, I will get my meal.",
    "The dog is eating his food.",
]
query = ["The dog is enjoying his food."]

encoder = roberta_regression_siamese.get_encoder()

sentence_embeddings = encoder(tf.constant(sentences))
query_embedding = encoder(tf.constant(query))

cosine_simalarities = tf.matmul(query_embedding, tf.transpose(sentence_embeddings))
for i, sim in enumerate(cosine_simalarities[0]):
    print(f"cosine similarity between sentence {i+1} and the query = {sim} ")
```

{{% details title="Result" closed="true" %}}

```plain
cosine similarity between sentence 1 and the query = 0.10986328125
cosine similarity between sentence 2 and the query = 0.53466796875
cosine similarity between sentence 3 and the query = 0.83544921875
```

{{% /details %}}

### Fine-tune Using the triplet Objective Function

For the Siamese network with the triplet objective function, three sentences are passed to the Siamese network _anchor_, _positive_, and _negative_ sentences. _anchor_ and _positive_ sentences are semantically similar, and _anchor_ and _negative_ sentences are semantically dissimilar. The objective is to minimize the distance between the _anchor_ sentence and the _positive_ sentence, and to maximize the distance between the _anchor_ sentence and the _negative_ sentence.

#### Load the dataset

We will use the Wikipedia-sections-triplets dataset for fine-tuning. This data set consists of sentences derived from the Wikipedia website. It has a collection of 3 sentences _anchor_, _positive_, _negative_. _anchor_ and _positive_ are derived from the same section. _anchor_ and _negative_ are derived from different sections.

This dataset has 1.8 million training triplets and 220,000 test triplets. In this example, we will only use 1200 triplets for training and 300 for testing.

```python
!wget https://sbert.net/datasets/wikipedia-sections-triplets.zip -q
!unzip wikipedia-sections-triplets.zip  -d  wikipedia-sections-triplets
```

```python
NUM_TRAIN_BATCHES = 200
NUM_TEST_BATCHES = 75
AUTOTUNE = tf.data.experimental.AUTOTUNE


def prepare_wiki_data(dataset, num_batches):
    dataset = dataset.map(
        lambda z: ((z["Sentence1"], z["Sentence2"], z["Sentence3"]), 0)
    )
    dataset = dataset.batch(6)
    dataset = dataset.take(num_batches)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset


wiki_train = tf.data.experimental.make_csv_dataset(
    "wikipedia-sections-triplets/train.csv",
    batch_size=1,
    num_epochs=1,
)
wiki_test = tf.data.experimental.make_csv_dataset(
    "wikipedia-sections-triplets/test.csv",
    batch_size=1,
    num_epochs=1,
)

wiki_train = prepare_wiki_data(wiki_train, NUM_TRAIN_BATCHES)
wiki_test = prepare_wiki_data(wiki_test, NUM_TEST_BATCHES)
```

{{% details title="Result" closed="true" %}}

```plain
Archive:  wikipedia-sections-triplets.zip
  inflating: wikipedia-sections-triplets/validation.csv
  inflating: wikipedia-sections-triplets/Readme.txt
  inflating: wikipedia-sections-triplets/test.csv
  inflating: wikipedia-sections-triplets/train.csv
```

{{% /details %}}

#### Build the encoder model

For this encoder model, we will use RoBERTa with mean pooling and we will not normalize the output embeddings. The encoder model consists of:

- A preprocessor layer to tokenize and generate padding masks for the sentences.
- A backbone model that will generate the contextual representation of each token in the sentence.
- A mean pooling layer to produce the embeddings.

```python
preprocessor = keras_hub.models.RobertaPreprocessor.from_preset("roberta_base_en")
backbone = keras_hub.models.RobertaBackbone.from_preset("roberta_base_en")
input = keras.Input(shape=(1,), dtype="string", name="sentence")

x = preprocessor(input)
h = backbone(x)
embedding = keras.layers.GlobalAveragePooling1D(name="pooling_layer")(
    h, x["padding_mask"]
)

roberta_encoder = keras.Model(inputs=input, outputs=embedding)


roberta_encoder.summary()
```

{{% details title="Result" closed="true" %}}

```plain
Model: "functional_3"
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
┃ Layer (type)        ┃ Output Shape      ┃ Param # ┃ Connected to         ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
│ sentence            │ (None, 1)         │       0 │ -                    │
│ (InputLayer)        │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ roberta_preprocess… │ [(None, 512),     │       0 │ sentence[0][0]       │
│ (RobertaPreprocess… │ (None, 512)]      │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ roberta_backbone_1  │ (None, 512, 768)  │ 124,05… │ roberta_preprocesso… │
│ (RobertaBackbone)   │                   │         │ roberta_preprocesso… │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ pooling_layer       │ (None, 768)       │       0 │ roberta_backbone_1[… │
│ (GlobalAveragePool… │                   │         │ roberta_preprocesso… │
└─────────────────────┴───────────────────┴─────────┴──────────────────────┘
 Total params: 124,052,736 (473.22 MB)
 Trainable params: 124,052,736 (473.22 MB)
 Non-trainable params: 0 (0.00 B)
```

{{% /details %}}

#### Build the Siamese network with the triplet objective function

For the Siamese network with the triplet objective function, we will build the model with an encoder, and we will pass the three sentences through that encoder. We will get an embedding for each sentence, and we will calculate the `positive_dist` and `negative_dist` that will be passed to the loss function described below.

```python
class TripletSiamese(keras.Model):
    def __init__(self, encoder, **kwargs):
        anchor = keras.Input(shape=(1,), dtype="string")
        positive = keras.Input(shape=(1,), dtype="string")
        negative = keras.Input(shape=(1,), dtype="string")

        ea = encoder(anchor)
        ep = encoder(positive)
        en = encoder(negative)

        positive_dist = keras.ops.sum(keras.ops.square(ea - ep), axis=1)
        negative_dist = keras.ops.sum(keras.ops.square(ea - en), axis=1)

        positive_dist = keras.ops.sqrt(positive_dist)
        negative_dist = keras.ops.sqrt(negative_dist)

        output = keras.ops.stack([positive_dist, negative_dist], axis=0)

        super().__init__(inputs=[anchor, positive, negative], outputs=output, **kwargs)

        self.encoder = encoder

    def get_encoder(self):
        return self.encoder
```

We will use a custom loss function for the triplet objective. The loss function will receive the distance between the _anchor_ and the _positive_ embeddings `positive_dist`, and the distance between the _anchor_ and the _negative_ embeddings `negative_dist`, where they are stacked together in `y_pred`.

We will use `positive_dist` and `negative_dist` to compute the loss such that `negative_dist` is larger than `positive_dist` at least by a specific margin. Mathematically, we will minimize this loss function: `max( positive_dist - negative_dist + margin, 0)`.

There is no `y_true` used in this loss function. Note that we set the labels in the dataset to zero, but they will not be used.

```python
class TripletLoss(keras.losses.Loss):
    def __init__(self, margin=1, **kwargs):
        super().__init__(**kwargs)
        self.margin = margin

    def call(self, y_true, y_pred):
        positive_dist, negative_dist = tf.unstack(y_pred, axis=0)

        losses = keras.ops.relu(positive_dist - negative_dist + self.margin)
        return keras.ops.mean(losses, axis=0)
```

#### Fit the model

For the training, we will use the custom `TripletLoss()` loss function, and `Adam()` optimizer with a learning rate = 2e-5.

```python
roberta_triplet_siamese = TripletSiamese(roberta_encoder)

roberta_triplet_siamese.compile(
    loss=TripletLoss(),
    optimizer=keras.optimizers.Adam(2e-5),
    jit_compile=False,
)

roberta_triplet_siamese.fit(wiki_train, validation_data=wiki_test, epochs=1)
```

{{% details title="Result" closed="true" %}}

```plain
 200/200 ━━━━━━━━━━━━━━━━━━━━ 128s 467ms/step - loss: 0.7822 - val_loss: 0.7126

<keras.src.callbacks.history.History at 0x7f5c3636c580>
```

{{% /details %}}

Let's try this model in a clustering example. Here are 6 questions. first 3 questions about learning English, and the last 3 questions about working online. Let's see if the embeddings produced by our encoder will cluster them correctly.

```python
questions = [
    "What should I do to improve my English writting?",
    "How to be good at speaking English?",
    "How can I improve my English?",
    "How to earn money online?",
    "How do I earn money online?",
    "How to work and earn money through internet?",
]

encoder = roberta_triplet_siamese.get_encoder()
embeddings = encoder(tf.constant(questions))
kmeans = cluster.KMeans(n_clusters=2, random_state=0, n_init="auto").fit(embeddings)

for i, label in enumerate(kmeans.labels_):
    print(f"sentence ({questions[i]}) belongs to cluster {label}")
```

{{% details title="Result" closed="true" %}}

```plain
sentence (What should I do to improve my English writting?) belongs to cluster 1
sentence (How to be good at speaking English?) belongs to cluster 1
sentence (How can I improve my English?) belongs to cluster 1
sentence (How to earn money online?) belongs to cluster 0
sentence (How do I earn money online?) belongs to cluster 0
sentence (How to work and earn money through internet?) belongs to cluster 0
```

{{% /details %}}