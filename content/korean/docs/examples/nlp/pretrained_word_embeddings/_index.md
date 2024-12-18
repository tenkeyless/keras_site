---
title: 사전 트레이닝된 단어 임베딩 사용
linkTitle: 사전 트레이닝된 단어 임베딩 사용
toc: true
weight: 8
type: docs
---

{{< keras/original checkedAt="2024-11-21" >}}

**{{< t f_author >}}** [fchollet](https://twitter.com/fchollet)  
**{{< t f_date_created >}}** 2020/05/05  
**{{< t f_last_modified >}}** 2020/05/05  
**{{< t f_description >}}** Text classification on the Newsgroup20 dataset using pre-trained GloVe word embeddings.

{{< keras/version v=3 >}}

{{< cards cols="2" >}}
{{< card link="https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/nlp/ipynb/pretrained_word_embeddings.ipynb" title="Colab" tag="Colab" tagType="warning">}}
{{< card link="https://github.com/keras-team/keras-io/blob/master/examples/nlp/pretrained_word_embeddings.py" title="GitHub" tag="GitHub">}}
{{< /cards >}}

## Setup {#setup}

```python
import os

# Only the TensorFlow backend supports string inputs.
os.environ["KERAS_BACKEND"] = "tensorflow"

import pathlib
import numpy as np
import tensorflow.data as tf_data
import keras
from keras import layers
```

## Introduction {#introduction}

In this example, we show how to train a text classification model that uses pre-trained word embeddings.

We'll work with the Newsgroup20 dataset, a set of 20,000 message board messages belonging to 20 different topic categories.

For the pre-trained word embeddings, we'll use [GloVe embeddings](http://nlp.stanford.edu/projects/glove/).

## Download the Newsgroup20 data {#download-the-newsgroup20-data}

```python
data_path = keras.utils.get_file(
    "news20.tar.gz",
    "http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.tar.gz",
    untar=True,
)
```

## Let's take a look at the data {#lets-take-a-look-at-the-data}

```python
data_dir = pathlib.Path(data_path).parent / "20_newsgroup"
dirnames = os.listdir(data_dir)
print("Number of directories:", len(dirnames))
print("Directory names:", dirnames)

fnames = os.listdir(data_dir / "comp.graphics")
print("Number of files in comp.graphics:", len(fnames))
print("Some example filenames:", fnames[:5])
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Number of directories: 20
Directory names: ['comp.sys.ibm.pc.hardware', 'comp.os.ms-windows.misc', 'comp.windows.x', 'sci.space', 'sci.crypt', 'sci.med', 'alt.atheism', 'rec.autos', 'rec.sport.hockey', 'talk.politics.misc', 'talk.politics.mideast', 'rec.motorcycles', 'talk.politics.guns', 'misc.forsale', 'sci.electronics', 'talk.religion.misc', 'comp.graphics', 'soc.religion.christian', 'comp.sys.mac.hardware', 'rec.sport.baseball']
Number of files in comp.graphics: 1000
Some example filenames: ['39638', '38747', '38242', '39057', '39031']
```

{{% /details %}}

Here's a example of what one file contains:

```python
print(open(data_dir / "comp.graphics" / "38987").read())
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Newsgroups: comp.graphics
Path: cantaloupe.srv.cs.cmu.edu!das-news.harvard.edu!noc.near.net!howland.reston.ans.net!agate!dog.ee.lbl.gov!network.ucsd.edu!usc!rpi!nason110.its.rpi.edu!mabusj
From: mabusj@nason110.its.rpi.edu (Jasen M. Mabus)
Subject: Looking for Brain in CAD
Message-ID: <c285m+p@rpi.edu>
Nntp-Posting-Host: nason110.its.rpi.edu
Reply-To: mabusj@rpi.edu
Organization: Rensselaer Polytechnic Institute, Troy, NY.
Date: Thu, 29 Apr 1993 23:27:20 GMT
Lines: 7
```

```plain
Jasen Mabus
RPI student
```

```plain
    I am looking for a hman brain in any CAD (.dxf,.cad,.iges,.cgm,etc.) or picture (.gif,.jpg,.ras,etc.) format for an animation demonstration. If any has or knows of a location please reply by e-mail to mabusj@rpi.edu.
```

```plain
Thank you in advance,
Jasen Mabus
```

{{% /details %}}

As you can see, there are header lines that are leaking the file's category, either explicitly (the first line is literally the category name), or implicitly, e.g. via the `Organization` filed. Let's get rid of the headers:

```python
samples = []
labels = []
class_names = []
class_index = 0
for dirname in sorted(os.listdir(data_dir)):
    class_names.append(dirname)
    dirpath = data_dir / dirname
    fnames = os.listdir(dirpath)
    print("Processing %s, %d files found" % (dirname, len(fnames)))
    for fname in fnames:
        fpath = dirpath / fname
        f = open(fpath, encoding="latin-1")
        content = f.read()
        lines = content.split("\n")
        lines = lines[10:]
        content = "\n".join(lines)
        samples.append(content)
        labels.append(class_index)
    class_index += 1

print("Classes:", class_names)
print("Number of samples:", len(samples))
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Processing alt.atheism, 1000 files found
Processing comp.graphics, 1000 files found
Processing comp.os.ms-windows.misc, 1000 files found
Processing comp.sys.ibm.pc.hardware, 1000 files found
Processing comp.sys.mac.hardware, 1000 files found
Processing comp.windows.x, 1000 files found
Processing misc.forsale, 1000 files found
Processing rec.autos, 1000 files found
Processing rec.motorcycles, 1000 files found
Processing rec.sport.baseball, 1000 files found
Processing rec.sport.hockey, 1000 files found
Processing sci.crypt, 1000 files found
Processing sci.electronics, 1000 files found
Processing sci.med, 1000 files found
Processing sci.space, 1000 files found
Processing soc.religion.christian, 997 files found
Processing talk.politics.guns, 1000 files found
Processing talk.politics.mideast, 1000 files found
Processing talk.politics.misc, 1000 files found
Processing talk.religion.misc, 1000 files found
Classes: ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']
Number of samples: 19997
```

{{% /details %}}

There's actually one category that doesn't have the expected number of files, but the difference is small enough that the problem remains a balanced classification problem.

## Shuffle and split the data into training & validation sets {#shuffle-and-split-the-data-into-training-validation-sets}

```python
# Shuffle the data
seed = 1337
rng = np.random.RandomState(seed)
rng.shuffle(samples)
rng = np.random.RandomState(seed)
rng.shuffle(labels)

# Extract a training & validation split
validation_split = 0.2
num_validation_samples = int(validation_split * len(samples))
train_samples = samples[:-num_validation_samples]
val_samples = samples[-num_validation_samples:]
train_labels = labels[:-num_validation_samples]
val_labels = labels[-num_validation_samples:]
```

## Create a vocabulary index {#create-a-vocabulary-index}

Let's use the `TextVectorization` to index the vocabulary found in the dataset. Later, we'll use the same layer instance to vectorize the samples.

Our layer will only consider the top 20,000 words, and will truncate or pad sequences to be actually 200 tokens long.

```python
vectorizer = layers.TextVectorization(max_tokens=20000, output_sequence_length=200)
text_ds = tf_data.Dataset.from_tensor_slices(train_samples).batch(128)
vectorizer.adapt(text_ds)
```

You can retrieve the computed vocabulary used via `vectorizer.get_vocabulary()`. Let's print the top 5 words:

```python
vectorizer.get_vocabulary()[:5]
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
['', '[UNK]', 'the', 'to', 'of']
```

{{% /details %}}

Let's vectorize a test sentence:

```python
output = vectorizer([["the cat sat on the mat"]])
output.numpy()[0, :6]
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
array([   2, 3480, 1818,   15,    2, 5830])
```

{{% /details %}}

As you can see, "the" gets represented as "2". Why not 0, given that "the" was the first word in the vocabulary? That's because index 0 is reserved for padding and index 1 is reserved for "out of vocabulary" tokens.

Here's a dict mapping words to their indices:

```python
voc = vectorizer.get_vocabulary()
word_index = dict(zip(voc, range(len(voc))))
```

As you can see, we obtain the same encoding as above for our test sentence:

```python
test = ["the", "cat", "sat", "on", "the", "mat"]
[word_index[w] for w in test]
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
[2, 3480, 1818, 15, 2, 5830]
```

{{% /details %}}

## Load pre-trained word embeddings {#load-pre-trained-word-embeddings}

Let's download pre-trained GloVe embeddings (a 822M zip file).

You'll need to run the following commands:

```python
!wget https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip
!unzip -q glove.6B.zip
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
--2023-11-19 22:45:27--  https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip
Resolving downloads.cs.stanford.edu (downloads.cs.stanford.edu)... 171.64.64.22
Connecting to downloads.cs.stanford.edu (downloads.cs.stanford.edu)|171.64.64.22|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 862182613 (822M) [application/zip]
Saving to: ‘glove.6B.zip’
glove.6B.zip        100%[===================>] 822.24M  5.05MB/s    in 2m 39s
2023-11-19 22:48:06 (5.19 MB/s) - ‘glove.6B.zip’ saved [862182613/862182613]
```

{{% /details %}}

The archive contains text-encoded vectors of various sizes: 50-dimensional, 100-dimensional, 200-dimensional, 300-dimensional. We'll use the 100D ones.

Let's make a dict mapping words (strings) to their NumPy vector representation:

```python
path_to_glove_file = "glove.6B.100d.txt"

embeddings_index = {}
with open(path_to_glove_file) as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index[word] = coefs

print("Found %s word vectors." % len(embeddings_index))
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Found 400000 word vectors.
```

{{% /details %}}

Now, let's prepare a corresponding embedding matrix that we can use in a Keras `Embedding` layer. It's a simple NumPy matrix where entry at index `i` is the pre-trained vector for the word of index `i` in our `vectorizer`'s vocabulary.

```python
num_tokens = len(voc) + 2
embedding_dim = 100
hits = 0
misses = 0

# Prepare embedding matrix
embedding_matrix = np.zeros((num_tokens, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # Words not found in embedding index will be all-zeros.
        # This includes the representation for "padding" and "OOV"
        embedding_matrix[i] = embedding_vector
        hits += 1
    else:
        misses += 1
print("Converted %d words (%d misses)" % (hits, misses))
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Converted 18021 words (1979 misses)
```

{{% /details %}}

Next, we load the pre-trained word embeddings matrix into an `Embedding` layer.

Note that we set `trainable=False` so as to keep the embeddings fixed (we don't want to update them during training).

```python
from keras.layers import Embedding

embedding_layer = Embedding(
    num_tokens,
    embedding_dim,
    trainable=False,
)
embedding_layer.build((1,))
embedding_layer.set_weights([embedding_matrix])
```

## Build the model {#build-the-model}

A simple 1D convnet with global max pooling and a classifier at the end.

```python
int_sequences_input = keras.Input(shape=(None,), dtype="int32")
embedded_sequences = embedding_layer(int_sequences_input)
x = layers.Conv1D(128, 5, activation="relu")(embedded_sequences)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(128, 5, activation="relu")(x)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(128, 5, activation="relu")(x)
x = layers.GlobalMaxPooling1D()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.5)(x)
preds = layers.Dense(len(class_names), activation="softmax")(x)
model = keras.Model(int_sequences_input, preds)
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
│ embedding (Embedding)           │ (None, None, 100)         │  2,000,200 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ conv1d (Conv1D)                 │ (None, None, 128)         │     64,128 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ max_pooling1d (MaxPooling1D)    │ (None, None, 128)         │          0 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ conv1d_1 (Conv1D)               │ (None, None, 128)         │     82,048 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ max_pooling1d_1 (MaxPooling1D)  │ (None, None, 128)         │          0 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ conv1d_2 (Conv1D)               │ (None, None, 128)         │     82,048 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ global_max_pooling1d            │ (None, 128)               │          0 │
│ (GlobalMaxPooling1D)            │                           │            │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ dense (Dense)                   │ (None, 128)               │     16,512 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ dropout (Dropout)               │ (None, 128)               │          0 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ dense_1 (Dense)                 │ (None, 20)                │      2,580 │
└─────────────────────────────────┴───────────────────────────┴────────────┘
 Total params: 2,247,516 (8.57 MB)
 Trainable params: 2,247,516 (8.57 MB)
 Non-trainable params: 0 (0.00 B)
```

{{% /details %}}

## Train the model {#train-the-model}

First, convert our list-of-strings data to NumPy arrays of integer indices. The arrays are right-padded.

```python
x_train = vectorizer(np.array([[s] for s in train_samples])).numpy()
x_val = vectorizer(np.array([[s] for s in val_samples])).numpy()

y_train = np.array(train_labels)
y_val = np.array(val_labels)
```

We use categorical crossentropy as our loss since we're doing softmax classification. Moreover, we use `sparse_categorical_crossentropy` since our labels are integers.

```python
model.compile(
    loss="sparse_categorical_crossentropy", optimizer="rmsprop", metrics=["acc"]
)
model.fit(x_train, y_train, batch_size=128, epochs=20, validation_data=(x_val, y_val))
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Epoch 1/20
   2/125 [37m━━━━━━━━━━━━━━━━━━━━  9s 78ms/step - acc: 0.0352 - loss: 3.2164

WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1700434131.619687    6780 device_compiler.h:187] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.

 125/125 ━━━━━━━━━━━━━━━━━━━━ 22s 123ms/step - acc: 0.0926 - loss: 2.8961 - val_acc: 0.2451 - val_loss: 2.1965
Epoch 2/20
 125/125 ━━━━━━━━━━━━━━━━━━━━ 10s 78ms/step - acc: 0.2628 - loss: 2.1377 - val_acc: 0.4421 - val_loss: 1.6594
Epoch 3/20
 125/125 ━━━━━━━━━━━━━━━━━━━━ 10s 78ms/step - acc: 0.4504 - loss: 1.5765 - val_acc: 0.5849 - val_loss: 1.2577
Epoch 4/20
 125/125 ━━━━━━━━━━━━━━━━━━━━ 10s 76ms/step - acc: 0.5711 - loss: 1.2639 - val_acc: 0.6277 - val_loss: 1.1153
Epoch 5/20
 125/125 ━━━━━━━━━━━━━━━━━━━━ 9s 74ms/step - acc: 0.6430 - loss: 1.0318 - val_acc: 0.6684 - val_loss: 0.9902
Epoch 6/20
 125/125 ━━━━━━━━━━━━━━━━━━━━ 9s 72ms/step - acc: 0.6990 - loss: 0.8844 - val_acc: 0.6619 - val_loss: 1.0109
Epoch 7/20
 125/125 ━━━━━━━━━━━━━━━━━━━━ 9s 70ms/step - acc: 0.7330 - loss: 0.7614 - val_acc: 0.6832 - val_loss: 0.9585
Epoch 8/20
 125/125 ━━━━━━━━━━━━━━━━━━━━ 8s 68ms/step - acc: 0.7795 - loss: 0.6328 - val_acc: 0.6847 - val_loss: 0.9917
Epoch 9/20
 125/125 ━━━━━━━━━━━━━━━━━━━━ 8s 64ms/step - acc: 0.8203 - loss: 0.5242 - val_acc: 0.7187 - val_loss: 0.9224
Epoch 10/20
 125/125 ━━━━━━━━━━━━━━━━━━━━ 8s 60ms/step - acc: 0.8506 - loss: 0.4265 - val_acc: 0.7342 - val_loss: 0.9098
Epoch 11/20
 125/125 ━━━━━━━━━━━━━━━━━━━━ 7s 56ms/step - acc: 0.8756 - loss: 0.3659 - val_acc: 0.7204 - val_loss: 1.0022
Epoch 12/20
 125/125 ━━━━━━━━━━━━━━━━━━━━ 7s 54ms/step - acc: 0.8921 - loss: 0.3079 - val_acc: 0.7209 - val_loss: 1.0477
Epoch 13/20
 125/125 ━━━━━━━━━━━━━━━━━━━━ 7s 54ms/step - acc: 0.9077 - loss: 0.2767 - val_acc: 0.7169 - val_loss: 1.0915
Epoch 14/20
 125/125 ━━━━━━━━━━━━━━━━━━━━ 6s 50ms/step - acc: 0.9244 - loss: 0.2253 - val_acc: 0.7382 - val_loss: 1.1397
Epoch 15/20
 125/125 ━━━━━━━━━━━━━━━━━━━━ 6s 49ms/step - acc: 0.9301 - loss: 0.2054 - val_acc: 0.7562 - val_loss: 1.0984
Epoch 16/20
 125/125 ━━━━━━━━━━━━━━━━━━━━ 5s 42ms/step - acc: 0.9373 - loss: 0.1769 - val_acc: 0.7387 - val_loss: 1.2294
Epoch 17/20
 125/125 ━━━━━━━━━━━━━━━━━━━━ 5s 41ms/step - acc: 0.9467 - loss: 0.1626 - val_acc: 0.7009 - val_loss: 1.4906
Epoch 18/20
 125/125 ━━━━━━━━━━━━━━━━━━━━ 5s 39ms/step - acc: 0.9471 - loss: 0.1544 - val_acc: 0.7184 - val_loss: 1.6050
Epoch 19/20
 125/125 ━━━━━━━━━━━━━━━━━━━━ 5s 37ms/step - acc: 0.9532 - loss: 0.1388 - val_acc: 0.7407 - val_loss: 1.4360
Epoch 20/20
 125/125 ━━━━━━━━━━━━━━━━━━━━ 5s 37ms/step - acc: 0.9519 - loss: 0.1388 - val_acc: 0.7309 - val_loss: 1.5327

<keras.src.callbacks.history.History at 0x7fbf50e6b910>
```

{{% /details %}}

## Export an end-to-end model {#export-an-end-to-end-model}

Now, we may want to export a `Model` object that takes as input a string of arbitrary length, rather than a sequence of indices. It would make the model much more portable, since you wouldn't have to worry about the input preprocessing pipeline.

Our `vectorizer` is actually a Keras layer, so it's simple:

```python
string_input = keras.Input(shape=(1,), dtype="string")
x = vectorizer(string_input)
preds = model(x)
end_to_end_model = keras.Model(string_input, preds)

probabilities = end_to_end_model(
    keras.ops.convert_to_tensor(
        [["this message is about computer graphics and 3D modeling"]]
    )
)

print(class_names[np.argmax(probabilities[0])])
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
comp.graphics
```

{{% /details %}}
