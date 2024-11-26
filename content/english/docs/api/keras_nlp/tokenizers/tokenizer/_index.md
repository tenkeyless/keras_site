---
title: Tokenizer
toc: true
weight: 1
type: docs
---

{{< keras/original checkedAt="2024-11-26" >}}

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/tokenizers/tokenizer.py#L18" >}}

### `Tokenizer` class

```python
keras_nlp.tokenizers.Tokenizer()
```

A base class for tokenizer layers.

Tokenizers in the KerasHub library should all subclass this layer.
The class provides two core methods `tokenize()` and `detokenize()` for
going from plain text to sequences and back. A tokenizer is a subclass of
[`keras.layers.Layer`]({{< relref "/docs/api/layers/base_layer#layer-class" >}}) and can be combined into a [`keras.Model`]({{< relref "/docs/api/models/model#model-class" >}}).

Subclassers should always implement the `tokenize()` method, which will also
be the default when calling the layer directly on inputs.

Subclassers can optionally implement the `detokenize()` method if the
tokenization is reversible. Otherwise, this can be skipped.

Subclassers should implement `get_vocabulary()`, `vocabulary_size()`,
`token_to_id()` and `id_to_token()` if applicable. For some simple
"vocab free" tokenizers, such as a whitespace splitter show below, these
methods do not apply and can be skipped.

**Example**

```python
class WhitespaceSplitterTokenizer(keras_hub.tokenizers.Tokenizer):
    def tokenize(self, inputs):
        return tf.strings.split(inputs)
    def detokenize(self, inputs):
        return tf.strings.reduce_join(inputs, separator=" ", axis=-1)
tokenizer = WhitespaceSplitterTokenizer()
# Tokenize some inputs.
tokenizer.tokenize("This is a test")
# Shorthard for `tokenize()`.
tokenizer("This is a test")
# Detokenize some outputs.
tokenizer.detokenize(["This", "is", "a", "test"])
```

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/tokenizers/tokenizer.py#L213" >}}

### `from_preset` method

```python
Tokenizer.from_preset(preset, config_file="tokenizer.json", **kwargs)
```

Instantiate a `keras_hub.models.Tokenizer` from a model preset.

A preset is a directory of configs, weights and other file assets used
to save and load a pre-trained model. The `preset` can be passed as
one of:

1. a built-in preset identifier like `'bert_base_en'`
2. a Kaggle Models handle like `'kaggle://user/bert/keras/bert_base_en'`
3. a Hugging Face handle like `'hf://user/bert_base_en'`
4. a path to a local preset directory like `'./bert_base_en'`

For any `Tokenizer` subclass, you can run `cls.presets.keys()` to list
all built-in presets available on the class.

This constructor can be called in one of two ways. Either from the base
class like `keras_hub.models.Tokenizer.from_preset()`, or from
a model class like `keras_hub.models.GemmaTokenizer.from_preset()`.
If calling from the base class, the subclass of the returning object
will be inferred from the config in the preset directory.

**Arguments**

- **preset**: string. A built-in preset identifier, a Kaggle Models
  handle, a Hugging Face handle, or a path to a local directory.
- **load_weights**: bool. If `True`, the weights will be loaded into the
  model architecture. If `False`, the weights will be randomly
  initialized.

**Examples**

```python
# Load a preset tokenizer.
tokenizer = keras_hub.tokenizer.Tokenizer.from_preset("bert_base_en")
# Tokenize some input.
tokenizer("The quick brown fox tripped.")
# Detokenize some input.
tokenizer.detokenize([5, 6, 7, 8, 9])
```

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/tokenizers/tokenizer.py#L186" >}}

### `save_to_preset` method

```python
Tokenizer.save_to_preset(preset_dir)
```

Save tokenizer to a preset directory.

**Arguments**

- **preset_dir**: The path to the local model preset directory.
