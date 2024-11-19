---
title: causal_lm
toc: false
---

[\[source\]](https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/causal_lm.py#L18)

### `CausalLM` class

`keras_hub.models.CausalLM()`

Base class for generative language modeling tasks.

`CausalLM` tasks wrap a [`keras_hub.models.Backbone`](/api/keras_hub/base_classes/backbone#backbone-class) and a [`keras_hub.models.Preprocessor`](/api/keras_hub/base_classes/preprocessor#preprocessor-class) to create a model that can be used for generation and generative fine-tuning.

`CausalLM` tasks provide an additional, high-level `generate()` function which can be used to auto-regressively sample a model token by token with a string in, string out signature. The `compile()` method of all `CausalLM` classes contains an additional `sampler` argument, which can be used to pass a [`keras_hub.samplers.Sampler`](/api/keras_hub/samplers/samplers#sampler-class) to control how the predicted distribution will be sampled.

When calling `fit()`, the tokenized input will be predicted token-by-token with a causal mask applied, which gives both a pre-training and supervised fine-tuning setup for controlling inference-time generation.

All `CausalLM` tasks include a `from_preset()` constructor which can be used to load a pre-trained config and weights.

**Example**

`# Load a GPT2 backbone with pre-trained weights. causal_lm = keras_hub.models.CausalLM.from_preset(     "gpt2_base_en", ) causal_lm.compile(sampler="top_k") causal_lm.generate("Keras is a", max_length=64)  # Load a Mistral instruction tuned checkpoint at bfloat16 precision. causal_lm = keras_hub.models.CausalLM.from_preset(     "mistral_instruct_7b_en",     dtype="bfloat16", ) causal_lm.compile(sampler="greedy") causal_lm.generate("Keras is a", max_length=64)`

---

[\[source\]](https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/task.py#L129)

### `from_preset` method

`CausalLM.from_preset(preset, load_weights=True, **kwargs)`

Instantiate a [`keras_hub.models.Task`](/api/keras_hub/base_classes/task#task-class) from a model preset.

A preset is a directory of configs, weights and other file assets used to save and load a pre-trained model. The `preset` can be passed as one of:

1.  a built-in preset identifier like `'bert_base_en'`
2.  a Kaggle Models handle like `'kaggle://user/bert/keras/bert_base_en'`
3.  a Hugging Face handle like `'hf://user/bert_base_en'`
4.  a path to a local preset directory like `'./bert_base_en'`

For any `Task` subclass, you can run `cls.presets.keys()` to list all built-in presets available on the class.

This constructor can be called in one of two ways. Either from a task specific base class like `keras_hub.models.CausalLM.from_preset()`, or from a model class like `keras_hub.models.BertTextClassifier.from_preset()`. If calling from the a base class, the subclass of the returning object will be inferred from the config in the preset directory.

**Arguments**

- **preset**: string. A built-in preset identifier, a Kaggle Models handle, a Hugging Face handle, or a path to a local directory.
- **load_weights**: bool. If `True`, saved weights will be loaded into the model architecture. If `False`, all weights will be randomly initialized.

**Examples**

`# Load a Gemma generative task. causal_lm = keras_hub.models.CausalLM.from_preset(     "gemma_2b_en", )  # Load a Bert classification task. model = keras_hub.models.TextClassifier.from_preset(     "bert_base_en",     num_classes=2, )`

---

[\[source\]](https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/causal_lm.py#L62)

### `compile` method

`CausalLM.compile(     optimizer="auto", loss="auto", weighted_metrics="auto", sampler="top_k", **kwargs )`

Configures the `CausalLM` task for training and generation.

The `CausalLM` task extends the default compilation signature of [`keras.Model.compile`](/api/models/model_training_apis#compile-method) with defaults for `optimizer`, `loss`, and `weighted_metrics`. To override these defaults, pass any value to these arguments during compilation.

The `CausalLM` task adds a new `sampler` to `compile`, which can be used to control the sampling strategy used with the `generate` function.

Note that because training inputs include padded tokens which are excluded from the loss, it is almost always a good idea to compile with `weighted_metrics` and not `metrics`.

**Arguments**

- **optimizer**: `"auto"`, an optimizer name, or a `keras.Optimizer` instance. Defaults to `"auto"`, which uses the default optimizer for the given model and task. See [`keras.Model.compile`](/api/models/model_training_apis#compile-method) and `keras.optimizers` for more info on possible `optimizer` values.
- **loss**: `"auto"`, a loss name, or a [`keras.losses.Loss`](/api/losses#loss-class) instance. Defaults to `"auto"`, where a [`keras.losses.SparseCategoricalCrossentropy`](/api/losses/probabilistic_losses#sparsecategoricalcrossentropy-class) loss will be applied for the token classification `CausalLM` task. See [`keras.Model.compile`](/api/models/model_training_apis#compile-method) and `keras.losses` for more info on possible `loss` values.
- **weighted_metrics**: `"auto"`, or a list of metrics to be evaluated by the model during training and testing. Defaults to `"auto"`, where a [`keras.metrics.SparseCategoricalAccuracy`](/api/metrics/accuracy_metrics#sparsecategoricalaccuracy-class) will be applied to track the accuracy of the model at guessing masked token values. See [`keras.Model.compile`](/api/models/model_training_apis#compile-method) and `keras.metrics` for more info on possible `weighted_metrics` values.
- **sampler**: A sampler name, or a [`keras_hub.samplers.Sampler`](/api/keras_hub/samplers/samplers#sampler-class) instance. Configures the sampling method used during `generate()` calls. See `keras_hub.samplers` for a full list of built-in sampling strategies.
- **\*\*kwargs**: See [`keras.Model.compile`](/api/models/model_training_apis#compile-method) for a full list of arguments supported by the compile method.

---

[\[source\]](https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/causal_lm.py#L272)

### `generate` method

`CausalLM.generate(inputs, max_length=None, stop_token_ids="auto", strip_prompt=False)`

Generate text given prompt `inputs`.

This method generates text based on given `inputs`. The sampling method used for generation can be set via the `compile()` method.

If `inputs` are a [`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset), outputs will be generated "batch-by-batch" and concatenated. Otherwise, all inputs will be handled as a single batch.

If a `preprocessor` is attached to the model, `inputs` will be preprocessed inside the `generate()` function and should match the structure expected by the `preprocessor` layer (usually raw strings). If a `preprocessor` is not attached, inputs should match the structure expected by the `backbone`. See the example usage above for a demonstration of each.

**Arguments**

- **inputs**: python data, tensor data, or a [`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset). If a `preprocessor` is attached to the model, `inputs` should match the structure expected by the `preprocessor` layer. If a `preprocessor` is not attached, `inputs` should match the structure expected the `backbone` model.
- **max_length**: Optional. int. The max length of the generated sequence. Will default to the max configured `sequence_length` of the `preprocessor`. If `preprocessor` is `None`, `inputs` should be should be padded to the desired maximum length and this argument will be ignored.
- **stop_token_ids**: Optional. `None`, "auto", or tuple of token ids. Defaults to "auto" which uses the `preprocessor.tokenizer.end_token_id`. Not specifying a processor will produce an error. None stops generation after generating `max_length` tokens. You may also specify a list of token id's the model should stop on. Note that sequences of tokens will each be interpreted as a stop token, multi-token stop sequences are not supported.
- **strip_prompt**: Optional. By default, generate() returns the full prompt followed by its completion generated by the model. If this option is set to True, only the newly generated text is returned.

---

[\[source\]](https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/task.py#L238)

### `save_to_preset` method

`CausalLM.save_to_preset(preset_dir)`

Save task to a preset directory.

**Arguments**

- **preset_dir**: The path to the local model preset directory.

---

### `preprocessor` property

`keras_hub.models.CausalLM.preprocessor`

A [`keras_hub.models.Preprocessor`](/api/keras_hub/base_classes/preprocessor#preprocessor-class) layer used to preprocess input.

---

### `backbone` property

`keras_hub.models.CausalLM.backbone`

A [`keras_hub.models.Backbone`](/api/keras_hub/base_classes/backbone#backbone-class) model with the core architecture.

---
