---
title: Models API
toc: true
weight: 1
type: docs
---

{{< keras/original checkedAt="2024-11-24" >}}

There are three ways to create Keras models:

- The [Sequential model]({{< relref "/docs/guides/sequential_model" >}}), which is very straightforward (a simple list of layers), but is limited to single-input, single-output stacks of layers (as the name gives away).
- The [Functional API]({{< relref "/docs/guides/functional_api" >}}), which is an easy-to-use, fully-featured API that supports arbitrary model architectures. For most people and most use cases, this is what you should be using. This is the Keras "industry strength" model.
- [Model subclassing]({{< relref "/docs/guides/making_new_layers_and_models_via_subclassing" >}}), where you implement everything from scratch on your own. Use this if you have complex, out-of-the-box research use cases.

## Models API overview

### [The Model class]({{< relref "/docs/api/models/model/" >}})

- [Model class]({{< relref "/docs/api/models/model/#model-class" >}})
- [summary method]({{< relref "/docs/api/models/model/#summary-method" >}})
- [get_layer method]({{< relref "/docs/api/models/model/#get_layer-method" >}})

### [The Sequential class]({{< relref "/docs/api/models/sequential/" >}})

- [Sequential class]({{< relref "/docs/api/models/sequential/#sequential-class" >}})
- [add method]({{< relref "/docs/api/models/sequential/#add-method" >}})
- [pop method]({{< relref "/docs/api/models/sequential/#pop-method" >}})

### [Model training APIs]({{< relref "/docs/api/models/model_training_apis/" >}})

- [compile method]({{< relref "/docs/api/models/model_training_apis/#compile-method" >}})
- [fit method]({{< relref "/docs/api/models/model_training_apis/#fit-method" >}})
- [evaluate method]({{< relref "/docs/api/models/model_training_apis/#evaluate-method" >}})
- [predict method]({{< relref "/docs/api/models/model_training_apis/#predict-method" >}})
- [train_on_batch method]({{< relref "/docs/api/models/model_training_apis/#train_on_batch-method" >}})
- [test_on_batch method]({{< relref "/docs/api/models/model_training_apis/#test_on_batch-method" >}})
- [predict_on_batch method]({{< relref "/docs/api/models/model_training_apis/#predict_on_batch-method" >}})

### [Saving & serialization]({{< relref "/docs/api/models/model_saving_apis/" >}})

- [Whole model saving & loading]({{< relref "/docs/api/models/model_saving_apis/model_saving_and_loading" >}})
- [Weights-only saving & loading]({{< relref "/docs/api/models/model_saving_apis/weights_saving_and_loading" >}})
- [Model config serialization]({{< relref "/docs/api/models/model_saving_apis/model_config_serialization" >}})
- [Model export for inference]({{< relref "/docs/api/models/model_saving_apis/export" >}})
- [Serialization utilities]({{< relref "/docs/api/models/model_saving_apis/serialization_utils" >}})
- [Keras weights file editor]({{< relref "/docs/api/models/model_saving_apis/keras_file_editor" >}})
