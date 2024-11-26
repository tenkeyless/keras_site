---
title: Saving & serialization
toc: true
weight: 4
type: docs
---

{{< keras/original checkedAt="2024-11-24" >}}

### [Whole model saving & loading]({{< relref "/docs/api/models/model_saving_apis/model_saving_and_loading/" >}})

- [save method]({{< relref "/docs/api/models/model_saving_apis/model_saving_and_loading/#save-method" >}})
- [save_model function]({{< relref "/docs/api/models/model_saving_apis/model_saving_and_loading/#save_model-function" >}})
- [load_model function]({{< relref "/docs/api/models/model_saving_apis/model_saving_and_loading/#load_model-function" >}})

### [Weights-only saving & loading]({{< relref "/docs/api/models/model_saving_apis/weights_saving_and_loading/" >}})

- [save_weights method]({{< relref "/docs/api/models/model_saving_apis/weights_saving_and_loading/#save_weights-method" >}})
- [load_weights method]({{< relref "/docs/api/models/model_saving_apis/weights_saving_and_loading/#load_weights-method" >}})

### [Model config serialization]({{< relref "/docs/api/models/model_saving_apis/model_config_serialization/" >}})

- [get_config method]({{< relref "/docs/api/models/model_saving_apis/model_config_serialization/#get_config-method" >}})
- [from_config method]({{< relref "/docs/api/models/model_saving_apis/model_config_serialization/#from_config-method" >}})
- [clone_model function]({{< relref "/docs/api/models/model_saving_apis/model_config_serialization/#clone_model-function" >}})

### [Model export for inference]({{< relref "/docs/api/models/model_saving_apis/export/" >}})

- [export method]({{< relref "/docs/api/models/model_saving_apis/export/#export-method" >}})
- [ExportArchive class]({{< relref "/docs/api/models/model_saving_apis/export/#exportarchive-class" >}})
- [add_endpoint method]({{< relref "/docs/api/models/model_saving_apis/export/#add_endpoint-method" >}})
- [add_variable_collection method]({{< relref "/docs/api/models/model_saving_apis/export/#add_variable_collection-method" >}})
- [track method]({{< relref "/docs/api/models/model_saving_apis/export/#track-method" >}})
- [write_out method]({{< relref "/docs/api/models/model_saving_apis/export/#write_out-method" >}})

### [Serialization utilities]({{< relref "/docs/api/models/model_saving_apis/serialization_utils/" >}})

- [serialize_keras_object function]({{< relref "/docs/api/models/model_saving_apis/serialization_utils/#serialize_keras_object-function" >}})
- [deserialize_keras_object function]({{< relref "/docs/api/models/model_saving_apis/serialization_utils/#deserialize_keras_object-function" >}})
- [custom_object_scope class]({{< relref "/docs/api/models/model_saving_apis/serialization_utils/#customobjectscope-class" >}})
- [get_custom_objects function]({{< relref "/docs/api/models/model_saving_apis/serialization_utils/#get_custom_objects-function" >}})
- [register_keras_serializable function]({{< relref "/docs/api/models/model_saving_apis/serialization_utils/#register_keras_serializable-function" >}})

### [Keras weights file editor]({{< relref "/docs/api/models/model_saving_apis/keras_file_editor/" >}})

- [KerasFileEditor class]({{< relref "/docs/api/models/model_saving_apis/keras_file_editor/#kerasfileeditor-class" >}})
- [summary method]({{< relref "/docs/api/models/model_saving_apis/keras_file_editor/#summary-method" >}})
- [compare method]({{< relref "/docs/api/models/model_saving_apis/keras_file_editor/#compare-method" >}})
- [save method]({{< relref "/docs/api/models/model_saving_apis/keras_file_editor/#save-method" >}})
- [rename_object method]({{< relref "/docs/api/models/model_saving_apis/keras_file_editor/#rename_object-method" >}})
- [delete_object method]({{< relref "/docs/api/models/model_saving_apis/keras_file_editor/#delete_object-method" >}})
- [add_object method]({{< relref "/docs/api/models/model_saving_apis/keras_file_editor/#add_object-method" >}})
- [delete_weight method]({{< relref "/docs/api/models/model_saving_apis/keras_file_editor/#delete_weight-method" >}})
- [add_weights method]({{< relref "/docs/api/models/model_saving_apis/keras_file_editor/#add_weights-method" >}})
