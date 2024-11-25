---
title: ImageClassifier
toc: true
weight: 4
type: docs
---

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/image_classifier.py#L7" >}}

### `ImageClassifier` class

`keras_hub.models.ImageClassifier(     backbone,     num_classes,     preprocessor=None,     pooling="avg",     activation=None,     dropout=0.0,     head_dtype=None,     **kwargs )`

Base class for all image classification tasks.

`ImageClassifier` tasks wrap a [`keras_hub.models.Backbone`](/api/keras_hub/base_classes/backbone#backbone-class) and a [`keras_hub.models.Preprocessor`](/api/keras_hub/base_classes/preprocessor#preprocessor-class) to create a model that can be used for image classification. `ImageClassifier` tasks take an additional `num_classes` argument, controlling the number of predicted output classes.

To fine-tune with `fit()`, pass a dataset containing tuples of `(x, y)` labels where `x` is a string and `y` is a integer from `[0, num_classes)`. All `ImageClassifier` tasks include a `from_preset()` constructor which can be used to load a pre-trained config and weights.

**Arguments**

- **backbone**: A [`keras_hub.models.Backbone`](/api/keras_hub/base_classes/backbone#backbone-class) instance or a [`keras.Model`](/api/models/model#model-class).
- **num_classes**: int. The number of classes to predict.
- **preprocessor**: `None`, a [`keras_hub.models.Preprocessor`](/api/keras_hub/base_classes/preprocessor#preprocessor-class) instance, a `keras.Layer` instance, or a callable. If `None` no preprocessing will be applied to the inputs.
- **pooling**: `"avg"` or `"max"`. The type of pooling to apply on backbone output. Defaults to average pooling.
- **activation**: `None`, str, or callable. The activation function to use on the `Dense` layer. Set `activation=None` to return the output logits. Defaults to `"softmax"`.
- **head_dtype**: `None`, str, or `keras.mixed_precision.DTypePolicy`. The dtype to use for the classification head's computations and weights.

**Examples**

Call `predict()` to run inference.

`# Load preset and train images = np.random.randint(0, 256, size=(2, 224, 224, 3)) classifier = keras_hub.models.ImageClassifier.from_preset(     "resnet_50_imagenet" ) classifier.predict(images)`

Call `fit()` on a single batch.

`# Load preset and train images = np.random.randint(0, 256, size=(2, 224, 224, 3)) labels = [0, 3] classifier = keras_hub.models.ImageClassifier.from_preset(     "resnet_50_imagenet" ) classifier.fit(x=images, y=labels, batch_size=2)`

Call `fit()` with custom loss, optimizer and backbone.

`classifier = keras_hub.models.ImageClassifier.from_preset(     "resnet_50_imagenet" ) classifier.compile(     loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),     optimizer=keras.optimizers.Adam(5e-5), ) classifier.backbone.trainable = False classifier.fit(x=images, y=labels, batch_size=2)`

Custom backbone.

`images = np.random.randint(0, 256, size=(2, 224, 224, 3)) labels = [0, 3] backbone = keras_hub.models.ResNetBackbone(     stackwise_num_filters=[64, 64, 64],     stackwise_num_blocks=[2, 2, 2],     stackwise_num_strides=[1, 2, 2],     block_type="basic_block",     use_pre_activation=True,     pooling="avg", ) classifier = keras_hub.models.ImageClassifier(     backbone=backbone,     num_classes=4, ) classifier.fit(x=images, y=labels, batch_size=2)`

---

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/task.py#L129" >}}

### `from_preset` method

`ImageClassifier.from_preset(preset, load_weights=True, **kwargs)`

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

Preset name

Parameters

Description

vgg_11_imagenet

9.22M

11-layer vgg model pre-trained on the ImageNet 1k dataset at a 224x224 resolution.

vgg_13_imagenet

9.40M

13-layer vgg model pre-trained on the ImageNet 1k dataset at a 224x224 resolution.

vgg_16_imagenet

14.71M

16-layer vgg model pre-trained on the ImageNet 1k dataset at a 224x224 resolution.

vgg_19_imagenet

20.02M

19-layer vgg model pre-trained on the ImageNet 1k dataset at a 224x224 resolution.

mit_b0_ade20k_512

3.32M

MiT (MixTransformer) model with 8 transformer blocks.

mit_b1_ade20k_512

13.16M

MiT (MixTransformer) model with 8 transformer blocks.

mit_b2_ade20k_512

24.20M

MiT (MixTransformer) model with 16 transformer blocks.

mit_b3_ade20k_512

44.08M

MiT (MixTransformer) model with 28 transformer blocks.

mit_b4_ade20k_512

60.85M

MiT (MixTransformer) model with 41 transformer blocks.

mit_b5_ade20k_640

81.45M

MiT (MixTransformer) model with 52 transformer blocks.

mit_b0_cityscapes_1024

3.32M

MiT (MixTransformer) model with 8 transformer blocks.

mit_b1_cityscapes_1024

13.16M

MiT (MixTransformer) model with 8 transformer blocks.

mit_b2_cityscapes_1024

24.20M

MiT (MixTransformer) model with 16 transformer blocks.

mit_b3_cityscapes_1024

44.08M

MiT (MixTransformer) model with 28 transformer blocks.

mit_b4_cityscapes_1024

60.85M

MiT (MixTransformer) model with 41 transformer blocks.

mit_b5_cityscapes_1024

81.45M

MiT (MixTransformer) model with 52 transformer blocks.

resnet_18_imagenet

11.19M

18-layer ResNet model pre-trained on the ImageNet 1k dataset at a 224x224 resolution.

resnet_50_imagenet

23.56M

50-layer ResNet model pre-trained on the ImageNet 1k dataset at a 224x224 resolution.

resnet_101_imagenet

42.61M

101-layer ResNet model pre-trained on the ImageNet 1k dataset at a 224x224 resolution.

resnet_152_imagenet

58.30M

152-layer ResNet model pre-trained on the ImageNet 1k dataset at a 224x224 resolution.

resnet_v2_50_imagenet

23.56M

50-layer ResNetV2 model pre-trained on the ImageNet 1k dataset at a 224x224 resolution.

resnet_v2_101_imagenet

42.61M

101-layer ResNetV2 model pre-trained on the ImageNet 1k dataset at a 224x224 resolution.

resnet_vd_18_imagenet

11.72M

18-layer ResNetVD (ResNet with bag of tricks) model pre-trained on the ImageNet 1k dataset at a 224x224 resolution.

resnet_vd_34_imagenet

21.84M

34-layer ResNetVD (ResNet with bag of tricks) model pre-trained on the ImageNet 1k dataset at a 224x224 resolution.

resnet_vd_50_imagenet

25.63M

50-layer ResNetVD (ResNet with bag of tricks) model pre-trained on the ImageNet 1k dataset at a 224x224 resolution.

resnet_vd_50_ssld_imagenet

25.63M

50-layer ResNetVD (ResNet with bag of tricks) model pre-trained on the ImageNet 1k dataset at a 224x224 resolution with knowledge distillation.

resnet_vd_50_ssld_v2_imagenet

25.63M

50-layer ResNetVD (ResNet with bag of tricks) model pre-trained on the ImageNet 1k dataset at a 224x224 resolution with knowledge distillation and AutoAugment.

resnet_vd_50_ssld_v2_fix_imagenet

25.63M

50-layer ResNetVD (ResNet with bag of tricks) model pre-trained on the ImageNet 1k dataset at a 224x224 resolution with knowledge distillation, AutoAugment and additional fine-tuning of the classification head.

resnet_vd_101_imagenet

44.67M

101-layer ResNetVD (ResNet with bag of tricks) model pre-trained on the ImageNet 1k dataset at a 224x224 resolution.

resnet_vd_101_ssld_imagenet

44.67M

101-layer ResNetVD (ResNet with bag of tricks) model pre-trained on the ImageNet 1k dataset at a 224x224 resolution with knowledge distillation.

resnet_vd_152_imagenet

60.36M

152-layer ResNetVD (ResNet with bag of tricks) model pre-trained on the ImageNet 1k dataset at a 224x224 resolution.

resnet_vd_200_imagenet

74.93M

200-layer ResNetVD (ResNet with bag of tricks) model pre-trained on the ImageNet 1k dataset at a 224x224 resolution.

densenet_121_imagenet

7.04M

121-layer DenseNet model pre-trained on the ImageNet 1k dataset at a 224x224 resolution.

densenet_169_imagenet

12.64M

169-layer DenseNet model pre-trained on the ImageNet 1k dataset at a 224x224 resolution.

densenet_201_imagenet

18.32M

201-layer DenseNet model pre-trained on the ImageNet 1k dataset at a 224x224 resolution.

---

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/image_classifier.py#L168" >}}

### `compile` method

`ImageClassifier.compile(optimizer="auto", loss="auto", metrics="auto", **kwargs)`

Configures the `ImageClassifier` task for training.

The `ImageClassifier` task extends the default compilation signature of [`keras.Model.compile`](/api/models/model_training_apis#compile-method) with defaults for `optimizer`, `loss`, and `metrics`. To override these defaults, pass any value to these arguments during compilation.

**Arguments**

- **optimizer**: `"auto"`, an optimizer name, or a `keras.Optimizer` instance. Defaults to `"auto"`, which uses the default optimizer for the given model and task. See [`keras.Model.compile`](/api/models/model_training_apis#compile-method) and `keras.optimizers` for more info on possible `optimizer` values.
- **loss**: `"auto"`, a loss name, or a [`keras.losses.Loss`](/api/losses#loss-class) instance. Defaults to `"auto"`, where a [`keras.losses.SparseCategoricalCrossentropy`](/api/losses/probabilistic_losses#sparsecategoricalcrossentropy-class) loss will be applied for the classification task. See [`keras.Model.compile`](/api/models/model_training_apis#compile-method) and `keras.losses` for more info on possible `loss` values.
- **metrics**: `"auto"`, or a list of metrics to be evaluated by the model during training and testing. Defaults to `"auto"`, where a [`keras.metrics.SparseCategoricalAccuracy`](/api/metrics/accuracy_metrics#sparsecategoricalaccuracy-class) will be applied to track the accuracy of the model during training. See [`keras.Model.compile`](/api/models/model_training_apis#compile-method) and `keras.metrics` for more info on possible `metrics` values.
- **\*\*kwargs**: See [`keras.Model.compile`](/api/models/model_training_apis#compile-method) for a full list of arguments supported by the compile method.

---

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/task.py#L238" >}}

### `save_to_preset` method

`ImageClassifier.save_to_preset(preset_dir)`

Save task to a preset directory.

**Arguments**

- **preset_dir**: The path to the local model preset directory.

---

### `preprocessor` property

`keras_hub.models.ImageClassifier.preprocessor`

A [`keras_hub.models.Preprocessor`](/api/keras_hub/base_classes/preprocessor#preprocessor-class) layer used to preprocess input.

---

### `backbone` property

`keras_hub.models.ImageClassifier.backbone`

A [`keras_hub.models.Backbone`](/api/keras_hub/base_classes/backbone#backbone-class) model with the core architecture.

---
