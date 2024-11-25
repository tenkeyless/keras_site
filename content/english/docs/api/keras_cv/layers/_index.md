---
title: KerasCV Layers
linkTitle: Layers
toc: true
weight: 1
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

KerasCV layers are [`keras.layers.Layer`]({{< relref "/docs/api/layers/base_layer#layer-class" >}}) subclasses for computer vision specific use cases.

### [Augmentation layers]({{< relref "/docs/api/keras_cv/layers/augmentation/" >}})

- [AutoContrast layer]({{< relref "/docs/api/keras_cv/layers/augmentation/auto_contrast" >}})
- [AugMix layer]({{< relref "/docs/api/keras_cv/layers/augmentation/aug_mix" >}})
- [ChannelShuffle layer]({{< relref "/docs/api/keras_cv/layers/augmentation/channel_shuffle" >}})
- [CutMix layer]({{< relref "/docs/api/keras_cv/layers/augmentation/cut_mix" >}})
- [FourierMix layer]({{< relref "/docs/api/keras_cv/layers/augmentation/fourier_mix" >}})
- [GridMask layer]({{< relref "/docs/api/keras_cv/layers/augmentation/grid_mask" >}})
- [JitteredResize layer]({{< relref "/docs/api/keras_cv/layers/augmentation/jittered_resize" >}})
- [MixUp layer]({{< relref "/docs/api/keras_cv/layers/augmentation/mix_up" >}})
- [RandAugment layer]({{< relref "/docs/api/keras_cv/layers/augmentation/rand_augment" >}})
- [RandomAugmentationPipeline layer]({{< relref "/docs/api/keras_cv/layers/augmentation/random_augmentation_pipeline" >}})
- [RandomChannelShift layer]({{< relref "/docs/api/keras_cv/layers/augmentation/random_channel_shift" >}})
- [RandomColorDegeneration layer]({{< relref "/docs/api/keras_cv/layers/augmentation/random_color_degeneration" >}})
- [RandomCutout layer]({{< relref "/docs/api/keras_cv/layers/augmentation/random_cutout" >}})
- [RandomHue layer]({{< relref "/docs/api/keras_cv/layers/augmentation/random_hue" >}})
- [RandomSaturation layer]({{< relref "/docs/api/keras_cv/layers/augmentation/random_saturation" >}})
- [RandomSharpness layer]({{< relref "/docs/api/keras_cv/layers/augmentation/random_sharpness" >}})
- [RandomShear layer]({{< relref "/docs/api/keras_cv/layers/augmentation/random_shear" >}})
- [Solarization layer]({{< relref "/docs/api/keras_cv/layers/augmentation/solarization" >}})

### [Preprocessing layers]({{< relref "/docs/api/keras_cv/layers/preprocessing/" >}})

- [Resizing layer]({{< relref "/docs/api/keras_cv/layers/preprocessing/resizing" >}})
- [Grayscale layer]({{< relref "/docs/api/keras_cv/layers/preprocessing/grayscale" >}})
- [Equalization layer]({{< relref "/docs/api/keras_cv/layers/preprocessing/equalization" >}})
- [Posterization layer]({{< relref "/docs/api/keras_cv/layers/preprocessing/posterization" >}})

### [Regularization layers]({{< relref "/docs/api/keras_cv/layers/regularization/" >}})

- [DropBlock2D layer]({{< relref "/docs/api/keras_cv/layers/regularization/dropblock2d" >}})
- [DropPath layer]({{< relref "/docs/api/keras_cv/layers/regularization/drop_path" >}})
- [SqueezeAndExcite2D layer]({{< relref "/docs/api/keras_cv/layers/regularization/squeeze_and_excite_2d" >}})
- [SqueezeAndExcite2D layer]({{< relref "/docs/api/keras_cv/layers/regularization/squeeze_and_excite_2d" >}})
- [StochasticDepth layer]({{< relref "/docs/api/keras_cv/layers/regularization/stochastic_depth" >}})
