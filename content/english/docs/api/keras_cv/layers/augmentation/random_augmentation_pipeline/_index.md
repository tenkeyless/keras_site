---
title: random_augmentation_pipeline
toc: false
---

[\[source\]](https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/layers/preprocessing/random_augmentation_pipeline.py#L25)

### `RandomAugmentationPipeline` class

`keras_cv.layers.RandomAugmentationPipeline(     layers, augmentations_per_image, rate=1.0, auto_vectorize=False, seed=None, **kwargs )`

RandomAugmentationPipeline constructs a pipeline based on provided arguments.

The implemented policy does the following: for each input provided in `call`(), the policy first inputs a random number, if the number is < rate, the policy then selects a random layer from the provided list of `layers`. It then calls the `layer()` on the inputs. This is done `augmentations_per_image` times.

This layer can be used to create custom policies resembling `RandAugment` or `AutoAugment`.

**Example**

`# construct a list of layers layers = keras_cv.layers.RandAugment.get_standard_policy(     value_range=(0, 255), magnitude=0.75, magnitude_stddev=0.3 ) layers = layers[:4]  # slice out some layers you don't want for whatever                        reason layers = layers + [keras_cv.layers.GridMask()]  # create the pipeline. pipeline = keras_cv.layers.RandomAugmentationPipeline(     layers=layers, augmentations_per_image=3 )  augmented_images = pipeline(images)`

**Arguments**

- **layers**: a list of `keras.Layers`. These are randomly inputs during augmentation to augment the inputs passed in `call()`. The layers passed should subclass `BaseImageAugmentationLayer`. Passing `layers=[]` would result in a no-op.
- **augmentations_per_image**: the number of layers to apply to each inputs in the `call()` method.
- **rate**: the rate at which to apply each augmentation. This is applied on a per augmentation bases, so if `augmentations_per_image=3` and `rate=0.5`, the odds an image will receive no augmentations is 0.5^3, or 0.5_0.5_0.5.
- **auto_vectorize**: whether to use [`tf.vectorized_map`](https://www.tensorflow.org/api_docs/python/tf/vectorized_map) or [`tf.map_fn`](https://www.tensorflow.org/api_docs/python/tf/map_fn) to apply the augmentations. This offers a significant performance boost, but can only be used if all the layers provided to the `layers` argument support auto vectorization.
- **seed**: Integer. Used to create a random seed.

---
