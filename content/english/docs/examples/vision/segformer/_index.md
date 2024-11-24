---
title: Semantic segmentation with SegFormer and Hugging Face Transformers
toc: true
weight: 75
type: docs
---

{{< keras/original checkedAt="2024-11-21" >}}

**Author:** [Sayak Paul](https://twitter.com/RisingSayak)  
**Date created:** 2023/01/25  
**Last modified:** 2023/01/29  
**Description:** Fine-tuning a SegFormer model variant for semantic segmentation.

{{< keras/version v=2 >}}

{{< cards cols="2" >}}
{{< card link="https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/vision/ipynb/segformer.ipynb" title="Colab" tag="Colab" tagType="warning">}}
{{< card link="https://github.com/keras-team/keras-io/blob/master/examples/vision/segformer.py" title="GitHub" tag="GitHub">}}
{{< /cards >}}

## Introduction

In this example, we show how to fine-tune a SegFormer model variant to do semantic segmentation on a custom dataset. Semantic segmentation is the task of assigning a category to each and every pixel of an image. SegFormer was proposed in [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203). SegFormer uses a hierarchical Transformer architecture (called "Mix Transformer") as its encoder and a lightweight decoder for segmentation. As a result, it yields state-of-the-art performance on semantic segmentation while being more efficient than existing models. For more details, check out the original paper.

![segformer-arch](/images/examples/vision/segformer/BsrVwYe.png)

We leverage [Hugging Face Transformers](https://github.com/huggingface/transformers) to load a pretrained SegFormer checkpoint and fine-tune it on a custom dataset.

**Note:** this example reuses code from the following sources:

- [Official tutorial on segmentation from the TensorFlow team](https://www.tensorflow.org/tutorials/images/segmentation)
- [Hugging Face Task guide on segmentation](https://huggingface.co/docs/transformers/main/en/tasks/semantic_segmentation)

To run this example, we need to install the `transformers` library:

```python
!!pip install transformers -q
```

{{% details title="Result" closed="true" %}}

```plain
[]
```

{{% /details %}}

## Load the data

We use the [Oxford-IIIT Pets](https://www.robots.ox.ac.uk/~vgg/data/pets/) dataset for this example. We leverage `tensorflow_datasets` to load the dataset.

```python
import tensorflow_datasets as tfds

dataset, info = tfds.load("oxford_iiit_pet:3.*.*", with_info=True)
```

{{% details title="Result" closed="true" %}}

```plain
/opt/conda/lib/python3.7/site-packages/tensorflow_io/python/ops/__init__.py:98: UserWarning: unable to load libtensorflow_io_plugins.so: unable to open file: libtensorflow_io_plugins.so, from paths: ['/opt/conda/lib/python3.7/site-packages/tensorflow_io/python/ops/libtensorflow_io_plugins.so']
caused by: ['/opt/conda/lib/python3.7/site-packages/tensorflow_io/python/ops/libtensorflow_io_plugins.so: undefined symbol: _ZN3tsl5mutexC1Ev']
  warnings.warn(f"unable to load libtensorflow_io_plugins.so: {e}")
/opt/conda/lib/python3.7/site-packages/tensorflow_io/python/ops/__init__.py:104: UserWarning: file system plugins are not loaded: unable to open file: libtensorflow_io.so, from paths: ['/opt/conda/lib/python3.7/site-packages/tensorflow_io/python/ops/libtensorflow_io.so']
caused by: ['/opt/conda/lib/python3.7/site-packages/tensorflow_io/python/ops/libtensorflow_io.so: undefined symbol: _ZNK10tensorflow4data11DatasetBase8FinalizeEPNS_15OpKernelContextESt8functionIFN3tsl8StatusOrISt10unique_ptrIS1_NS5_4core15RefCountDeleterEEEEvEE']
  warnings.warn(f"file system plugins are not loaded: {e}")
```

{{% /details %}}

## Prepare the datasets

For preparing the datasets for training and evaluation, we:

- Normalize the images with the mean and standard deviation used during pre-training SegFormer.
- Subtract 1 from the segmentation masks so that the pixel values start from 0.
- Resize the images.
- Transpose the images such that they are in `"channels_first"` format. This is to make them compatible with the SegFormer model from Hugging Face Transformers.

```python
import tensorflow as tf
from tensorflow.keras import backend

image_size = 512
mean = tf.constant([0.485, 0.456, 0.406])
std = tf.constant([0.229, 0.224, 0.225])


def normalize(input_image, input_mask):
    input_image = tf.image.convert_image_dtype(input_image, tf.float32)
    input_image = (input_image - mean) / tf.maximum(std, backend.epsilon())
    input_mask -= 1
    return input_image, input_mask


def load_image(datapoint):
    input_image = tf.image.resize(datapoint["image"], (image_size, image_size))
    input_mask = tf.image.resize(
        datapoint["segmentation_mask"],
        (image_size, image_size),
        method="bilinear",
    )

    input_image, input_mask = normalize(input_image, input_mask)
    input_image = tf.transpose(input_image, (2, 0, 1))
    return {"pixel_values": input_image, "labels": tf.squeeze(input_mask)}
```

We now use the above utilities to prepare [`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset) objects including `prefetch()` for performance. Change the `batch_size` to match the size of the GPU memory on the GPU that you're using for training.

```python
auto = tf.data.AUTOTUNE
batch_size = 4

train_ds = (
    dataset["train"]
    .cache()
    .shuffle(batch_size * 10)
    .map(load_image, num_parallel_calls=auto)
    .batch(batch_size)
    .prefetch(auto)
)
test_ds = (
    dataset["test"]
    .map(load_image, num_parallel_calls=auto)
    .batch(batch_size)
    .prefetch(auto)
)
```

We can check the shapes of the input images and their segmentation maps:

```python
print(train_ds.element_spec)
```

{{% details title="Result" closed="true" %}}

```plain
{'pixel_values': TensorSpec(shape=(None, 3, 512, 512), dtype=tf.float32, name=None), 'labels': TensorSpec(shape=(None, 512, 512), dtype=tf.float32, name=None)}
```

{{% /details %}}

## Visualize dataset

```python
import matplotlib.pyplot as plt


def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ["Input Image", "True Mask", "Predicted Mask"]

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis("off")
    plt.show()


for samples in train_ds.take(2):
    sample_image, sample_mask = samples["pixel_values"][0], samples["labels"][0]
    sample_image = tf.transpose(sample_image, (1, 2, 0))
    sample_mask = tf.expand_dims(sample_mask, -1)
    display([sample_image, sample_mask])
```

![png](/images/examples/vision/segformer/segformer_12_0.png)

![png](/images/examples/vision/segformer/segformer_12_1.png)

## Load a pretrained SegFormer checkpoint

We now load a pretrained SegFormer model variant from Hugging Face Transformers. The SegFormer model comes in different variants dubbed as **MiT-B0** to **MiT-B5**. You can find these checkpoints [here](https://huggingface.co/models?pipeline_tag=image-segmentation&sort=downloads&search=segformer). We load the smallest variant Mix-B0, which produces a good trade-off between inference efficiency and predictive performance.

```python
from transformers import TFSegformerForSemanticSegmentation

model_checkpoint = "nvidia/mit-b0"
id2label = {0: "outer", 1: "inner", 2: "border"}
label2id = {label: id for id, label in id2label.items()}
num_labels = len(id2label)
model = TFSegformerForSemanticSegmentation.from_pretrained(
    model_checkpoint,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)
```

{{% details title="Result" closed="true" %}}

```plain
WARNING:tensorflow:5 out of the last 5 calls to <function Conv._jit_compiled_convolution_op at 0x7fa8cc1139e0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.

WARNING:tensorflow:5 out of the last 5 calls to <function Conv._jit_compiled_convolution_op at 0x7fa8cc1139e0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.

WARNING:tensorflow:6 out of the last 6 calls to <function Conv._jit_compiled_convolution_op at 0x7fa8bde37440> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.

WARNING:tensorflow:6 out of the last 6 calls to <function Conv._jit_compiled_convolution_op at 0x7fa8bde37440> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
Some layers from the model checkpoint at nvidia/mit-b0 were not used when initializing TFSegformerForSemanticSegmentation: ['classifier']
- This IS expected if you are initializing TFSegformerForSemanticSegmentation from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing TFSegformerForSemanticSegmentation from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Some layers of TFSegformerForSemanticSegmentation were not initialized from the model checkpoint at nvidia/mit-b0 and are newly initialized: ['decode_head']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
```

{{% /details %}}

The warning is telling us that we're throwing away some weights and newly initializing some others. Don't panic! This is absolutely normal. Since we're using a custom dataset which has a different set of semantic class labels than the pre-training dataset, [`TFSegformerForSemanticSegmentation`](https://huggingface.co/docs/transformers/model_doc/segformer#transformers.TFSegformerForSemanticSegmentation) is initializing a new decoder head.

We can now initialize an optimizer and compile the model with it.

## Compile the model

```python
lr = 0.00006
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
model.compile(optimizer=optimizer)
```

{{% details title="Result" closed="true" %}}

```plain
No loss specified in compile() - the model's internal loss computation will be used as the loss. Don't panic - this is a common way to train TensorFlow models in Transformers! To disable this behaviour please pass a loss argument, or explicitly pass `loss=None` if you do not want your model to compute a loss.
```

{{% /details %}}

Notice that we are not using any loss function for compiling the model. This is because the forward pass of the model [implements](https://github.com/huggingface/transformers/blob/820c46a707ddd033975bc3b0549eea200e64c7da/src/transformers/models/segformer/modeling_tf_segformer.py#L873) the loss computation part when we provide labels alongside the input images. After computing the loss, the model returned a structured `dataclass` object which is then used to guide the training process.

With the compiled model, we can proceed and call `fit()` on it to begin the fine-tuning process!

## Prediction callback to monitor training progress

It helps us to visualize some sample predictions when the model is being fine-tuned, thereby helping us to monitor the progress of the model. This callback is inspired from [this tutorial](https://www.tensorflow.org/tutorials/images/segmentation).

```python
from IPython.display import clear_output


def create_mask(pred_mask):
    pred_mask = tf.math.argmax(pred_mask, axis=1)
    pred_mask = tf.expand_dims(pred_mask, -1)
    return pred_mask[0]


def show_predictions(dataset=None, num=1):
    if dataset:
        for sample in dataset.take(num):
            images, masks = sample["pixel_values"], sample["labels"]
            masks = tf.expand_dims(masks, -1)
            pred_masks = model.predict(images).logits
            images = tf.transpose(images, (0, 2, 3, 1))
            display([images[0], masks[0], create_mask(pred_masks)])
    else:
        display(
            [
                sample_image,
                sample_mask,
                create_mask(model.predict(tf.expand_dims(sample_image, 0))),
            ]
        )


class DisplayCallback(tf.keras.callbacks.Callback):
    def __init__(self, dataset, **kwargs):
        super().__init__(**kwargs)
        self.dataset = dataset

    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        show_predictions(self.dataset)
        print("\nSample Prediction after epoch {}\n".format(epoch + 1))
```

## Train model

```python
# Increase the number of epochs if the results are not of expected quality.
epochs = 5

history = model.fit(
    train_ds,
    validation_data=test_ds,
    callbacks=[DisplayCallback(test_ds)],
    epochs=epochs,
)
```

{{% details title="Result" closed="true" %}}

```plain
1/1 [==============================] - 0s 54ms/step
```

![png](/images/examples/vision/segformer/segformer_22_1.png)

```plain
Sample Prediction after epoch 5
```

```plain
920/920 [==============================] - 89s 97ms/step - loss: 0.1742 - val_loss: 0.1927
```

{{% /details %}}

## Inference

We perform inference on a few samples from the test set.

```python
show_predictions(test_ds, 5)
```

{{% details title="Result" closed="true" %}}

```plain
1/1 [==============================] - 0s 54ms/step
```

![png](/images/examples/vision/segformer/segformer_24_1.png)

```plain
1/1 [==============================] - 0s 54ms/step
```

![png](/images/examples/vision/segformer/segformer_24_3.png)

```plain
1/1 [==============================] - 0s 53ms/step
```

![png](/images/examples/vision/segformer/segformer_24_5.png)

```plain
1/1 [==============================] - 0s 53ms/step
```

![png](/images/examples/vision/segformer/segformer_24_7.png)

```plain
1/1 [==============================] - 0s 53ms/step
```

![png](/images/examples/vision/segformer/segformer_24_9.png)

{{% /details %}}

## Conclusion

In this example, we learned how to fine-tune a SegFormer model variant on a custom dataset for semantic segmentation. In the interest of brevity, the example was kept short. However, there are a couple of things, you can further try out:

- Incorporate data augmentation to potentially improve the results.
- Use a larger SegFormer model checkpoint to see how the results are affected.
- Push the fine-tuned model to the Hugging Face for sharing with the community easily. You can do so just by doing `model.push_to_hub("your-username/your-awesome-model")`. And then you can load the model by doing `TFSegformerForSemanticSegmentation.from_pretrained("your-username/your-awesome-model"`). [Here](https://github.com/huggingface/notebooks/blob/main/examples/semantic_segmentation-tf.ipynb) is an end-to-end example if you're looking for a reference.
- If you'd rather push the model checkpoints to the Hub as the model is being fine-tuned you can instead use the `PushToHubCallback` Keras callback. [Here](https://gist.github.com/sayakpaul/f474ffb01f0cdcc8ba239357965c3bca) is an example. [Here](https://huggingface.co/sayakpaul/mit-b0-finetuned-pets) is an example of a model repository that was created using this callback.
