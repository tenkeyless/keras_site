---
title: Knowledge Distillation
toc: true
weight: 62
type: docs
---

{{< keras/original checkedAt="2024-11-21" >}}

**{{< t f_author >}}** [Kenneth Borup](https://twitter.com/Kennethborup)  
**{{< t f_date_created >}}** 2020/09/01  
**{{< t f_last_modified >}}** 2020/09/01  
**{{< t f_description >}}** Implementation of classical Knowledge Distillation.

{{< keras/version v=3 >}}

{{< cards cols="2" >}}
{{< card link="https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/vision/ipynb/knowledge_distillation.ipynb" title="Colab" tag="Colab" tagType="warning">}}
{{< card link="https://github.com/keras-team/keras-io/blob/master/examples/vision/knowledge_distillation.py" title="GitHub" tag="GitHub">}}
{{< /cards >}}

## Introduction to Knowledge Distillation

Knowledge Distillation is a procedure for model compression, in which a small (student) model is trained to match a large pre-trained (teacher) model. Knowledge is transferred from the teacher model to the student by minimizing a loss function, aimed at matching softened teacher logits as well as ground-truth labels.

The logits are softened by applying a "temperature" scaling function in the softmax, effectively smoothing out the probability distribution and revealing inter-class relationships learned by the teacher.

**Reference:**

- [Hinton et al. (2015)](https://arxiv.org/abs/1503.02531)

## Setup

```python
import os

import keras
from keras import layers
from keras import ops
import numpy as np
```

## Construct `Distiller()` class

The custom `Distiller()` class, overrides the `Model` methods `compile`, `compute_loss`, and `call`. In order to use the distiller, we need:

- A trained teacher model
- A student model to train
- A student loss function on the difference between student predictions and ground-truth
- A distillation loss function, along with a `temperature`, on the difference between the soft student predictions and the soft teacher labels
- An `alpha` factor to weight the student and distillation loss
- An optimizer for the student and (optional) metrics to evaluate performance

In the `compute_loss` method, we perform a forward pass of both the teacher and student, calculate the loss with weighting of the `student_loss` and `distillation_loss` by `alpha` and `1 - alpha`, respectively. Note: only the student weights are updated.

```python
class Distiller(keras.Model):
    def __init__(self, student, teacher):
        super().__init__()
        self.teacher = teacher
        self.student = student

    def compile(
        self,
        optimizer,
        metrics,
        student_loss_fn,
        distillation_loss_fn,
        alpha=0.1,
        temperature=3,
    ):
        """Configure the distiller.

        Args:
            optimizer: Keras optimizer for the student weights
            metrics: Keras metrics for evaluation
            student_loss_fn: Loss function of difference between student
                predictions and ground-truth
            distillation_loss_fn: Loss function of difference between soft
                student predictions and soft teacher predictions
            alpha: weight to student_loss_fn and 1-alpha to distillation_loss_fn
            temperature: Temperature for softening probability distributions.
                Larger temperature gives softer distributions.
        """
        super().compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def compute_loss(
        self, x=None, y=None, y_pred=None, sample_weight=None, allow_empty=False
    ):
        teacher_pred = self.teacher(x, training=False)
        student_loss = self.student_loss_fn(y, y_pred)

        distillation_loss = self.distillation_loss_fn(
            ops.softmax(teacher_pred / self.temperature, axis=1),
            ops.softmax(y_pred / self.temperature, axis=1),
        ) * (self.temperature**2)

        loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss
        return loss

    def call(self, x):
        return self.student(x)
```

## Create student and teacher models

Initialy, we create a teacher model and a smaller student model. Both models are convolutional neural networks and created using `Sequential()`, but could be any Keras model.

```python
# Create the teacher
teacher = keras.Sequential(
    [
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(256, (3, 3), strides=(2, 2), padding="same"),
        layers.LeakyReLU(negative_slope=0.2),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same"),
        layers.Conv2D(512, (3, 3), strides=(2, 2), padding="same"),
        layers.Flatten(),
        layers.Dense(10),
    ],
    name="teacher",
)

# Create the student
student = keras.Sequential(
    [
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(16, (3, 3), strides=(2, 2), padding="same"),
        layers.LeakyReLU(negative_slope=0.2),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same"),
        layers.Conv2D(32, (3, 3), strides=(2, 2), padding="same"),
        layers.Flatten(),
        layers.Dense(10),
    ],
    name="student",
)

# Clone student for later comparison
student_scratch = keras.models.clone_model(student)
```

## Prepare the dataset

The dataset used for training the teacher and distilling the teacher is [MNIST]({{< relref "/docs/api/datasets/mnist" >}}), and the procedure would be equivalent for any other dataset, e.g. [CIFAR-10]({{< relref "/docs/api/datasets/cifar10" >}}), with a suitable choice of models. Both the student and teacher are trained on the training set and evaluated on the test set.

```python
# Prepare the train and test dataset.
batch_size = 64
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize data
x_train = x_train.astype("float32") / 255.0
x_train = np.reshape(x_train, (-1, 28, 28, 1))

x_test = x_test.astype("float32") / 255.0
x_test = np.reshape(x_test, (-1, 28, 28, 1))
```

## Train the teacher

In knowledge distillation we assume that the teacher is trained and fixed. Thus, we start by training the teacher model on the training set in the usual way.

```python
# Train teacher as usual
teacher.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

# Train and evaluate teacher on data.
teacher.fit(x_train, y_train, epochs=5)
teacher.evaluate(x_test, y_test)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Epoch 1/5
 1875/1875 ━━━━━━━━━━━━━━━━━━━━ 8s 3ms/step - loss: 0.2408 - sparse_categorical_accuracy: 0.9259
Epoch 2/5
 1875/1875 ━━━━━━━━━━━━━━━━━━━━ 5s 3ms/step - loss: 0.0912 - sparse_categorical_accuracy: 0.9726
Epoch 3/5
 1875/1875 ━━━━━━━━━━━━━━━━━━━━ 7s 4ms/step - loss: 0.0758 - sparse_categorical_accuracy: 0.9777
Epoch 4/5
 1875/1875 ━━━━━━━━━━━━━━━━━━━━ 5s 3ms/step - loss: 0.0690 - sparse_categorical_accuracy: 0.9797
Epoch 5/5
 1875/1875 ━━━━━━━━━━━━━━━━━━━━ 5s 3ms/step - loss: 0.0582 - sparse_categorical_accuracy: 0.9825
 313/313 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - loss: 0.0931 - sparse_categorical_accuracy: 0.9760

[0.09044107794761658, 0.978100061416626]
```

{{% /details %}}

## Distill teacher to student

We have already trained the teacher model, and we only need to initialize a `Distiller(student, teacher)` instance, `compile()` it with the desired losses, hyperparameters and optimizer, and distill the teacher to the student.

```python
# Initialize and compile distiller
distiller = Distiller(student=student, teacher=teacher)
distiller.compile(
    optimizer=keras.optimizers.Adam(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
    student_loss_fn=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    distillation_loss_fn=keras.losses.KLDivergence(),
    alpha=0.1,
    temperature=10,
)

# Distill teacher to student
distiller.fit(x_train, y_train, epochs=3)

# Evaluate student on test dataset
distiller.evaluate(x_test, y_test)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Epoch 1/3
 1875/1875 ━━━━━━━━━━━━━━━━━━━━ 8s 3ms/step - loss: 1.8752 - sparse_categorical_accuracy: 0.7357
Epoch 2/3
 1875/1875 ━━━━━━━━━━━━━━━━━━━━ 6s 3ms/step - loss: 0.0333 - sparse_categorical_accuracy: 0.9475
Epoch 3/3
 1875/1875 ━━━━━━━━━━━━━━━━━━━━ 6s 3ms/step - loss: 0.0223 - sparse_categorical_accuracy: 0.9621
 313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 4ms/step - loss: 0.0189 - sparse_categorical_accuracy: 0.9629

[0.017046602442860603, 0.969200074672699]
```

{{% /details %}}

## Train student from scratch for comparison

We can also train an equivalent student model from scratch without the teacher, in order to evaluate the performance gain obtained by knowledge distillation.

```python
# Train student as doen usually
student_scratch.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

# Train and evaluate student trained from scratch.
student_scratch.fit(x_train, y_train, epochs=3)
student_scratch.evaluate(x_test, y_test)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Epoch 1/3
 1875/1875 ━━━━━━━━━━━━━━━━━━━━ 4s 1ms/step - loss: 0.5111 - sparse_categorical_accuracy: 0.8460
Epoch 2/3
 1875/1875 ━━━━━━━━━━━━━━━━━━━━ 3s 1ms/step - loss: 0.1039 - sparse_categorical_accuracy: 0.9687
Epoch 3/3
 1875/1875 ━━━━━━━━━━━━━━━━━━━━ 3s 1ms/step - loss: 0.0748 - sparse_categorical_accuracy: 0.9780
 313/313 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - loss: 0.0744 - sparse_categorical_accuracy: 0.9737

[0.0629437193274498, 0.9778000712394714]
```

{{% /details %}}

If the teacher is trained for 5 full epochs and the student is distilled on this teacher for 3 full epochs, you should in this example experience a performance boost compared to training the same student model from scratch, and even compared to the teacher itself. You should expect the teacher to have accuracy around 97.6%, the student trained from scratch should be around 97.6%, and the distilled student should be around 98.1%. Remove or try out different seeds to use different weight initializations.
