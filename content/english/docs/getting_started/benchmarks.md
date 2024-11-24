---
slug: benchmarks
title: Keras 3 benchmarks
toc: true
weight: 2
type: docs
---

{{< keras/original checkedAt="2024-11-18" >}}

We benchmark the three backends of Keras 3 ([TensorFlow](https://tensorflow.org/), [JAX](https://jax.readthedocs.io/en/latest/), [PyTorch](https://pytorch.org/)) alongside Keras 2 with TensorFlow. Find code and setup details for reproducing our results [here](https://github.com/haifeng-jin/keras-benchmarks/tree/v0.0.5).

## Models

We chose a set of popular computer vision and natural language processing models for both generative and non-generative AI tasks. See the table below for our selections.

**Table 1**: Models used in benchmarking.

|     | Non-Generative      | Generative             |
| --- | ------------------- | ---------------------- |
| CV  | SegmentAnything[^1] | StableDiffusion[^2]    |
| NLP | BERT[^3]            | Gemma[^4], Mistral[^5] |

We are not measuring the best possible performance achievable by each framework, but the out-of-the-box performance of common user workflows. With this goal in mind, we leveraged pre-existing implementations from KerasCV and KerasHub for the Keras versions of the models.

## Hardware

All benchmarks are done with a single NVIDIA A100 GPU with 40GB of GPU memory on a Google Cloud Compute Engine of machine type `a2-highgpu-1g` with 12 vCPUs and 85GB host memory.

## Results

Table 2 displays benchmarking results in milliseconds per step. Each step involves training or predicting on a single data batch. Results are averaged over 100 steps, excluding the first, which includes model creation and compilation overhead.

For fair comparison, we use the same batch size across frameworks if it is the same model and task (fit or predict). However, for different models and tasks, due to their different sizes and architectures, we use different batch sizes to avoid either running out of memory (too large) or under GPU utilization (too small).

For large language models (Gemma and Mistral), we also used the same batch size since they are the same model type with similar number of parameters (7B). We also benchmarked text generation with batch size equal to 1 since it is widely requested by the users. We used `bfloat16` precision for their training and inferencing, and LoRA[^6] for their training (fine-tuning).

To measure out-of-the-box performance, we try to use all default settings. For example, use high-level APIs (e.g. Use Keras `model.fit()`) with as little configuration as possible.

Note that this is quite different from measuring an optimized implementation for a particular hardware/framework/model combination. Please refer to [MLPerf](https://mlcommons.org/benchmarks/) for the best optimized results for different frameworks.

**Table 2**: Benchmarking results. The speed is measured in ms/step. Lower is better.

|                                | Batch size | Keras 2 (TensorFlow) | Keras 3 (TensorFlow) | Keras 3 (JAX) | Keras 3 (PyTorch) (eager) | Keras 3 (best) |
| ------------------------------ | ---------- | -------------------- | -------------------- | ------------- | ------------------------- | -------------- |
| **SegmentAnything (fit)**      | 1          | 386.93               | **355.25**           | 361.69        | 1,388.87                  | **355.25**     |
| **SegmentAnything (predict)**  | 4          | 1,859.27             | 438.50               | **376.34**    | 1,720.96                  | **376.34**     |
| **Stable Diffusion (fit)**     | 8          | 1,023.21             | 392.24               | **391.21**    | 823.44                    | **391.21**     |
| **Stable Diffusion (predict)** | 13         | 649.71               | **616.04**           | 627.27        | 1,337.17                  | **616.04**     |
| **BERT (fit)**                 | 32         | 486.00               | **214.49**           | 222.37        | 808.68                    | **214.49**     |
| **BERT (predict)**             | 256        | 470.12               | 466.01               | **418.72**    | 1,865.98                  | **418.72**     |
| **Gemma (fit)**                | 8          | NA                   | 232.52               | 273.67        | 525.15                    | **232.52**     |
| **Gemma (generate)**           | 32         | NA                   | 1,134.91             | **1,128.21**  | 7,952.67\*                | **1,128.21**   |
| **Gemma (generate)**           | 1          | NA                   | 758.57               | **703.46**    | 7,649.40\*                | **703.46**     |
| **Mistral (fit)**              | 8          | NA                   | **185.92**           | 213.22        | 452.12                    | **185.92**     |
| **Mistral (generate)**         | 32         | NA                   | 966.06               | **957.25**    | 10,932.59\*               | **957.25**     |
| **Mistral (generate)**         | 1          | NA                   | 743.28               | **679.30**    | 11,054.67\*               | **679.30**     |

\* _LLM inference with the PyTorch backend is abnormally slow at this time because KerasHub uses static sequence padding, unlike HuggingFace. This will be addressed soon._

## Discussion

### Key Finding 1: There is no "best" backend

Each of the three backends of Keras offers unique strengths. Crucially, from a performance standpoint, there's no single backend that consistently outpaces the others. The fastest backend often depends on your specific model architecture.

This underscores the value of framework optionality when chasing optimal performance. Keras 3 empowers you to seamlessly switch backends, ensuring you find the ideal match for your model.

### Key Finding 2: Keras 3 is faster than Keras 2

We also calculated the throughput (steps/ms) increase of Keras 3 (using its best-performing backend) over Keras 2 with TensorFlow from Table 1. Results are shown in the following figure.

![Figrue 2](/images/getting_started/benchmarks/jPncf0F.png "Figure 1: Keras 3 speedup over Keras 2 measured in throughput (steps/ms)")

Keras 3 consistently outperformed Keras 2 across all benchmarked models, with substantial speed increases in many cases. SegmentAnything inference saw a remarkable 380% boost, StableDiffusion training throughput increased by over 150%, and BERT training throughput rose by over 100%.

Importantly, you would still see a performance boost even if you simply upgrade to Keras 3 and continue using the TensorFlow backend. This is mainly because Keras 2 uses more TensorFlow fused ops directly, which may be sub-optimal for XLA compilation in certain use cases.

## Conclusions

Framework performance depends heavily on the specific model. Keras 3 empowers you to select the fastest framework for your task – an option almost always to outperform both Keras 2.

[^1]: Kirillov, Alexander, et al. "Segment anything." ICCV (2023).
[^2]: Rombach, Robin, et al. "High-resolution image synthesis with latent diffusion models." CVPR (2022).
[^3]: Kenton, Jacob, et al. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." NAACL (2019).
[^4]: Banks, Jeanine, et al. "Gemma: Introducing new state-of-the-art open models." The Keyword, Google (2024).
[^5]: Jiang, Albert Q., et al. "Mistral 7B." arXiv preprint arXiv:2310.06825 (2023).
[^6]: Hu, Edward J., et al. "Lora: Low-rank adaptation of large language models." ICLR (2022).
