---
title: Distribution utilities
toc: true
weight: 4
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/distribution/distribution_lib.py#L829" >}}

### `set_distribution` function

```python
keras.distribution.set_distribution(value)
```

Set the distribution as the global distribution setting.

**Arguments**

- **value**: a `Distribution` instance.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/distribution/distribution_lib.py#L823" >}}

### `distribution` function

```python
keras.distribution.distribution()
```

Retrieve the current distribution from global context.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/distribution/distribution_lib.py#L26" >}}

### `list_devices` function

```python
keras.distribution.list_devices(device_type=None)
```

Return all the available devices based on the device type.

Note: in a distributed setting, global devices are returned.

**Arguments**

- **device_type**: string, one of `"cpu"`, `"gpu"` or `"tpu"`.
  Defaults to `"gpu"` or `"tpu"` if available when
  `device_type` is not provided. Otherwise
  will return the `"cpu"` devices.

Return:
List of devices that are available for distribute computation.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/distribution/distribution_lib.py#L44" >}}

### `initialize` function

```python
keras.distribution.initialize(
    job_addresses=None, num_processes=None, process_id=None
)
```

Initialize the distribution system for multi-host/process setting.

Calling `initialize` will prepare the backend for execution on multi-host
GPU or TPUs. It should be called before any computations.

Note that the parameters can also be injected via environment variables,
which can be better controlled by the launch script at startup time.
For certain backend that also rely on the environment variables to
configure, Keras will properly forward them.

**Arguments**

- **job_addresses**: string. Comma separated IP addresses for all the jobs
  that will form the whole computation cluster. Note that for JAX
  backend, only the address for job 0 (coodinator) is needed. For
  certain runtime like cloud TPU, this value can be `None`, and the
  backend will figure it out with the TPU environment variables. You
  can also config this value via environment variable
  `KERAS_DISTRIBUTION_JOB_ADDRESSES`.
- **num_processes**: int. The number of worker/processes that will form the
  whole computation cluster. For certain runtime like cloud TPU, this
  value can be `None`, and the backend will figure it out with the TPU
  environment variables. You can also configure this value via
  environment variable `KERAS_DISTRIBUTION_NUM_PROCESSES`.
- **process_id**: int. The ID number of the current worker/process. The value
  should be ranged from `0` to `num_processes - 1`. `0` will indicate
  the current worker/process is the master/coordinate job. You can
  also configure this value via environment variable
  `KERAS_DISTRIBUTION_PROCESS_ID`.

## Example

Suppose there are two GPU processes, and process 0 is running at
address `10.0.0.1:1234`, and process 1 is running at address
`10.0.0.2:2345`. To configure such cluster, you can run

- **On process 0**:

```python
keras.distribute.initialize(
  job_addresses="10.0.0.1:1234,10.0.0.2:2345",
  num_processes=2,
  process_id=0)
```

- **On process 1**:

```python
keras.distribute.initialize(
  job_addresses="10.0.0.1:1234,10.0.0.2:2345",
  num_processes=2,
  process_id=1)
```

- **or via the environment variables**:
- **On process 0**:

```python
os.environ[
  "KERAS_DISTRIBUTION_JOB_ADDRESSES"] = "10.0.0.1:1234,10.0.0.2:2345"
os.environ["KERAS_DISTRIBUTION_NUM_PROCESSES"] = "2"
os.environ["KERAS_DISTRIBUTION_PROCESS_ID"] = "0"
keras.distribute.initialize()
```

- **On process 1**:

```python
os.environ[
  "KERAS_DISTRIBUTION_JOB_ADDRESSES"] = "10.0.0.1:1234,10.0.0.2:2345"
os.environ["KERAS_DISTRIBUTION_NUM_PROCESSES"] = "2"
os.environ["KERAS_DISTRIBUTION_PROCESS_ID"] = "1"
keras.distribute.initialize()
```

Also note that for JAX backend, the `job_addresses` can be further
reduced to just the master/coordinator address, which is - **`10.0.0.1__:1234`**.
