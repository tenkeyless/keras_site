---
title: Experiment management utilities
toc: true
weight: 1
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/utils/config.py#L12" >}}

### `Config` class

```python
keras.utils.Config(**kwargs)
```

A Config is a dict-like container for named values.

It offers a few advantages over a plain dict:

- Setting and retrieving values via attribute setting / getting.
- Ability to freeze the config to ensure no accidental config modifications
  occur past a certain point in your program.
- Easy serialization of the whole config as JSON.

**Examples**

```python
config = Config("learning_rate"=0.1, "momentum"=0.9)
config.use_ema = True
config.ema_overwrite_frequency = 100
config["seed"] = 123
assert config.seed == 100
assert config["learning_rate"] == 0.1
```

A config behaves like a dict:

```python
config = Config("learning_rate"=0.1, "momentum"=0.9)
for k, v in config.items():
    print(f"{k}={v}")
print(f"keys: {list(config.keys())}")
print(f"values: {list(config.values())}")
```

In fact, it can be turned into one:

```python
config = Config("learning_rate"=0.1, "momentum"=0.9)
dict_config = config.as_dict()
```

You can easily serialize a config to JSON:

```python
config = Config("learning_rate"=0.1, "momentum"=0.9)
json_str = config.to_json()
```

You can also freeze a config to prevent further changes:

```python
config = Config()
config.optimizer = "adam"
config.seed = 123
config.freeze()
assert config.frozen
config.foo = "bar"  # This will raise an error.
```
