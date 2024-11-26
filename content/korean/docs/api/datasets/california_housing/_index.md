---
title: California Housing price regression dataset
toc: true
weight: 7
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/datasets/california_housing.py#L9" >}}

### `load_data` function

```python
keras.datasets.california_housing.load_data(
    version="large", path="california_housing.npz", test_split=0.2, seed=113
)
```

Loads the California Housing dataset.

This dataset was obtained from the [StatLib repository](https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html).

It's a continuous regression dataset with 20,640 samples with
8 features each.

The target variable is a scalar: the median house value
for California districts, in dollars.

The 8 input features are the following:

- MedInc: median income in block group
- HouseAge: median house age in block group
- AveRooms: average number of rooms per household
- AveBedrms: average number of bedrooms per household
- Population: block group population
- AveOccup: average number of household members
- Latitude: block group latitude
- Longitude: block group longitude

This dataset was derived from the 1990 U.S. census, using one row
per census block group. A block group is the smallest geographical
unit for which the U.S. Census Bureau publishes sample data
(a block group typically has a population of 600 to 3,000 people).

A household is a group of people residing within a home.
Since the average number of rooms and bedrooms in this dataset are
provided per household, these columns may take surprisingly large
values for block groups with few households and many empty houses,
such as vacation resorts.

**Arguments**

- **version**: `"small"` or `"large"`. The small version
  contains 600 samples, the large version contains
  20,640 samples. The purpose of the small version is
  to serve as an approximate replacement for the
  deprecated `boston_housing` dataset.
- **path**: path where to cache the dataset locally
  (relative to `~/.keras/datasets`).
- **test_split**: fraction of the data to reserve as test set.
- **seed**: Random seed for shuffling the data
  before computing the test split.

**Returns**

- **Tuple of Numpy arrays**: `(x_train, y_train), (x_test, y_test)`.

**`x_train`, `x_test`**: numpy arrays with shape `(num_samples, 8)`
containing either the training samples (for `x_train`),
or test samples (for `y_train`).

**`y_train`, `y_test`**: numpy arrays of shape `(num_samples,)`
containing the target scalars. The targets are float scalars
typically between 25,000 and 500,000 that represent
the home prices in dollars.
