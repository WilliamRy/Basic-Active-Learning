# Active Learning

## Introduction

This is a python module for experimenting with different active learning
algorithms. There are a few key components to running active learning
experiments:

*   Main experiment script is
    [`run_experiment.py`](run_experiment.py)
    with many flags for different run options.

*   Supported datasets can be downloaded to a specified directory by running
    [`utils/create_data.py`](utils/create_data.py).

*   Supported active learning methods are in
    [`sampling_methods`](sampling_methods/).

Below I will go into each component in more detail.

### Adding new active learning methods

Implement either a base sampler that inherits from
[`SamplingMethod`](sampling_methods/sampling_def.py)
or a meta-sampler that calls base samplers which inherits from
[`WrapperSamplingMethod`](sampling_methods/wrapper_sampler_def.py).

The only method that must be implemented by any sampler is `select_batch_`,
which can have arbitrary named arguments. The only restriction is that the name
for the same input must be consistent across all the samplers (i.e. the indices
for already selected examples all have the same name across samplers). Adding a
new named argument that hasn't been used in other sampling methods will require
feeding that into the `select_batch` call in
[`run_experiment.py`](run_experiment.py).

After implementing your sampler, be sure to add it to
[`constants.py`](sampling_methods/constants.py)
so that it can be called from
[`run_experiment.py`](run_experiment.py).

## Available models

All available models are in the `get_model` method of
[`utils/utils.py`](utils/utils.py).

Supported methods:

*   Logistc Regression: scikit method with grid search wrapper for
    regularization parameter.

### Adding new models

New models must follow the scikit learn api and implement the following methods

*   `fit(X, y[, sample_weight])`: fit the model to the input features and
    target.

*   `predict(X)`: predict the value of the input features.

*   `score(X, y)`: returns target metric given test features and test targets.

*   `decision_function(X)` (optional): return class probabilities, distance to
    decision boundaries, or other metric that can be used by margin sampler as a
    measure of uncertainty.




