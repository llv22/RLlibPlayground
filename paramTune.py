# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 0.8.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.6.7
# ---

import ray
import ray.tune as tune

ray.init()

# https://ray.readthedocs.io/en/latest/rllib-algorithms.html?highlight=config%20sgd_stepsize
tune.run_experiments({
    "my_experiment": {
        "run": "PPO",
        "env": "CartPole-v0",
        "stop": {"episode_reward_mean": 200},
        "config": {
            "num_gpus": 0,
            "num_workers": 1,
            # Stepsize of SGD
            "lr": tune.grid_search([0.01, 0.001, 0.0001]),
#             "sgd_stepsize": tune.grid_search([0.01, 0.001, 0.0001]),
        },
    },
})


