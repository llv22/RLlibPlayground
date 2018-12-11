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

# ## Multi-Agent RL
#
# Don't find **MultiAgentTrafficEnv**, **car_obs_space**, **car_act_space** and **tl_obs_space**, **tl_act_space**
# **Status** : Not working now

import ray
# check in https://github.com/ray-project/ray/blob/master/python/ray/rllib/test/test_nested_spaces.py
from ray.rllib.agents import pg
from ray.rllib.agents.pg.pg_policy_graph import PGPolicyGraph
from ray.tune.logger import pretty_print

ray.init()

# +
# Example: using a multi-agent env
env = MultiAgentTrafficEnv(num_cars=20, num_traffic_lights=5)

# Observations are a dict mapping agent names to their obs. Not all agents
# may be present in the dict in each time step.
print(env.reset())
# {
#     "car_1": [[...]],
#     "car_2": [[...]],
#     "traffic_light_1": [[...]],
# }

# Actions should be provided for each agent that returned an observation.
new_obs, rewards, dones, infos = env.step(actions={"car_1": ..., "car_2": ...})

# Similarly, new_obs, rewards, dones, etc. also become dicts
print(rewards)
# {"car_1": 3, "car_2": -1, "traffic_light_1": 0}

# Individual agents can early exit; env is done when "__all__" = True
print(dones)
# {"car_2": True, "__all__": False}
# -

# https://rise.cs.berkeley.edu/blog/scaling-multi-agent-rl-with-rllib/
# https://ray.readthedocs.io/en/latest/rllib-env.html?highlight=%27PGAgent%27
# https://github.com/ray-project/ray/blob/master/python/ray/rllib/examples/multiagent_cartpole.py
trainer = pg.PGAgent(env="my_multiagent_env", config={
    "multiagent": {
        "policy_graphs": {
            "car1": (PGPolicyGraph, car_obs_space, car_act_space, {"gamma": 0.85}),
            "car2": (PGPolicyGraph, car_obs_space, car_act_space, {"gamma": 0.99}),
            "traffic_light": (PGPolicyGraph, tl_obs_space, tl_act_space, {}),
        },
        "policy_mapping_fn": {
            lambda agent_id:
                "traffic_light"  # Traffic lights are always controlled by this policy
                if agent_id.startswith("traffic_light_")
                else random.choice(["car1", "car2"])  # Randomly choose from car policies
        },
    },
})

while True:
    print(trainer.train())
