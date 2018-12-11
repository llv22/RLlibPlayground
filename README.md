# RLlibPlayground
First playground for RLlib for CS294-112 demo in Lecture 21

## 1. Preparation of RLlib installation
Reference: https://ray.readthedocs.io/en/master/rllib.html#installation 
```bash
pip install 'ray[rllib]'
# debugging purpose
pip install setproctitle
# bokeh 1.0.0 is incompatiable version
pip install bokeh==0.13.0
```
Version 0.6.0 for current experiment

## 2. Startup code
Reference: page 39 - 42 of http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-21.pdf 

## 3. Issues:

* bokeh can't visualize issue : children not iterate
* pg.PGAgent with multiagent-env can't work