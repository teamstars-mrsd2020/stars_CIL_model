#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2

export CARLA_ROOT=/home/stars/Code/carla_0.9.9
export PORT=2000

export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.8-py3.5-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.9-py3.7-linux-x86_64.egg


python team_stars/run.py

