# Dagger

Dataset Aggregation (DAGGER) is a type of imitation learning algorithm that improves the policy by running it on the task
and getting new samples with labels from experts, appends it to the existing training dataset and learns a new policy based
on this data. This process repeats until the desired performance is achieved.


## Requirements

 - MuJuCo (1.3.1)
 - Tensorflow >= 1.2.0
 - OpenAI Gym


## Expert Policies
In `experts/`, the provided expert policies are:
* Ant-v1.pkl
* HalfCheetah-v1.pkl
* Hopper-v1.pkl
* Humanoid-v1.pkl
* Reacher-v1.pkl
* Walker2d-v1.pkl


## Disclaimer
The basic framework is taken from CS294 Fall 2017 homework1 assignment.
