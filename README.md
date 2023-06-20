# Introduction
Implementation of the paper Transferable Virtual-Physical Environmental Alignment with Redirected Walking (TRAVEL)

# Acknowledgment

This repo is a basic implementation of the paper [Transferable Virtual-Physical Environmental Alignment with Redirected Walking](https://ieeexplore.ieee.org/document/9961901/). Limited by ability and time, I cannot guarantee that the code is completely consistent with the origin paper, only the basic framework can be provided.

# Dataset
* Generate the dataset by ```dataset_gen_mutiTarget.py```
* ```python dataset_gen_mutiTarget.py --mode eval/train```

# Training
* Update your env configuaration like [w, h] of your physical space in envs/envs_general.py
* Fine-tune your hyper-parameters in ```config/train.txt``` and ```config/test.txt```
* Training a new model  
    ```bash
    python main.py --config config/train.txt
    ```
# Evaluation 
* ```bash
  python evaluation.py --config args/eval.txt
# Customize your space and reward function
* Decide your space configuaration in envs/envs_general.py/configure_space(), you could change the shape, width, height and obstacle
* Customize your reward function in envs/envs_genearl.py/get_reward()
