# Deep-Q Network for Atari Breakout
*Note: Template code borrowed and adapted from Arthur Juliani (https://github.com/awjuliani/DeepRL-Agents). Thank you!*

### Description
This is a Deep-Q Network designed to play the Atari Breakout environment (Breakout-v0) in the OpenAI gym toolkit. It was able to achieve ~35 score after several hours of training on my i5-6600K CPU and GTX960 GPU.
This was an educational experiment and DQN has since (Jun 2016) been superseded by A3C as the preferred method for reinforcement learning (in online environments), see: [arXiv:1602.01783 [cs.LG]](https://arxiv.org/abs/1602.01783)

### Requirements
- Python 3
- Tensorflow
- CUDNN for GPU support
- Scipy, Scikit-image, and Numpy

### Features
- Double Q-Network enabled to separate Q-Value generation and Action Selection
- Dueling Q-Network enabled to separate Value and Advantage streams
- Experience replay & batch training to reduce non-stationarity
- Frame stacking to preserve state momentum information
