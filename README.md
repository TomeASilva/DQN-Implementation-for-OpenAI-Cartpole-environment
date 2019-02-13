# DQN-Implementation-for-OpenAI-Cartpole-environment

This is an implementation of the [DQN algorithm](https://deepmind.com/research/dqn) introduced by the DeepMind team in 2015 created to 
interact with [OpenAI CartPole environment](https://github.com/openai/gym), which is a set of standard environments 
used to evaluate the performance of Reinforcement Learning Algorithms.

The implementation is in script.py  file

#### Requirements
+ tensorflow ver. 1.12.0
+ python ver 3.x
+ numpy

## Implementation characteristics:

+ script.py can be run directly
+ **Model Architecture**

![architecture](https://github.com/TomeASilva/DQN-Implementation-for-OpenAI-Cartpole-environment/blob/master/support_images/Tensor_Flow_graph.png "Model Architecture")


+ Tensorboard summaries are provided for:
  - Loss Function
  - ANN parameters
  - Episode Rewards
  - Episode Predicted Rewards
  
  
 + Hyperparameters can be changed by altering the args dictionary:
 
 ```python 
  args = {"num_layers": 3,
            "layers_sizes": [32, 64, 2],
            "activations": ["relu", "relu", "linear"],
            "gamma": 0.9,
            "state_space_size": env.observation_space.shape[0],
            "action_space_size": env.action_space.n,
            "learning_rate": 0.001,
            "epsilon_start": 1,
            "minibatch_size": 64,
            "C_step_target_up": 1,
            "max_steps_episode": 200,
            "max_number_episodes": 50,
            "soft_update_tau": 8e-2,
            "epsilon_stop": 0.01,
            "epsilon_decay": 0.999}
  
  
  ```
 
 




