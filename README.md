### This repo is the Udacity Deep Reinforcement Learning course Continue Control Project
> The README should be designed for a general audience that may not be familiar with the Nanodegree program; you should describe the environment that you solved, along with how to install the requirements before running the code in your repository.

#### Steps
1. install python
2. ```pip install pytorch``` reference [pytorch website](https://pytorch.org/)
3. ```pip install unityagents```
4. download **reacher env** from [Reacher_Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
5. Run ```components/env_tst.py``` to test the environment work all right.

#### Environments
* XUbuntu 18.04
* CUDA 10.0
* cudnn 7.4.1
* Python 3.6
* Pytorch 1.0
* yacs v0.1.5
* OpenAI baseline https://github.com/openai/baselines


#### How to use the code
1. Install python
2. ```pip install pytorch``` reference [pytorch website](https://pytorch.org/)
3. ```pip install unityagents```
4. download **banana env** from [Banana_Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip) and [Banana_Linux_Pixels](https://classroom.udacity.com/nanodegrees/nd893/parts/6b0c03a7-6667-4fcf-a9ed-dd41a2f76485/modules/4eeb16ab-5ac5-47bf-974d-12784e9730d7/lessons/69bd42c6-b70e-4866-9764-9bfa8c03cdea/concepts/80164380-a9ad-460d-bee7-59fe2d776036)
5. Run ```Navigation_Env.py``` and ```Navigation_Env_Pixels.py``` to test the environment work all right.
6. Run ```train.py``` to train the agent. This module is for the vector state space.
7. Run ```test.py``` to test the agent performance.
8. You default config file is  ```configs/agent_dqn_config.py``` and the config file with good documentation. You can modify the default parameter value to retrain the agent.
9. Run ```train_pix.py``` to train the agent. This module is for the pixel state space.



#### TO-DO-LIST
* ~~DQN (The state space is a 37 dim vector).~~ You can find the training recode in **Results** part ```dqn_vec```. The saved model ```checkpoint_dqn_vec.pth``` is in folder results. 
* DQN (The state space is 84*84 image). This agent is still in training.
* Double DQN vector state space
* Double DQN pixel state space
* Dueling DQN vector state space
* Dueling DQN pixel state space


#### Results
*  **dqn_vec**

    ```
    Episode 100	Average Score: 0.76
    Episode 200	Average Score: 3.49
    Episode 300	Average Score: 6.43
    Episode 400	Average Score: 9.61
    Episode 500	Average Score: 12.18
    Episode 559	 Score: 16.00
    Environment solved in 459 episodes!	Average Score: 13.00
    ```
    ![sedv](./results/scores_episode_dqn_vec.png)



#### References
1. [AdaBound](https://github.com/Luolc/AdaBound)
2. [DeepRL](https://github.com/ShangtongZhang/DeepRL)