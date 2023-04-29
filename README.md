# continuous-control
This repo is an implementation of the research paper "[Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)"(DDPG). 

# Demo
[![Demo](https://img.youtube.com/vi/Mt7O2pCBmrY/0.jpg)](https://youtu.be/Mt7O2pCBmrY "Demo")

# Files
* train.py: train the agent
* random_process.py: the random noise file(Ornstein-Uhlenbeck process here)
* ddpg.py: the agent(the actor and critic networks)
* display.py: load the trained model and see how the agent interact with the environment
(Here are the pretrained models on HalfCheetah-v4, LunarLanderContinuous-v2 and Pendulum-v1 from this code https://drive.google.com/drive/u/0/folders/1gGhH0ad7OT9_tuMAaJq36lVR6S-C1Ag6)

