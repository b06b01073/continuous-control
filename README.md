# continuous-control
This repo is an implementation of the research paper "[Deep Deterministic Policy Gradient](https://arxiv.org/abs/1509.02971)". 


# Files
* train.py: train the agent
* random_process.py: the random noise file(you can change the noise sampled from Ornstein-Uhlenbeck process here)
* ddpg.py: the agent(you can change the layernorm in the actor and critic networks)
* display.py: you can load the trained model and see how the agent interact with the environment
(Here are the pretrained models on LunarLanderContinuous-v2 and Pendulum-v1 from this code https://drive.google.com/drive/u/0/folders/1gGhH0ad7OT9_tuMAaJq36lVR6S-C1Ag6)

