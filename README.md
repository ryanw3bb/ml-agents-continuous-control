# ML-Agents Continuous Control
![gif](images/trained.gif)

## Environment
In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. The goal of the agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

To solve the environment the agents must get an average score of +30 (over 100 consecutive episodes, and over all agents).
- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 20 (potentially different) scores. We then take the average of these 20 scores. 
- This yields an **average score** for each episode (where the average is over all 20 agents).
- The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30. 

## Setup
1. Clone this repository.  

2. Make sure you have python 3.6 installed and a virtual environment activated, then install the required packages torch, numpy and unityagents. They can be installed using pip:
    ```
    python pip install torch 
    ```
3. Download the environment from one of the links below and extract in the repository directory:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)


## Running the Project
Browse to the project directory and run `train.py` to begin training. The trained weights are saved to the files `checkpoint_actor.pth` and `checkpoint_critic.pth` once the required score is reached.
```
python train.py
```
Run `play.py` to watch an episode where smart agents run using the trained data.