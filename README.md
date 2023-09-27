### Spiking Control
This work is developed by Justus Huebotter in 2022 and 2023 as part of the <a href="https://spikeference.eu/">SPIKEFERENCE project</a>, co-founded by the Human Brain Project (HBP) Specific Grant Agreement 3 (ID: 945539) and the Donders Institute for Brain, Cognition and Behaviour. 

In this project, we revisit policy optimization for low-level continuous control tasks with spiking neural networks. In particular, we use prediction errors to learn the dynamics of the system in a recurrent *transition model* (see figure left). We show that we can then use this model to  drive the learning of an amortized *policy model* (see figure right) for goal reaching by imagining state trajectory rollouts, even when interaction with the real environment is limited.

<img src="figures/models.png" width="600" />

The exact method used in this code base is explained in more detail in:

J. Huebotter, S. Thill, M. van Gerven, P. Lanillos (2022): **Learning Policies for Continuous Control via Transition Models**, *3rd International Workshop on Active Inference* 

This publication is also available [here](https://arxiv.org/abs/2209.08033).

### Using the code

To use the code please clone this git via:

`git clone https://github.com/jhuebotter/spiking_control.git`

To install the required packages create a new local environment and run:

`pip install -r requirements.txt`

To enable wandb logging you will have to sign up at https://wandb.ai and call

`wandb login`

After this, the code should be executed by running :

`python train.py`

If desired, the parameters for the experiments can be changed in the `config.yaml` file. 
There are two environments currently supported: `plane` and `reacher2`. 
Please see below for example results for both environments with either static or moving targets.

### Example Results

#### Continuous control in a planar linear environment

<img src="figures/plane_static.gif" width="200" height="200" /> <img src="figures/plane_move.gif" width="200" height="200" /> 

#### Continuous control of a planar robot arm

<img src="figures/reacher_static.gif" width="200" height="200" /> <img src="figures/reacher_move.gif" width="200" height="200" />

The auto-regressive prediction model learns to accurately forecast the state trajectory based on control inputs:

<img src="figures/reacher_predict.gif" width="250" height="400" />