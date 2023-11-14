[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/act3d-infinite-resolution-action-detection/robot-manipulation-on-rlbench)](https://paperswithcode.com/sota/robot-manipulation-on-rlbench?p=act3d-infinite-resolution-action-detection)


# Act3D & ChainedDiffuser


This repo contains a language-conditioned policy learner proposed in the following 2 papers:

> **[Act3D: 3D Feature Field Transformers for Multi-Task Robotic Manipulation](https://act3d.github.io/)**
> [Theophile Gervet*](https://theophilegervet.github.io/), [Zhou Xian*](https://zhou-xian.com/), [Nikolaos Gkanatsios](https://nickgkan.github.io/), [Katerina Fragkiadaki](https://www.cs.cmu.edu/~katef/)

and 

> **[ChainedDiffuser: Unifying Trajectory Diffusion and Keypose Prediction for Robotic Manipulation](https://chained-diffuser.github.io/)**
> [Zhou Xian*](https://zhou-xian.com/), [Nikolaos Gkanatsios*](https://nickgkan.github.io/), [Theophile Gervet*](https://theophilegervet.github.io/), [Tsung-Wei Ke](https://twke18.github.io/) [Katerina Fragkiadaki](https://www.cs.cmu.edu/~katef/)


![](imgs/tasks.png)

## Conda Environment Setup

We train on a remote cluster without the RLbench/PyRep libraries and evaluate locally (on a PC that supports graphics) with those libraries.

Environment setup on both remote and locally:
```
conda create -n chained_diffuser python=3.9
conda activate chained_diffuser;
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia;
pip install numpy pillow einops typed-argument-parser tqdm transformers absl-py matplotlib scipy tensorboard opencv-python diffusers blosc trimesh wandb open3d;
pip install git+https://github.com/openai/CLIP.git;
```

To install RLBench locally (make sure to edit the COPPELIASIM_ROOT to point to your path to PyRep!):
```
# Install PyRep
cd PyRep; 
wget https://www.coppeliarobotics.com/files/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz; 
tar -xf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz;
echo "export COPPELIASIM_ROOT=/your_path_to_PyRep/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04" >> ~/.bashrc; 
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\$COPPELIASIM_ROOT" >> ~/.bashrc;
echo "export QT_QPA_PLATFORM_PLUGIN_PATH=\$COPPELIASIM_ROOT" >> ~/.bashrc;
source ~/.bashrc;
pip install -r requirements.txt; pip install -e .; cd ..

# Install RLBench
cd RLBench; pip install -r requirements.txt; pip install -e .; cd ..;
sudo apt-get update; sudo apt-get install xorg libxcb-randr0-dev libxrender-dev libxkbcommon-dev libxkbcommon-x11-0 libavcodec-dev libavformat-dev libswscale-dev;
sudo nvidia-xconfig -a --virtual=1280x1024;
wget https://sourceforge.net/projects/virtualgl/files/2.5.2/virtualgl_2.5.2_amd64.deb/download -O virtualgl_2.5.2_amd64.deb --no-check-certificate;
sudo dpkg -i virtualgl*.deb; rm virtualgl*.deb;
sudo reboot  # Need to reboot for changes to take effect
```

## Dataset Generation

See `data_preprocessing` folder. Additionally download [instructions.pkl](https://drive.google.com/file/d/1ZGp18GBzzM9oP866vlTBg4DgfXn7BliL/view?usp=sharing).

## Training Act3D

`bash scripts/train_act3d.sh`. Make sure to change the dataset path to where the generated data is stored!

## Training a trajectory diffusion model

`bash scripts/train_trajectory.sh`. Make sure to change the dataset path to where the generated data is stored!

## Evaluation

`bash online_evaluation/eval.sh`. Make sure to edit the data/checkpoint paths!
