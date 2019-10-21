# Sonic-AI
## Prerequisite:
- Ubuntu
- Python 3
- Make virtual environment
```
python3 -m venv venv
```


## Setup (Linux)

1. Activate virtual environment
```
. venv/bin/activate
```
2. Install baselines dependencies
```
sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev
```
3. Install Tensorflow
- Choice A: For CPU (slower)
```
pip install tensorflow==1.15.0rc2
```
- Choice B: For Nvidia Cuda GPU (faster)
```
pip install tensorflow-gpu==1.15.0rc2
```
4. Install baselines (Current baselines in pip is outdated)
```
pip install git+https://github.com/openai/baselines.git
```
5. Install Other dependencies
```
pip install -r requirements.txt
```
6. Import sonic roms
```
python -m retro.import ./Roms
```
7. Install retro-contest
```
git clone --recursive https://github.com/openai/retro-contest.git
pip install -e "retro-contest/support[docker,rest]"
```
## Training model
```
python train.py
```


## Monitoring
Monitoring is done through tensorboard.
To run TensorBoard run the command below, make sure venv is activated

```
tensorboard --logdir ./tb_log
```
This will run tensorboard, and tensorboard will search for Events from the specified directory


## Play sonic with trained model
Replace saved_model_filename with actual name of the saved model zip file. E.g. sonic_stable_dqn.zip 
```
python play.py saved_model_filename

```
```
python slow_play.py saved_model_filename
```

## Hall of Fame
Models that has successfully completed a level are stored inside saved_models
