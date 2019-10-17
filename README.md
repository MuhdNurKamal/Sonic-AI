# Sonic-AI
## Prerequisite:
- Ubuntu
- Python 3
- Make virtual environment
```
python3 -m venv venv
```


## Setup (Linux)

1.Activate virtual environment
```
. venv/bin/activate
```
2.Install baselines dependencies
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
4.Install baselines (Current baselines in pip is outdated)
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

## Training model
```
python train.py
```


## Play sonic with trained model
```
python play.py
```
