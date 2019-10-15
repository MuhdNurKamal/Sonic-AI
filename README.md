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
3.Install baselines (Current baselines in pip is outdated)
```
pip install git+https://github.com/openai/baselines.git
```
4. Install Other dependencies
```
pip install -r requirements.txt
```


## Training model
```
python train.py
```


## Play sonic with trained model
```
python play.py
```
