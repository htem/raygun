[![test build](https://github.com/htem/raygun/actions/workflows/test-build.yml/badge.svg)](https://github.com/htem/raygun/actions/workflows/test-build.yml)

# Install:
Run the following -->
```bash
conda create -n raygun python=3.9 tensorflow pytorch torchvision torchaudio cudatoolkit=11.3 affogato -c pytorch -c nvidia -c conda-forge 
conda activate raygun
pip install git+https://github.com/htem/raygun
```

Should you run into gcc / boost errors when conda/pip installing raygun, try installing ```libboost``` first:
```bash
sudo apt-get update
sudo apt-get install libboost-all-dev
```

# Example train:
```bash
raygun-train path/to/train_config.json
```
