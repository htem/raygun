# Install:
Run the following -->
```bash
conda create -n raygun python=3.9 tensorflow pytorch torchvision torchaudio cudatoolkit=11.3 affogato -c pytorch -c nvidia -c conda-forge 
conda activate raygun
pip install git+https://github.com/htem/raygun
```

# Example train:
```bash
raygun-train path/to/train_config.json
```
