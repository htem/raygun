# Install:
Run the following -->
```bash
conda create -n raygun python=3.10 pytorch torchvision torchaudio cudatoolkit=11.3 affogato -c pytorch -c nvidia -c conda-forge 
conda activate raygun
pip install git+https://github.com/htem/raygun@full_refactor

```

# Example train:
```bash
raygun-train path/to/train_config.json
```