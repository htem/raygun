# Install:
Run the following -->
```bash
conda create -n raygun python=3.10 pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -c nvidia
conda activate raygun
pip install git+https://github.com/htem/raygun@full_refactor

```

# Example train:
```bash
raygun-train path/to/train_config.json
```