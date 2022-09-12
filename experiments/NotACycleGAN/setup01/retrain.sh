rm train.out
rm -r tensorboard
rm -r models
rm -r snapshots
raygun-train-cluster ./train_conf.json
