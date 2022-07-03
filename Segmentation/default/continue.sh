scancel -u $USER

rm */.done
rm */.running

python batch_train_affinities.py &

