scancel -u $USER

rm */.done
rm */.running
rm -r train_*
rm logs/*

python batch_train_affinities.py &

