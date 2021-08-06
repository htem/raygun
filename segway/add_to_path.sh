
segway_dir=`realpath segway`

# check if path is correct
if [[ $PYTHONPATH != *${segway_dir}* ]]; then
    echo INFO: PYTHONPATH env does not have segway... adding it
    export PYTHONPATH=$PYTHONPATH:${segway_dir}
fi

