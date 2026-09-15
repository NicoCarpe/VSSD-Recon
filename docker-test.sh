docker build -t debug -f dockerfile .
docker run --gpus all -v $PWD:/output --rm debug