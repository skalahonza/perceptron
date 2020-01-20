# Perceptron
Perceptron CPU vs GPU implementation.
Project website: https://cent.felk.cvut.cz/courses/GPU/archives/2019-2020/W/skalaja7/

     CUDA Perceptron
     Usage: GPU.exe [OPTIONS]

     Options:
     -h,--help                   Print this help message and exit
     -t,--train TEXT REQUIRED    Training dataset, CSV file with data and expected value.
     -e,--eval TEXT REQUIRED     Dataset for evaluation, CSV file with data with expected output.
     -i,--iterations INT=5       Number of iterations for training.
     -l,--lrate FLOAT=0.1        Learning rate.
     -v,--verbose                Specify for verbose output
