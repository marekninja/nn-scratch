## Feed Forward NN from scratch

* without any external libraries
* intended for learning of 
  * the internals of nn
  * matrix operations optimization


## Steps to run:

1. Assuming folder structure:

```
\.
\data\fashion_mnist_test_labels.csv
\data\fashion_mnist_train_labels.csv
\data\fashion_mnist_train_vectors.csv
\data\fashion_mnist_train_vectors.csv
\src\_contents_of_this_repository_
```

2. You can simply run the following commands in bash(script):

```bash
#!/bin/bash

echo "#################"
echo "    COMPILING    "
echo "#################"

g++ -std=c++11 -O3 ./src/*.cpp -pthread -lpthread -o ./src/network

echo "#################"
echo "     RUNNING     "
echo "#################"
cd ./src
./network
```