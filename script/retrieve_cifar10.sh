#!/bin/bash
dest_dir=$1
uri=https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz

mkdir -p $dest_dir
wget -P $dest_dir $uri
tar xvf $dest_dir/cifar-10-binary.tar.gz -C $dest_dir
