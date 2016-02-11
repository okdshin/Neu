#!/bin/bash
dest_dir=$1/mnist
uri_list="
	http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz \
	http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz \
	http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz \
	http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
"

mkdir -p $dest_dir
for uri in $uri_list; do 
	wget -P $dest_dir  $uri
done

for line in `find $dest_dir -name *ubyte.gz`; do
	gunzip $line
done
