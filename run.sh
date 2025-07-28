#!/bin/bash

# Navigate to the exp directory (adjust path as needed)
cd ./exp/20250714

echo "Running experiment 1..."
cd 1
python encoder-proto-kmeans_attention.py
cd ..

echo "Running experiment 2..."
cd 2
python encoder-proto-kmeans_attention.py
cd ..

echo "Running experiment 3..."
cd 3
python encoder-proto-kmeans_attention.py
cd ..

echo "Running experiment 4..."
cd 4
python encoder-proto-kmeans_attention.py
cd ..

echo "Running experiment 5..."
cd 5
python encoder-proto-kmeans_attention.py
cd ..

# echo "Running experiment 6..."
# cd 6
# python encoder-proto-kmeans_attention.py
# cd ..

# echo "Running experiment 7..."
# cd 7
# python encoder-proto-kmeans_attention.py
# cd ..

# echo "Running experiment 8..."
# cd 8
# python encoder-proto-kmeans_attention.py
# cd ..

# echo "Running experiment 9..."
# cd 9
# python encoder-proto-kmeans_attention.py
# cd ..

# echo "Running experiment 10..."
# cd 10
# python encoder-proto-kmeans_attention.py
# cd ..


echo "All experiments completed!"
