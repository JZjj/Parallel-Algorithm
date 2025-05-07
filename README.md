Parallelism in Deep Learning

A parallel and distributed framework for high-performance deep-learning

1. Data‐Parallel Loading: Implemented a multithreading-based data loader to overlap I/O and preprocessing for better efficiency. Achieved up to 1.5× speed-up in batch‐preparation time compared to a single‐process loader.

2. Forward Convolution Parallelism: Developed a tiled GPU kernel using shared‐memory and unrolling looping to accelerate the convolution forward pass. Measured up to 900× speed-up over the naïve CPU implementation, with numerical results matching the sequential baseline.

3. Gradient‐Aggregation Parallelism: Integrated an all-reduce–based scheme to sum and average per-GPU gradients synchronously, offering training speed-up through efficient collective communication.

4. Model‐Parallel Framework: Built a layer-wise partitioning strategy to distribute model shards across multiple devices. Achieved a 1.15× training-throughput improvement on large networks, while maintaining comparable accuracy to the single-GPU run.


