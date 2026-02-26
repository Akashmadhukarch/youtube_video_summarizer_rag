⚡ Important Note: Use GPU Instead of CPU

Ensure the project runs on GPU (CUDA) instead of CPU.

Running on CPU may significantly increase execution time.

GPU usage improves training speed and inference performance.

Large models (LLMs, Deep Learning models) perform much faster on GPU.

Verify GPU availability before running the project.

Use torch.cuda.is_available() (PyTorch) or tf.config.list_physical_devices('GPU') (TensorFlow) to confirm GPU access.

Monitor GPU usage using the nvidia-smi command.

Install the correct CUDA-enabled versions of required libraries.
