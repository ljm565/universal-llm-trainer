# Bug Reports

&nbsp;

&nbsp;

## FSDP Errors
### Case 1. FSDP mimium parameter setting
```
terminate called after throwing an instance of 'terminate called after throwing an instance of 'c10::Errorc10::Error'
'
  what():  CUDA error: an illegal memory access was encountered
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

Exception raised from c10_cuda_check_implementation at ../c10/cuda/CUDAException.cpp:43 (most recent call first):
frame #0: c10::Error::Error(c10::SourceLocation, std::string) + 0x96 (0x7fd2c04b9446 in /opt/conda/lib/python3.11/site-packages/torch/lib/libc10.so)
frame #1: c10::detail::torchCheckFail(char const*, char const*, unsigned int, std::string const&) + 0x64 (0x7fd2c04636e4 in /opt/conda/lib/python3.11/site-packages/torch/lib/libc10.so)
frame #2: c10::cuda::c10_cuda_check_implementation(int, char const*, char const*, int, bool) + 0x118 (0x7fd2c05a5a18 in /opt/conda/lib/python3.11/site-packages/torch/lib/libc10_cuda.so)
frame #3: <unknown function> + 0x1f92e (0x7fd2c056c92e in /opt/conda/lib/python3.11/site-packages/torch/lib/libc10_cuda.so)
frame #4: <unknown function> + 0x20a57 (0x7fd2c056da57 in /opt/conda/lib/python3.11/site-packages/torch/lib/libc10_cuda.so)
frame #5: <unknown function> + 0x20c5f (0x7fd2c056dc5f in /opt/conda/lib/python3.11/site-packages/torch/lib/libc10_cuda.so)
frame #6: <unknown function> + 0x5fcdd0 (0x7fd2bf38bdd0 in /opt/conda/lib/python3.11/site-packages/torch/lib/libtorch_python.so)
frame #7: <unknown function> + 0x6f69f (0x7fd2c049a69f in /opt/conda/lib/python3.11/site-packages/torch/lib/libc10.so)
frame #8: c10::TensorImpl::~TensorImpl() + 0x21b (0x7fd2c049337b in /opt/conda/lib/python3.11/site-packages/torch/lib/libc10.so)
frame #9: c10::TensorImpl::~TensorImpl() + 0x9 (0x7fd2c0493529 in /opt/conda/lib/python3.11/site-packages/torch/lib/libc10.so)
frame #10: std::vector<at::Tensor, std::allocator<at::Tensor> >::~vector() + 0x88 (0x7fd2bf38dd08 in /opt/conda/lib/python3.11/site-packages/torch/lib/libtorch_python.so)
frame #11: torch::autograd::Engine::thread_main(std::shared_ptr<torch::autograd::GraphTask> const&) + 0x129 (0x7fd2af7217b9 in /opt/conda/lib/python3.11/site-packages/torch/lib/libtorch_cpu.so)
frame #12: torch::autograd::Engine::thread_init(int, std::shared_ptr<torch::autograd::ReadyQueue> const&, bool) + 0x13f (0x7fd2af718b8f in /opt/conda/lib/python3.11/site-packages/torch/lib/libtorch_cpu.so)
frame #13: torch::autograd::python::PythonEngine::thread_init(int, std::shared_ptr<torch::autograd::ReadyQueue> const&, bool) + 0x5c (0x7fd2bf62c27c in /opt/conda/lib/python3.11/site-packages/torch/lib/libtorch_python.so)
frame #14: <unknown function> + 0x145c0 (0x7fd2c08d55c0 in /opt/conda/lib/python3.11/site-packages/torch/lib/libtorch.so)
frame #15: <unknown function> + 0x94ac3 (0x7fd2c16faac3 in /usr/lib/x86_64-linux-gnu/libc.so.6)
frame #16: clone + 0x44 (0x7fd2c178ba04 in /usr/lib/x86_64-linux-gnu/libc.so.6)

[rank1]:[E409 05:10:01.930967482 ProcessGroupNCCL.cpp:1595] [PG ID 0 PG GUID 0(default_pg) Rank 1] Process group watchdog thread terminated with exception: CUDA error: an illegal memory access was encountered
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

Exception raised from c10_cuda_check_implementation at ../c10/cuda/CUDAException.cpp:43 (most recent call first):
frame #0: c10::Error::Error(c10::SourceLocation, std::string) + 0x96 (0x7fd43a36c446 in /opt/conda/lib/python3.11/site-packages/torch/lib/libc10.so)
frame #1: c10::detail::torchCheckFail(char const*, char const*, unsigned int, std::string const&) + 0x64 (0x7fd43a3166e4 in /opt/conda/lib/python3.11/site-packages/torch/lib/libc10.so)
frame #2: c10::cuda::c10_cuda_check_implementation(int, char const*, char const*, int, bool) + 0x118 (0x7fd43a71ca18 in /opt/conda/lib/python3.11/site-packages/torch/lib/libc10_cuda.so)
frame #3: c10d::ProcessGroupNCCL::WorkNCCL::finishedGPUExecutionInternal() const + 0x56 (0x7fd3f0214726 in /opt/conda/lib/python3.11/site-packages/torch/lib/libtorch_cuda.so)
frame #4: c10d::ProcessGroupNCCL::WorkNCCL::isCompleted() + 0xa0 (0x7fd3f02193f0 in /opt/conda/lib/python3.11/site-packages/torch/lib/libtorch_cuda.so)
frame #5: c10d::ProcessGroupNCCL::watchdogHandler() + 0x1da (0x7fd3f0220b5a in /opt/conda/lib/python3.11/site-packages/torch/lib/libtorch_cuda.so)
frame #6: c10d::ProcessGroupNCCL::ncclCommWatchdog() + 0x14d (0x7fd3f022261d in /opt/conda/lib/python3.11/site-packages/torch/lib/libtorch_cuda.so)
frame #7: <unknown function> + 0x145c0 (0x7fd43a7975c0 in /opt/conda/lib/python3.11/site-packages/torch/lib/libtorch.so)
frame #8: <unknown function> + 0x94ac3 (0x7fd43b5bcac3 in /usr/lib/x86_64-linux-gnu/libc.so.6)
frame #9: clone + 0x44 (0x7fd43b64da04 in /usr/lib/x86_64-linux-gnu/libc.so.6)

terminate called recursively
Traceback (most recent call last):
  File "/universal-llm-trainer/src/run/train.py", line 111, in <module>
    main(args)
  File "/universal-llm-trainer/src/run/train.py", line 42, in main
    torch.multiprocessing.spawn(multi_gpu_train, nprocs=ngpus_per_node, args=(ngpus_per_node, config, args))
  File "/opt/conda/lib/python3.11/site-packages/torch/multiprocessing/spawn.py", line 328, in spawn
    return start_processes(fn, args, nprocs, join, daemon, start_method="spawn")
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/multiprocessing/spawn.py", line 284, in start_processes
    while not context.join():
              ^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/multiprocessing/spawn.py", line 184, in join
    raise ProcessExitedException(
torch.multiprocessing.spawn.ProcessExitedException: process 1 terminated with signal SIGABRT
root@0dd7379cd6f8:/universal-llm-trainer# /opt/conda/lib/python3.11/multiprocessing/resource_tracker.py:254: UserWarning: resource_tracker: There appear to be 24 leaked semaphore objects to clean up at shutdown
  warnings.warn('resource_tracker: There appear to be %d '
```
* Please reduce `fsdp_hyperparameters.min_num_params` value of your training configuration.

&nbsp;

&nbsp;