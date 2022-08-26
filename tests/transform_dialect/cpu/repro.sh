iree-build;

# Hanhan's command
build/tools/iree-compile --iree-mlir-to-vm-bytecode-module --iree-hal-target-backends=llvm-cpu \
  --iree-llvm-target-triple=x86_64-pc-linux-gnu --iree-llvm-target-cpu-features=host \
  ./tests/transform_dialect/cpu/matmul_128x384x384.mlir | \
taskset 80 build/tools/iree-benchmark-module --device=local-task   --task_topology_group_count=1

build/tools/iree-compile --iree-mlir-to-vm-bytecode-module --iree-hal-target-backends=llvm-cpu \
  --iree-llvm-target-triple=x86_64-pc-linux-gnu --iree-llvm-target-cpu-features=host \
  --iree-hal-benchmark-dispatch-repeat-count=100 \
  ./tests/transform_dialect/cpu/matmul_128x384x384.mlir | \
taskset 80 build/tools/iree-benchmark-module --device=local-task --task_topology_group_count=1 --batch_size=100

# TD
./build/tools/iree-compile ./tests/transform_dialect/cpu/matmul_128x384x384.mlir \
  --iree-hal-target-backends=llvm-cpu \
  --iree-flow-dispatch-use-transform-dialect=./tests/transform_dialect/cpu/matmul_128x384x384_dispatch_spec.mlir \
  --iree-codegen-llvmcpu-use-transform-dialect=./tests/transform_dialect/cpu/matmul_128x384x384_codegen_spec.mlir \
  --iree-llvm-target-triple=x86_64-pc-linux-gnu --iree-llvm-target-cpu-features=host \
  --iree-hal-benchmark-dispatch-repeat-count=10 | \
taskset 80 ./build/tools/iree-benchmark-module --device=local-task --task_topology_group_count=1 --batch_size=10
