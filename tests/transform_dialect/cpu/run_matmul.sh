./build/tools/iree-compile ./tests/transform_dialect/cpu/matmul_128x384x384.mlir \
  --iree-hal-target-backends=llvm-cpu \
  --iree-llvm-target-triple=x86_64-pc-linux-gnu \
  --iree-llvm-target-cpu-features=host \
  --iree-hal-benchmark-dispatch-repeat-count=1000 | \
./build/tools/iree-benchmark-module --device=local-task --task_topology_group_count=0 --batch_size=1000

./build/tools/iree-compile ./tests/transform_dialect/cpu/matmul_128x384x384.mlir \
  --iree-hal-target-backends=llvm-cpu \
  --iree-flow-dispatch-use-transform-dialect=./tests/transform_dialect/cpu/matmul_128x384x384_dispatch_spec.mlir \
  --iree-codegen-llvmcpu-use-transform-dialect=./tests/transform_dialect/cpu/matmul_128x384x384_codegen_spec.mlir \
  --iree-llvm-target-triple=x86_64-pc-linux-gnu \
  --iree-llvm-target-cpu-features=host \
  --iree-hal-benchmark-dispatch-repeat-count=1000 | \
./build/tools/iree-benchmark-module --device=local-task --task_topology_group_count=0 --batch_size=1000

./build/tools/iree-compile ./tests/transform_dialect/cpu/matmul_128x384x1536.mlir \
  --iree-hal-target-backends=llvm-cpu \
  --iree-llvm-target-triple=x86_64-pc-linux-gnu \
  --iree-llvm-target-cpu-features=host \
  --iree-hal-benchmark-dispatch-repeat-count=1000 | \
./build/tools/iree-benchmark-module --device=local-task --task_topology_group_count=0 --batch_size=1000

./build/tools/iree-compile ./tests/transform_dialect/cpu/matmul_128x384x1536.mlir \
  --iree-hal-target-backends=llvm-cpu \
  --iree-flow-dispatch-use-transform-dialect=./tests/transform_dialect/cpu/matmul_128x384x1536_dispatch_spec.mlir \
  --iree-codegen-llvmcpu-use-transform-dialect=./tests/transform_dialect/cpu/matmul_128x384x1536_codegen_spec.mlir \
  --iree-llvm-target-triple=x86_64-pc-linux-gnu \
  --iree-llvm-target-cpu-features=host \
  --iree-hal-benchmark-dispatch-repeat-count=1000 | \
./build/tools/iree-benchmark-module --device=local-task --task_topology_group_count=0 --batch_size=1000

./build/tools/iree-compile ./tests/transform_dialect/cpu/matmul_128x1536x384.mlir \
  --iree-hal-target-backends=llvm-cpu \
  --iree-llvm-target-triple=x86_64-pc-linux-gnu \
  --iree-llvm-target-cpu-features=host \
  --iree-hal-benchmark-dispatch-repeat-count=1000 | \
./build/tools/iree-benchmark-module --device=local-task --task_topology_group_count=0 --batch_size=1000

./build/tools/iree-compile ./tests/transform_dialect/cpu/matmul_128x1536x384.mlir \
  --iree-hal-target-backends=llvm-cpu \
  --iree-flow-dispatch-use-transform-dialect=./tests/transform_dialect/cpu/matmul_128x1536x384_dispatch_spec.mlir \
  --iree-codegen-llvmcpu-use-transform-dialect=./tests/transform_dialect/cpu/matmul_128x1536x384_codegen_spec.mlir \
  --iree-llvm-target-triple=x86_64-pc-linux-gnu \
  --iree-llvm-target-cpu-features=host \
  --iree-hal-benchmark-dispatch-repeat-count=1000 | \
./build/tools/iree-benchmark-module --device=local-task --task_topology_group_count=0 --batch_size=1000

