ROCM_DIR := /opt/rocm

CXX_FLAGS := -std=c++17 -mf16c
CXX_FLAGS += -Wall -Wextra -Wno-unused-result
CXX_FLAGS += -I$(ROCM_DIR)/include -L$(ROCM_DIR)/lib

CXX_DEFS := -D__HIP_PLATFORM_AMD__
CXX_DEFS += -DROCM_USE_FLOAT16

run_hipblaslt: run_hipblaslt.cpp
	hipcc $(CXX_FLAGS) $(CXX_DEFS) -lhipblaslt $^ -o $@

.PHONY: clean
clean:
	rm --recursive --force run_hipblaslt *~
