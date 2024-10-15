ROCM_DIR := /opt/rocm

CXX_FLAGS := -std=c++17 -mf16c
CXX_FLAGS += -Wall -Wextra -Wno-unused-result -pedantic
CXX_FLAGS += -I$(ROCM_DIR)/include -L$(ROCM_DIR)/lib -lhipblaslt

CXX_DEFS := -D__HIP_PLATFORM_AMD__
CXX_DEFS += -DROCM_USE_FLOAT16

SOURCE := run_hipblaslt.cpp
PROGRAM := $(basename $(SOURCE))

$(PROGRAM): $(SOURCE)
	hipcc $(CXX_FLAGS) $(CXX_DEFS) $^ -o $@

.PHONY: run
run: $(PROGRAM)
	./$(PROGRAM)

.PHONY: clean
clean:
	rm --recursive --force $(PROGRAM) *~ *'#'*
