.PHONY: all clean build-c build-go build-cython

# Compiler settings
CC = gcc
CFLAGS = -O3 -shared -fPIC -Wall

# Paths - Adjusted for new structure
C_SRC = src/envs/cartpole/c_src/cartpole.c
C_OUT = src/envs/cartpole/c_src/libcartpole.so

GO_SRC = src/envs/cartpole/go_src/cartpole.go
GO_OUT = src/envs/cartpole/go_src/cartpole.so

all: build-c build-go build-cython

build-c:
	@echo "Building C extension..."
	$(CC) $(CFLAGS) -o $(C_OUT) $(C_SRC)
	@echo "Done."

build-go:
	@echo "Building Go extension..."
	cd src/envs/cartpole/go_src && go build -buildmode=c-shared -o cartpole.so cartpole.go
	@echo "Done."

build-cython:
	@echo "Building Cython extension..."
	python setup.py build_ext --inplace
	@echo "Done."

clean:
	rm -f src/envs/cartpole/c_src/*.so
	rm -f src/envs/cartpole/go_src/*.so src/envs/cartpole/go_src/*.h
	rm -f src/envs/cartpole/*.so src/envs/cartpole/*.c
	rm -f src/envs/boids/*.so src/envs/boids/*.c
	rm -rf build/
