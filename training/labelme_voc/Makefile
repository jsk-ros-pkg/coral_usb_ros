all: tensorflow

tensorflow: Dockerfile
	echo "Building $@"
	docker build -t train-edgetpu-object-detection --build-arg CACHEBUST=$(shell date +%Y%m%d) -f Dockerfile .
