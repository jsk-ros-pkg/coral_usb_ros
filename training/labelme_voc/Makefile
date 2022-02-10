all: tensorflow

tensorflow: Dockerfile
	echo "Building $@"
	cp ../common/scripts scripts -r
	docker build -t train-edgetpu-object-detection --build-arg CACHEBUST=$(shell date +%Y%m%d) -f Dockerfile .
	rm scripts -rf
