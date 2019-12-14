.PHONY: build
build:
	docker build . -t bone

.PHONY: run
run: build
	docker run --rm -it -p 5000:5000 bone

.PHONY: proto
proto:
	python -m grpc_tools.protoc -I=./ --python_out=./ --grpc_python_out=./ bone.proto
