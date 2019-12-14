.PHONY: build
build:
	docker build . -t bone

.PHONY: run
run: build
	docker run --rm -it -p 5000:5000 bone
