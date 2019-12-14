from bone_pb2 import *
from bone_pb2_grpc import MLStub
import grpc
import cv2


def main(imgs):
    with grpc.insecure_channel('localhost:5000') as ch:
        stub = MLStub(ch)

        for img in imgs:
            with open(img, 'rb') as input:
                morphed = stub.Morph(Flame(id=1, data=input.read()))
                with open(f'{img}_bone.png', 'wb') as output:
                    output.write(morphed.data)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('img', type=str, nargs='+')
    args = parser.parse_args()
 
    main(args.img)
