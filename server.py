from bone_pb2 import *
from bone_pb2_grpc import MLServicer, add_MLServicer_to_server
import cv2
import numpy as np
from estimater import Estimater
import grpc
from concurrent import futures
import time
from tf_pose.estimator import TfPoseEstimator


def overlay_kimono(src_image, dst_image, src_point, dst_point):
    return src_image


class Servicer(MLServicer):
    def __init__(self, estimater, kimono):
        self.e = estimater
        self.kimono = kimono

    def Morph(self, request, context):
        kimono = self.kimono[request.id]
        kimono_img, kimono_point = kimono['img'], kimono['points'][0]
        human_array = np.asarray(bytearray(request.data), dtype=np.uint8)
        human_img = cv2.imdecode(human_array, 1)
        humans = estimater.run(human_img)
        human_point = Estimater.shape(humans)
        image = overlay_kimono(kimono_img, human_img, kimono_point, human_point)
        # image = TfPoseEstimator.draw_humans(img, humans, imgcopy=False)
        _, morphed = cv2.imencode('.jpg', image)
        return Image(data=morphed.tobytes())


def serve(estimater, kimono, port):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_MLServicer_to_server(Servicer(estimater, kimono), server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--size', type=str, default='432x368')
    parser.add_argument('--model', type=str, default='mobilenet_thin')
    args = parser.parse_args()

    import pickle
    with open('kimono.pickle', 'rb') as f:
        kimono = pickle.load(f)
        estimater = Estimater(size=args.size, model=args.model)
        serve(estimater, kimono, args.port)
