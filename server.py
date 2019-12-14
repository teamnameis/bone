import argparse
from flask import Flask, request, jsonify
import cv2
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh


app = Flask(__name__)
w, h = model_wh('432x368')
if w == 0 or h == 0:
    e = TfPoseEstimator(get_graph_path('mobilenet_thin'), target_size=(432, 368))
else:
    e = TfPoseEstimator(get_graph_path('mobilenet_thin'), target_size=(w, h))


@app.route("/", methods=['POST'])
def bone():
    if request.files['image']:
        st = request.files['image'].stream
        img_array = np.asarray(bytearray(st.read()), dtype=np.uint8)
        img = cv2.imdecode(img_array, 1)
        humans = e.inference(img, resize_to_default=(w > 0 and h > 0), upsample_size=4.0)
        resp = []
        for human in humans:
            bone = []
            for i in range(0, 18):
                body_parts = human.body_parts
                if i in body_parts:
                    bone.append({"x": body_parts[i].x, "y": body_parts[i].y, "score":body_parts[i].score})
                else:
                    bone.append({"x": -1, "y": -1, "score": -1})
            resp.append(bone)
        return jsonify(resp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5000)
    args = parser.parse_args()

    app.run(port=args.port)
