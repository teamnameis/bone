import cv2
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh


class Estimater:
    def __init__(self, size='432x368', model='mobilenet_thin'):
        w, h = model_wh(size)
        if w == 0 or h == 0:
            self.e = TfPoseEstimator(get_graph_path(model), target_size=(432, 368))
        else:
            self.e = TfPoseEstimator(get_graph_path(model), target_size=(w, h))
        self.w, self.h = w, h

    def run(self, img):
        return self.e.inference(img, resize_to_default=(self.w > 0 and self.h > 0), upsample_size=4.0)

    @staticmethod
    def shape(humans):
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
        return resp

    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('img', type=str, nargs='+')
    parser.add_argument('--size', type=str, default='432x368')
    parser.add_argument('--model', type=str, default='mobilenet_thin')
    args = parser.parse_args()

    e = Estimater(size=args.size, model=args.model)
    for img in args.img:
        src_img = cv2.imread(img)
        humans = e.run(src_img)
        dst_img = TfPoseEstimator.draw_humans(src_img, humans, imgcopy=False)
        cv2.imwrite(f'{img}_bone.png', dst_img)
