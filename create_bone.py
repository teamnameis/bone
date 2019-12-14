import cv2
from estimater import Estimater
import glob


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    #parser.add_argument('img', type=str, nargs='+')
    parser.add_argument('img', type=str)
    parser.add_argument('--size', type=str, default='432x368')
    parser.add_argument('--model', type=str, default='mobilenet_thin')
    args = parser.parse_args()

    e = Estimater(size=args.size, model=args.model)
    bones = []
    imgs = glob.glob(f'{args.img}/*/*.jpeg')
    for img in imgs:
        src_img = cv2.imread(img)
        humans = e.run(src_img)
        bone = Estimater.shape(humans)
        #kimono_img = cv2.imread(img.replace('bone', 'kimono').replace('jpeg', 'png'))
        bones.append({'image_path': img, 'points': bone})

    import pickle
    #with open('kimono.pickle', 'wb') as f:
    with open('bone.pickle', 'wb') as f:
        pickle.dump(bones, f)
    
