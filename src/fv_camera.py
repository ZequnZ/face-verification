"""Use laptop camera to tryout face detection"""
# Author: @ZequnZ

import cv2
import numpy
from math import floor, ceil
import tensorflow as tf
import numpy as np
import align.detect_face
from itertools import combinations
from evaluation import get_embeddings, load_embeddings
from PIL import Image
import facenet


def init_mtcnn(gpu_memory_fraction=1.0):

    print("Creating networks and loading parameters")
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(
            config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)
        )
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
    return pnet, rnet, onet


def face_detection(image, pnet, rnet, onet, margin=44, image_size=160):
    """Detect face"""

    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    img_list = []

    # img = imageio.imread(os.path.expanduser(image), pilmode="RGB")
    img = image
    img_size = np.asarray(img.shape)[0:2]
    bounding_boxes, points = align.detect_face.detect_face(
        img, minsize, pnet, rnet, onet, threshold, factor
    )
    if len(bounding_boxes) < 1:
        # image_paths.remove(image)
        print("can't detect face")
        return None, None, None
    print(f"bound_boxes: {bounding_boxes}")
    print(f"points: {points}")

    for box in bounding_boxes:
        det = np.squeeze(box[0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0] - margin / 2, 0)
        bb[1] = np.maximum(det[1] - margin / 2, 0)
        bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
        bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
        cropped = img[bb[1] : bb[3], bb[0] : bb[2], :]
        # aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        aligned = np.array(
            Image.fromarray(cropped).resize((image_size, image_size), Image.BILINEAR)
        ).astype(np.double)
        prewhitened = facenet.prewhiten(aligned)
        img_list.append(prewhitened)

    images = np.stack(img_list)
    return bounding_boxes, points, images


def img2emb(img, model="./model"):
    facenet_graph = tf.Graph()
    with facenet_graph.as_default():

        with tf.Session() as sess:

            facenet.load_model(model)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name(
                "phase_train:0"
            )
            # Run forward pass to calculate embeddings
            feed_dict = {
                images_placeholder: img,
                phase_train_placeholder: False,
            }
            emb = sess.run(embeddings, feed_dict=feed_dict)
            return emb


cap = cv2.VideoCapture(0)
pnet, rnet, onet = init_mtcnn()
DRAWBOX = True
DRAWPOINT = False
DRAWNAME = True
cur_flag = -1
pre_flag = -1

if DRAWNAME is True:
    save_emb = load_embeddings("me")
    print(save_emb.keys())
else:
    save_emb = None


facenet_graph = tf.Graph()
with facenet_graph.as_default():

    with tf.Session() as sess:
        if save_emb is not None:
            facenet.load_model("./model")

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name(
                "phase_train:0"
            )

        while 1:
            # get a frame
            ret, image = cap.read()

            # filename = 'savedImage.jpg'
            # cv2.imwrite(filename, frame)

            # Blue color in BGR
            color = (255, 0, 0)
            thickness = 2

            # face detection
            bounding_boxes, points, images = face_detection(image, pnet, rnet, onet)
            if DRAWBOX and bounding_boxes is not None:
                for box, img in zip(bounding_boxes, images):
                    print(f"box: {box}")
                    # represents the top left corner of rectangle
                    start_point = (floor(box[0]), floor(box[1]))
                    # represents the bottom right corner of rectangle
                    end_point = (ceil(box[2]), ceil(box[3]))
                    image = cv2.rectangle(
                        image, start_point, end_point, color, thickness
                    )

                    # Run forward pass to calculate embeddings
                    img = np.expand_dims(img, axis=0)
                    feed_dict = {
                        images_placeholder: img,
                        phase_train_placeholder: False,
                    }
                    emb = sess.run(embeddings, feed_dict=feed_dict)
                    for key in save_emb:
                        dist = np.sqrt(
                            np.sum(np.square(np.subtract(emb, save_emb[key])))
                        )
                        print(f"dist: {round(dist,4)}")
                        if dist < 1:
                            image = cv2.putText(
                                image,
                                key,
                                start_point,
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                color,
                                2,
                            )

                    # image = cv2.putText(image,"test",start_point,cv2.FONT_HERSHEY_SIMPLEX,1,color,2)

            if DRAWPOINT and points is not None:
                points = np.array(points)
                for i in range(points.shape[1]):
                    for x, y in zip(points[0:5:1, i], points[5::1, i]):
                        center_coordinates = (x, y)
                        radius = 2
                        # color = (0, 0, 255)
                        image = cv2.circle(
                            image, center_coordinates, radius, color, thickness
                        )

            cv2.imshow("capture", image)

            # 获取键盘事件
            flag = cv2.waitKey(1)
            # Esc，退出
            if flag == 27:
                break
            # 判断是否按下其他键
            if flag > -1 and flag != pre_flag:
                cur_flag = flag
            pre_flag = flag

            # 响应事件
            if cur_flag == ord("b"):
                DRAWBOX = not DRAWBOX
            elif cur_flag == ord("n"):
                DRAWPOINT = not DRAWPOINT

    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
    # if cv2.waitKey(1) & 0xFF == ord('b'):
    #     DRAWBOX = not DRAWBOX
    #     continue
    # if cv2.waitKey(1) & 0xFF == ord('n'):
    #     DRAWPOINT = not DRAWPOINT
    #     continue
    # if cv2.waitKey(1) & 0xFF == ord('p'):
    #     cv2.waitkey(0)

cap.release()
cv2.destroyAllWindows()
