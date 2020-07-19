import cv2  
import numpy
import sys
import os
from math import floor, ceil
import matplotlib.pyplot as plot
import tensorflow as tf
import numpy as np
from PIL import Image
import imageio
import align.detect_face
import copy


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

def face_detection(image, pnet, rnet, onet):
    '''Detect face'''

    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    # img = imageio.imread(os.path.expanduser(image), pilmode="RGB")
    img = image
    img_size = np.asarray(img.shape)[0:2]
    bounding_boxes, points = align.detect_face.detect_face(
        img, minsize, pnet, rnet, onet, threshold, factor
    )
    if len(bounding_boxes) < 1:
        # image_paths.remove(image)
        print("can't detect face")
        return None,None
    print(f"bound_boxes: {bounding_boxes}")
    print(f'points: {points}')
    return bounding_boxes, points
    # det = np.squeeze(bounding_boxes[0, 0:4])
    # bb = np.zeros(4, dtype=np.int32)
    # bb[0] = np.maximum(det[0] - margin / 2, 0)
    # bb[1] = np.maximum(det[1] - margin / 2, 0)
    # bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
    # bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
    # cropped = img[bb[1] : bb[3], bb[0] : bb[2], :]
    # # aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
    # aligned = np.array(
    #     Image.fromarray(cropped).resize((image_size, image_size), Image.BILINEAR)
    # ).astype(np.double)
    # prewhitened = facenet.prewhiten(aligned)
    # img_list.append(prewhitened)

    # images = np.stack(img_list)
    # return cropped


cap = cv2.VideoCapture(0)
pnet, rnet, onet = init_mtcnn()
DRAWBOX=True
DRAWPOINT=True
cur_flag = -1
pre_flag = -1

while(1):
    # get a frame
    ret, frame = cap.read()
    
    # filename = 'savedImage.jpg'
    # cv2.imwrite(filename, frame) 

    # Blue color in BGR 
    color = (255, 0, 0) 
    thickness = 2

    # face detection
    bounding_boxes, points = face_detection(frame, pnet, rnet, onet)
    if DRAWBOX and bounding_boxes is not None:
        for box in bounding_boxes:
            # represents the top left corner of rectangle 
            start_point = (floor(box[0]), floor(box[1]))
            # represents the bottom right corner of rectangle 
            end_point = (ceil(box[2]), ceil(box[3])) 
            image = cv2.rectangle(frame, start_point, end_point, color, thickness)
        cv2.imshow('capture', image)
    else:
        cv2.imshow("capture", frame)
    
    if DRAWPOINT and points is not None:
        points = np.array(points)
        for i in range(points.shape[1]):
            for x,y in zip(points[0:5:1,i],points[5::1,i]):
                center_coordinates = (x,y)
                radius = 2
                # color = (0, 0, 255)
                image = cv2.circle(image, center_coordinates, radius, color, thickness)
        cv2.imshow('capture', image)
    else:
        cv2.imshow("capture", frame)
        
    
    #获取键盘事件
    flag = cv2.waitKey(1)
    #Esc，退出
    if flag == 27:
        break
    #判断是否按下其他键
    if flag > -1 and flag != pre_flag:
        cur_flag = flag
    pre_flag = flag
    
    #响应事件
    if cur_flag == ord('b'):
        DRAWBOX = not DRAWBOX
    elif cur_flag == ord('n'):
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
