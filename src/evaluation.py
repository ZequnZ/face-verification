"""Evalaute the performance of face verification"""
# Author: @ZequnZ

import os
import tensorflow as tf
import numpy as np
import random
import pickle
import math
from collections import defaultdict
from itertools import combinations
import functools
from PIL import Image
import imageio
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import facenet
import align.detect_face


def init_mtcnn(gpu_memory_fraction=1.0, model="../model"):
    """Initialize MTCNN and get the cnn"""

    # print("Creating networks and loading parameters...")
    print("Initialize MTCNN........")
    mtcnn_graph = tf.Graph()
    with mtcnn_graph.as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(
            config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)
        )
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
    return pnet, rnet, onet


def get_thumbnails(image_path, folders=None):
    """Get the thumbnails of all the images in the image_path"""

    img_list = []
    max_length = 0

    # if folders are not provided
    if not folders:
        for folder in os.listdir(image_path):
            if folder == ".DS_Store":
                continue
            row = None
            for _, _, files in os.walk(os.path.join(image_path, folder)):
                for file in files:

                    img = imageio.imread(
                        os.path.expanduser(os.path.join(image_path, folder, file)),
                        pilmode="RGB",
                    )

                    resized = Image.fromarray(img).resize((40, 40), Image.BILINEAR)
                    if row is None:
                        row = resized
                    else:
                        row = np.append(row, resized, axis=1)
                max_length = max(max_length, row.shape[1])
                img_list.append(row)
        imgs = list(
            map(
                lambda x: np.pad(
                    x,
                    ((0, 0), (0, max_length - x.shape[1]), (0, 0)),
                    "constant",
                    constant_values=(255),
                ),
                img_list,
            )
        )
        return functools.reduce(lambda x, y: np.concatenate((x, y), axis=0), imgs)

    # provided folders
    else:
        for folder in folders:
            if folder == ".DS_Store":
                continue
            row = None
            for _, _, files in os.walk(os.path.join(image_path, folder)):
                for file in files:

                    img = imageio.imread(
                        os.path.expanduser(os.path.join(image_path, folder, file)),
                        pilmode="RGB",
                    )

                    resized = Image.fromarray(img).resize((40, 40), Image.BILINEAR)
                    if row is None:
                        row = resized
                    else:
                        row = np.append(row, resized, axis=1)
                max_length = max(max_length, row.shape[1])
                img_list.append(row)
        imgs = list(
            map(
                lambda x: np.pad(
                    x,
                    ((0, 0), (0, max_length - x.shape[1]), (0, 0)),
                    "constant",
                    constant_values=(255),
                ),
                img_list,
            )
        )
        return functools.reduce(lambda x, y: np.concatenate((x, y), axis=0), imgs)


def get_cropped_face_img(image_path, margin=44, image_size=160, folders=None):
    """return cropped face img if face is detected,
        otherwise remove the img
    """

    pnet, rnet, onet = init_mtcnn()

    img_file_dict = {}
    cropped_face_img_dict = {}
    if not folders:
        for folder in os.listdir(image_path):
            for _, _, files in os.walk(os.path.join(image_path, folder)):
                img_file_dict[folder] = files
    else:
        for folder in folders:
            for _, _, files in os.walk(os.path.join(image_path, folder)):
                img_file_dict[folder] = files

    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    for folder in img_file_dict:
        img_list = []
        for image in img_file_dict[folder]:

            img = imageio.imread(
                os.path.expanduser(os.path.join(image_path, folder, image)),
                pilmode="RGB",
            )
            img_size = np.asarray(img.shape)[0:2]
            bounding_boxes, points = align.detect_face.detect_face(
                img, minsize, pnet, rnet, onet, threshold, factor
            )
            if len(bounding_boxes) < 1:
                img_file_dict[folder].remove(image)
                print("can't detect face, remove ", image)
                continue
            #             print(f"bound_boxes: {bounding_boxes}")
            #             print(f'points: {points}')

            det = np.squeeze(bounding_boxes[0, 0:4])
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0] - margin / 2, 0)
            bb[1] = np.maximum(det[1] - margin / 2, 0)
            bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
            bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
            cropped = img[bb[1] : bb[3], bb[0] : bb[2], :]
            aligned = np.array(
                Image.fromarray(cropped).resize(
                    (image_size, image_size), Image.BILINEAR
                )
            ).astype(np.double)
            prewhitened = facenet.prewhiten(aligned)
            img_list.append(prewhitened)

        # Only add to dict when list is not empty
        if img_list:
            cropped_face_img_dict[folder] = np.stack(img_list)

    return cropped_face_img_dict, img_file_dict


def face_2_embeddings(img_dict, model="../model", use_num_key=True):
    """cropped face imgs -> embeddings"""

    facenet_graph = tf.Graph()
    img_embeddings_dict = {}
    with facenet_graph.as_default():

        with tf.Session() as sess:

            facenet.load_model(model)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name(
                "phase_train:0"
            )

            if use_num_key:
                count = 0
                for folder in img_dict:

                    # Run forward pass to calculate embeddings
                    feed_dict = {
                        images_placeholder: img_dict[folder],
                        phase_train_placeholder: False,
                    }
                    emb = sess.run(embeddings, feed_dict=feed_dict)
                    img_embeddings_dict[count] = emb
                    count += 1
            else:
                for folder in img_dict:

                    # Run forward pass to calculate embeddings
                    feed_dict = {
                        images_placeholder: img_dict[folder],
                        phase_train_placeholder: False,
                    }
                    emb = sess.run(embeddings, feed_dict=feed_dict)
                    img_embeddings_dict[folder] = emb
            return img_embeddings_dict


def get_embeddings(image_path, name, margin=44, image_size=160):
    """image files -> embeddings"""

    pnet, rnet, onet = init_mtcnn()

    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    img_list = []
    emb = {}

    img = imageio.imread(os.path.expanduser(os.path.join(image_path)), pilmode="RGB",)
    img_size = np.asarray(img.shape)[0:2]
    bounding_boxes, points = align.detect_face.detect_face(
        img, minsize, pnet, rnet, onet, threshold, factor
    )
    if len(bounding_boxes) < 1:
        # img_file_dict[folder].remove(image)
        print("can't detect face, end")
        return

    det = np.squeeze(bounding_boxes[0, 0:4])
    bb = np.zeros(4, dtype=np.int32)
    bb[0] = np.maximum(det[0] - margin / 2, 0)
    bb[1] = np.maximum(det[1] - margin / 2, 0)
    bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
    bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
    cropped = img[bb[1] : bb[3], bb[0] : bb[2], :]
    aligned = np.array(
        Image.fromarray(cropped).resize((image_size, image_size), Image.BILINEAR)
    ).astype(np.double)
    prewhitened = facenet.prewhiten(aligned)
    img_list.append(prewhitened)

    if img_list:
        emb[name] = np.stack(img_list)
    return face_2_embeddings(emb, use_num_key=False)


def get_distance_matrix(emb, img_file_dict=None, print_matrix=False):
    """embeddings -> distance matrix"""

    distance_same_class_dict = defaultdict(list)
    distance_diff_class_dict = defaultdict(list)
    img_num_dict = {}

    if not print_matrix:

        if img_file_dict is None:
            for person in emb:

                nrof_images = emb[person].shape[0]
                img_num_dict[person] = nrof_images
                for i in range(nrof_images):
                    for j in range(nrof_images):
                        dist = np.sqrt(
                            np.sum(
                                np.square(
                                    np.subtract(emb[person][i, :], emb[person][j, :])
                                )
                            )
                        )
                        if i < j:
                            distance_same_class_dict[person].append(round(dist, 4))
        else:
            for person in emb:

                nrof_images = emb[person].shape[0]
                img_num_dict[person] = nrof_images
                for i in range(nrof_images):
                    for j in range(nrof_images):
                        dist = np.sqrt(
                            np.sum(
                                np.square(
                                    np.subtract(emb[person][i, :], emb[person][j, :])
                                )
                            )
                        )
                        if i < j:
                            distance_same_class_dict[person].append(round(dist, 4))

        if img_file_dict is None:

            for x, y in combinations((emb.keys()), 2):
                nrof_x, nrof_y = emb[x].shape[0], emb[y].shape[0]
                for i in range(nrof_x):

                    for j in range(nrof_y):
                        dist = np.sqrt(
                            np.sum(np.square(np.subtract(emb[x][i, :], emb[y][j, :])))
                        )
                        distance_diff_class_dict[(x, y)].append(round(dist, 4))
        else:

            for x, y in combinations((emb.keys()), 2):
                nrof_x, nrof_y = emb[x].shape[0], emb[y].shape[0]
                for i in range(nrof_x):
                    for j in range(nrof_y):
                        dist = np.sqrt(
                            np.sum(np.square(np.subtract(emb[x][i, :], emb[y][j, :])))
                        )
                        distance_diff_class_dict[(x, y)].append(round(dist, 4))
        return distance_same_class_dict, distance_diff_class_dict, img_num_dict

    # print matrix
    print("##################################################################")
    print("##################################################################")
    print("######### Distance between images of the same person #############")
    print("##################################################################")
    print("##################################################################")
    print("\n")

    if img_file_dict is None:
        for person in emb:

            nrof_images = emb[person].shape[0]
            img_num_dict[person] = nrof_images
            print(f"Distance matrix of class {person}")
            print("    ", end="")
            for i in range(nrof_images):
                print("    %1d     " % i, end="")
            print("")
            for i in range(nrof_images):
                print("%1d  " % i, end="")
                for j in range(nrof_images):
                    dist = np.sqrt(
                        np.sum(
                            np.square(np.subtract(emb[person][i, :], emb[person][j, :]))
                        )
                    )
                    print("  %1.4f  " % dist, end="")
                    if i < j:
                        distance_same_class_dict[person].append(round(dist, 4))
                print("")
    #             print(person,':',distance_same_class_dict[person])
    else:

        for person in emb:

            nrof_images = emb[person].shape[0]
            img_num_dict[person] = nrof_images
            print(f"Distance matrix of class {person}")
            print("    ", end="")
            for i in range(nrof_images):
                print("    {}     ".format(img_file_dict[person][i]), end="")
            print("")
            for i in range(nrof_images):
                print("{}  ".format(img_file_dict[person][i]), end="")
                for j in range(nrof_images):
                    dist = np.sqrt(
                        np.sum(
                            np.square(np.subtract(emb[person][i, :], emb[person][j, :]))
                        )
                    )
                    print("  %1.4f  " % dist, end="")
                    if i < j:
                        distance_same_class_dict[person].append(round(dist, 4))
                print("")

    print("\n")
    print("------------------------------------------------------------------")
    print("-------------------------------END--------------------------------")
    print("------------------------------------------------------------------")
    print("\n")

    print("##################################################################")
    print("##################################################################")
    print("######### Distance between images of different people ############")
    print("##################################################################")
    print("##################################################################")
    print("\n")

    if img_file_dict is None:

        for x, y in combinations((emb.keys()), 2):
            print(f"Distance matrix between class {x} and {y}")
            print("    ", end="")

            nrof_x, nrof_y = emb[x].shape[0], emb[y].shape[0]
            for i in range(nrof_y):
                print("    %1d     " % i, end="")
            print("")
            for i in range(nrof_x):
                print("%1d  " % i, end="")
                for j in range(nrof_y):
                    dist = np.sqrt(
                        np.sum(np.square(np.subtract(emb[x][i, :], emb[y][j, :])))
                    )
                    print("  %1.4f  " % dist, end="")
                    distance_diff_class_dict[(x, y)].append(round(dist, 4))
                print("")
    #         print(x,y,distance_diff_class_dict[(x,y)])
    else:

        for x, y in combinations((emb.keys()), 2):
            print(f"Distance matrix between class {x} and {y}")
            print("    ", end="")

            nrof_x, nrof_y = emb[x].shape[0], emb[y].shape[0]
            for i in range(nrof_y):
                print("    {}     ".format(img_file_dict[y][i]), end="")
            print("")
            for i in range(nrof_x):
                print("    {}     ".format(img_file_dict[x][i]), end="")
                for j in range(nrof_y):
                    dist = np.sqrt(
                        np.sum(np.square(np.subtract(emb[x][i, :], emb[y][j, :])))
                    )
                    print("  %1.4f  " % dist, end="")
                    distance_diff_class_dict[(x, y)].append(round(dist, 4))
                print("")
    #         print(x,y,distance_diff_class_dict[(x,y)])

    print("\n")
    print("------------------------------------------------------------------")
    print("-------------------------------END--------------------------------")
    print("------------------------------------------------------------------")
    print("\n")
    return distance_same_class_dict, distance_diff_class_dict, img_num_dict


def get_metrics(
    same_dict,
    diff_dict,
    img_num_dict,
    threshold=1,
    marker="o",
    sample_rate_same=None,
    sample_rate_diff=None,
):
    """Obtain some evaluation metrics"""

    # Avg distance
    same_dis = np.array(np.sum(list(same_dict.values())))
    diff_dis = np.array(np.sum(list(diff_dict.values())))
    print("##################################################################")
    print("##################################################################")
    print("####################### Average distance #########################")
    print("##################################################################")
    print("##################################################################")
    print("\n")
    print(
        f"Average distance between images of the same class: {round(sum(same_dis)/len(same_dis),4)}"
    )
    print(
        f"Average distance between images of different classes: {round(sum(diff_dis)/len(diff_dis),4)}\n"
    )

    print("##################################################################")
    print("##################################################################")
    print("################## Number of images per class ####################")
    print("##################################################################")
    print("##################################################################")
    print("\n")

    total = 0
    for k in img_num_dict:
        print(f"class {k}: {img_num_dict[k]} images")
        total += img_num_dict[k]
    print(f"total: {total} images")
    print("\n")

    # sample wise metric
    print("##################################################################")
    print("##################################################################")
    print("###################### Sample-wise metric ########################")
    print("##################################################################")
    print("##################################################################")
    print("\n")
    tp, tn, fp, fn = 0, 0, 0, 0
    tp = sum(same_dis <= threshold)
    tn = sum(diff_dis > threshold)
    fp = sum(diff_dis <= threshold)
    fn = sum(same_dis > threshold)
    f1_score = tp / (tp + 0.5 * (fp + fn))

    print(f"True positive: {tp}")
    print(f"True negative: {tn}")
    print(f"False positive: {fp}")
    print(f"False negative: {fn}")
    print(f"Accuracy: {round((tp+tn)/(len(same_dis)+len(diff_dis)),4)}")
    print(f"Precision: {round(tp/(tp+fp),4)}")
    print(f"Recall: {round(tp/len(same_dis),4)}")
    print(f"F1 score: {round(f1_score,4)}")

    # class wise metric
    # ref: https://arxiv.org/pdf/1503.03832.pdf
    print("\n")
    print("##################################################################")
    print("##################################################################")
    print("####################### Class-wise metric ########################")
    print("##################################################################")
    print("#############################################################n#####")
    print("\n")

    # true accepts
    ta = tp  # = sum(same_dis<=threshold)
    # false accepts
    fa = fp  # = sum(diff_dis<=threshold)
    # Validation rate VAL
    val = ta / len(same_dis)
    # False accept rate FAR
    far = fa / len(diff_dis)

    print(f"Validation rate: {round(val,4)}")
    print(f"False accept rate: {round(far,4)}")

    print("\n")
    print("##################################################################")
    print("##################################################################")
    print("############################ ROC-AUC #############################")
    print("##################################################################")
    print("##################################################################")
    print("\n")

    maxd = max(diff_dis)
    nor_same = same_dis / maxd
    nor_diff = diff_dis / maxd
    score = np.concatenate((nor_same, nor_diff))
    label = np.concatenate((np.zeros(nor_same.shape), np.ones(nor_diff.shape)))
    fpr, tpr, _ = roc_curve(label, score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(12, 12))
    lw = 2
    plt.plot(
        fpr, tpr, color="darkorange", lw=lw, label="ROC curve (area = %0.3f)" % roc_auc
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", size=16)
    plt.ylabel("True Positive Rate", size=16)
    plt.title("Receiver operating characteristic", size=17)
    plt.legend(loc="lower right", fontsize=13)
    plt.show()

    print("\n")
    print("##################################################################")
    print("##################################################################")
    print("#################### Scatter plot of distance ####################")
    print("##################################################################")
    print("##################################################################")
    print("\n")
    print(f"Threhold: {threshold}")

    # scatter plot of distance
    if sample_rate_diff:
        # Sample the points to make a better visualization
        print(
            f"{sample_rate_diff*100}% of diff class distances are sampled here in the scatterplot."
        )

        diff_dis = random.sample(
            list(diff_dis), math.floor(len(diff_dis) * sample_rate_diff)
        )

    if sample_rate_same:
        print(
            f"{sample_rate_same*100}% of same class distances are sampled here in the scatterplot."
        )
        same_dis = random.sample(
            list(same_dis), math.floor(len(same_dis) * sample_rate_same)
        )

    sd = np.random.random(len(same_dis)) * (max(1, int(len(diff_dis) / 100)))
    yd = np.random.random(len(diff_dis)) * (max(1, int(len(diff_dis) / 100)))
    plt.figure(figsize=(12, 12))
    plt.scatter(diff_dis, yd, c="red", label="Diff class", alpha=0.3, marker=marker)
    plt.scatter(same_dis, sd, c="blue", label="Same class", alpha=0.3, marker=marker)
    #     plt.ylim(-0.1, 1.1)
    plt.xlim(0, 2)
    plt.xlabel("Distance", size=18)
    plt.xticks(fontsize=16)
    plt.yticks([], [])
    plt.title("Scatter plot of distance", size=17)
    plt.legend(fontsize=13, loc=2)
    plt.axvline(x=threshold, c="green", dashes=(5, 5, 5, 5))
    plt.show()

    print("\n")
    print("------------------------------------------------------------------")
    print("-------------------------------END--------------------------------")
    print("------------------------------------------------------------------")
    print("\n")


def save_embeddings(emb, name):
    file = open(f"../data/emb/{name}.pkl", "wb")
    pickle.dump(emb, file)
    file.close()


def load_embeddings(name):
    if not os.path.exists(f"../data/emb/{name}.pkl"):
        print("Emb file doesn't exist.")
        return
    file = open(f"../data/emb/{name}.pkl", "rb")
    emb = pickle.load(file)
    file.close()
    return emb


def main(image_path, use_num_key=True, use_file_name=False, print_matrix=True):
    thumbnails = get_thumbnails(image_path)
    plt.figure(figsize=(8, 8))
    plt.imshow(thumbnails)
    plt.title("Thumbnails:", size=15)
    plt.axis("off")
    plt.show()
    print("")

    if use_file_name:
        cropped_face_dict, img_file_dict = get_cropped_face_img(image_path)
    else:
        cropped_face_dict, _ = get_cropped_face_img(image_path)

    emb_dict = face_2_embeddings(cropped_face_dict, use_num_key=use_num_key)
    if use_file_name:
        same_dict, diff_dict, img_num_dict = get_distance_matrix(
            emb_dict, img_file_dict, print_matrix=print_matrix
        )
    else:
        same_dict, diff_dict, img_num_dict = get_distance_matrix(emb_dict)
    get_metrics(same_dict, diff_dict, img_num_dict)


def dataset_eva(
    image_path,
    use_num_key=True,
    use_file_name=False,
    folders=None,
    marker="o",
    sample_rate_same=None,
    sample_rate_diff=None,
):
    if use_file_name:
        cropped_face_dict, img_file_dict = get_cropped_face_img(
            image_path, folders=folders
        )
    else:
        cropped_face_dict, _ = get_cropped_face_img(image_path, folders=folders)

    emb_dict = face_2_embeddings(cropped_face_dict, use_num_key=use_num_key)
    if use_file_name:
        same_dict, diff_dict, img_num_dict = get_distance_matrix(
            emb_dict, img_file_dict
        )
    else:
        same_dict, diff_dict, img_num_dict = get_distance_matrix(emb_dict)
    get_metrics(
        same_dict,
        diff_dict,
        img_num_dict,
        marker=marker,
        sample_rate_same=sample_rate_same,
        sample_rate_diff=sample_rate_diff,
    )


#     return same_dict, diff_dict
