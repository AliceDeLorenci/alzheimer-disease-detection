import tensorflow as tf
import random

import numpy as np
from glob import glob


class_code = {"AD": 0, "CN": 1, "EMCI": 2, "LMCI": 3}


@tf.function
def parse_function_aug(img_file, label):

    # image input
    image = tf.io.read_file(img_file)
    image = tf.image.decode_image(image, expand_animations=False, channels=3)
    image = tf.image.resize(image, (224, 224))
    image = tf.cast(image, tf.float32)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)

    # print("img shape: ", tf.shape(image))
    noise = tf.random.normal(shape=[224, 224], mean=0.0, stddev=0.02, dtype=tf.float32)
    # print("noise shape: ", tf.shape(noise))

    img_list = [tf.add(image[:, :, 0], noise), tf.add(image[:, :, 1], noise), tf.add(image[:, :, 2], noise)]
    image = tf.stack(img_list)
    # print("img shape: ", tf.shape(image))
    image = tf.transpose(image, [1, 2, 0])
    # print("img shape: ", tf.shape(image))

    # label = tf.cast(tf.one_hot(tf.cast(label, tf.uint8), depth=4), tf.float32)

    # label
    # label = tf.convert_to_tensor(label)
    return image, label


@tf.function
def parse_function(img_file, label):

    # image input
    image = tf.io.read_file(img_file)
    image = tf.image.decode_image(image, expand_animations=False, channels=3)
    image = tf.image.resize(image, (224, 224))
    image = tf.cast(image, tf.float32)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)

    # print("img shape: ", tf.shape(image))

    # label = tf.cast(tf.one_hot(tf.cast(label, tf.uint8), depth=4), tf.float32)

    # label
    # label = tf.convert_to_tensor(label)
    return image, label


def loadDataset(path, aug=True, batch_size=16, val_split=0.1, test_split=0.1):

    print("start loading dataset\n")

    images_path = sorted(glob(path + "*.png"))
    images_path = list(filter(lambda x: x.split("_")[-1].split(".")[0] != "MCI", images_path))

    random.shuffle(images_path)

    val_size = int(len(images_path) * val_split)
    test_size = int(len(images_path) * test_split)

    train_imgs = images_path[0 : -val_size - test_size]
    val_imgs = images_path[-val_size - test_size : -test_size]
    test_imgs = images_path[-test_size:]

    print(train_imgs[10])
    # create label
    train_label = [class_code[train_img.split("_")[-1].split(".")[0]] for train_img in train_imgs]
    val_label = [class_code[val_img.split("_")[-1].split(".")[0]] for val_img in val_imgs]
    test_label = [class_code[test_img.split("_")[-1].split(".")[0]] for test_img in test_imgs]

    ndata = len(train_label)
    class_count = np.zeros(len(class_code))
    class_weights = np.zeros(len(class_code))
    for tl in train_label:
        class_count[tl] += 1

    for i, c in enumerate(class_code.keys()):
        class_weights[i] = ndata / (len(class_code) * class_count[i])

    class_weights = dict(enumerate(class_weights.flatten()))

    print("class_weights:", class_weights)

    print(train_label[10])
    # -----------dataset-----------
    print("tamanho do dataset train: " + str(len(train_imgs)) + " | " + str(len(train_label)))
    print("tamanho do dataset val: " + str(len(val_imgs)) + " | " + str(len(val_label)))
    print("tamanho do dataset test: " + str(len(test_imgs)) + " | " + str(len(test_label)))

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    trainDataset = tf.data.Dataset.from_tensor_slices((train_imgs, train_label)).repeat(1)
    trainDataset = trainDataset.map(parse_function_aug if aug else parse_function, num_parallel_calls=AUTOTUNE)
    trainDataset = trainDataset.batch(batch_size)
    trainDataset = trainDataset.prefetch(AUTOTUNE)
    trainDataset = trainDataset.shuffle(buffer_size=4)

    valDataset = tf.data.Dataset.from_tensor_slices((val_imgs, val_label)).repeat(1)
    valDataset = valDataset.map(parse_function, num_parallel_calls=AUTOTUNE)
    valDataset = valDataset.batch(batch_size)
    valDataset = valDataset.prefetch(AUTOTUNE)

    testDataset = tf.data.Dataset.from_tensor_slices((test_imgs, test_label)).repeat(1)
    testDataset = testDataset.map(parse_function, num_parallel_calls=AUTOTUNE)
    testDataset = testDataset.batch(batch_size)
    testDataset = testDataset.prefetch(AUTOTUNE)

    return trainDataset, valDataset, testDataset, class_weights
