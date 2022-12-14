
import struct
import numpy as np
import cv2
import collections
import os

def __resize_image(src_image, dst_image_height, dst_image_width):
    src_image_height = src_image.shape[0]
    src_image_width = src_image.shape[1]

    if src_image_height > dst_image_height or src_image_width > dst_image_width:
        height_scale = dst_image_height / src_image_height
        width_scale = dst_image_width / src_image_width
        scale = min(height_scale, width_scale)
        img = cv2.resize(src=src_image, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    else:
        img = src_image
        
    img_height = img.shape[0]
    img_width = img.shape[1]
    dst_image = np.zeros(shape=[dst_image_height, dst_image_width], dtype=np.uint8)
    y_offset = (dst_image_height - img_height) // 2
    x_offset = (dst_image_width - img_width) // 2
    dst_image[y_offset:y_offset+img_height, x_offset:x_offset+img_width] = img
    return dst_image

def read_hoda_cdb(file_name):
    with open(file_name, 'rb') as binary_file:
        data = binary_file.read()
        offset = 0
        # read private header
        yy = struct.unpack_from('H', data, offset)[0]
        offset += 2
        m = struct.unpack_from('B', data, offset)[0]
        offset += 1
        d = struct.unpack_from('B', data, offset)[0]
        offset += 1
        H = struct.unpack_from('B', data, offset)[0]
        offset += 1
        W = struct.unpack_from('B', data, offset)[0]
        offset += 1
        TotalRec = struct.unpack_from('I', data, offset)[0]
        offset += 4
        LetterCount = struct.unpack_from('128I', data, offset)
        offset += 128 * 4
        imgType = struct.unpack_from('B', data, offset)[0]  # 0: binary, 1: gray
        offset += 1
        Comments = struct.unpack_from('256c', data, offset)
        offset += 256 * 1
        Reserved = struct.unpack_from('245c', data, offset)
        offset += 245 * 1

        if (W > 0) and (H > 0):
            normal = True
        else:
            normal = False

        images = []
        labels = []
        for i in range(TotalRec):
            StartByte = struct.unpack_from('B', data, offset)[0]  # must be 0xff
            offset += 1
            label = struct.unpack_from('B', data, offset)[0]
            offset += 1
            if not normal:
                W = struct.unpack_from('B', data, offset)[0]
                offset += 1
                H = struct.unpack_from('B', data, offset)[0]
                offset += 1
            ByteCount = struct.unpack_from('H', data, offset)[0]
            offset += 2
            image = np.zeros(shape=[H, W], dtype=np.uint8)

            if imgType == 0:
                # Binary
                for y in range(H):
                    bWhite = True
                    counter = 0
                    while counter < W:
                        WBcount = struct.unpack_from('B', data, offset)[0]
                        offset += 1
                        # x = 0
                        # while x < WBcount:
                        #     if bWhite:
                        #         image[y, x + counter] = 0  # Background
                        #     else:
                        #         image[y, x + counter] = 255  # ForeGround
                        #     x += 1
                        if bWhite:
                            image[y, counter:counter + WBcount] = 0  # Background
                        else:
                            image[y, counter:counter + WBcount] = 255  # ForeGround
                        bWhite = not bWhite  # black white black white ...
                        counter += WBcount
            else:
                # GrayScale mode
                data = struct.unpack_from('{}B'.format(W * H), data, offset)
                offset += W * H
                image = np.asarray(data, dtype=np.uint8).reshape([W, H]).T

            images.append(image)
            labels.append(label)
        # print(labels)
        return images, labels


def img_writer(img_path, image,i):
    cv2.imwrite( img_path + "/" + str(i) + ".jpg",image)


def read_hoda_dataset(dataset_path,img_path, images_height=32, images_width=32, reshape=True):
    images, labels = read_hoda_cdb(dataset_path)
    for i,img in enumerate(images):
        img_writer(img_path,img,i)

    assert len(images) == len(labels)

    X = np.zeros(shape=[len(images), images_height, images_width], dtype=np.float32)
    Y = np.zeros(shape=[len(labels)], dtype=np.int)

    for i in range(len(images)):
        image = images[i]
        # Image resizing.
        image = __resize_image(src_image=image, dst_image_height=images_height, dst_image_width=images_width)
        # Image normalization.
        image = image / 255
        # Image binarization.
        image = np.where(image >= 0.5, 1, 0)
        # Image.
        X[i] = image
        # Label.
        Y[i] = labels[i]
    return X,Y

img_path_train = "../Hoda_dataset/hoda_images/train_images"
img_path_test = "../Hoda_dataset/hoda_images/test_images"
npz_file_path = "../Hoda_dataset/hoda_npz_files"
need_folders = [img_path_train, img_path_test,npz_file_path]

for i in need_folders:
    try:
        os.makedirs(i)
    except:
        pass

pathlist_data = [ "../Hoda_dataset/hoda_cdb_files/Train_60000.cdb", "../Hoda_dataset/hoda_cdb_files/Test_20000.cdb"]
mat_data_files_path = ["HODA_MAT_DATA_TRAIN.npz", "HODA_MAT_DATA_TEST.npz"]
for i,path in enumerate(pathlist_data):
    images, labels = read_hoda_dataset(path,need_folders[i])
    np.savez(os.path.join(need_folders[2], mat_data_files_path[i]), name1=images, name2=labels)

