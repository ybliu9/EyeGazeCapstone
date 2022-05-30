import os
import shutil

paths = ['train', 'trainval', 'val', 'test']

'''create folder'''
for path in paths:
    images_path = 'images/' + path
    labels_path = 'labels/' + path
    if not os.path.exists(images_path):
        os.makedirs(images_path)
    else:
        shutil.rmtree(images_path)
        os.makedirs(images_path)
    if not os.path.exists(labels_path):
        os.makedirs(labels_path)
    else:
        shutil.rmtree(labels_path)
        os.makedirs(labels_path)

'''move files under folder'''
for path in paths:
    txt_path = 'ImageSets/Main/' + path + '.txt'
    with open(txt_path, 'r') as txt_file:
        while True:
            image_name = txt_file.readline().strip('\n')
            if not image_name:
                break
            image_from = 'images/' + image_name + '.png'
            image_to = 'images/' + path + '/' + image_name + '.png'
            label_from = 'labels/' + image_name + '.txt'
            label_to = 'labels/' + path + '/' + image_name + '.txt'
            if not os.path.isfile(image_to):
                shutil.copy(image_from, image_to)
            if not os.path.isfile(label_to):
                shutil.copy(label_from, label_to)
print('copy done')