import os
from PIL import ImageFile
from keras.preprocessing import image

ImageFile.LOAD_TRUNCATED_IMAGES = True


def format_img(input_dir, output_dir, size=32):
    print('Reformating...')
    for file_name in os.listdir(input_dir):
        path_name = os.path.join(input_dir, file_name)
        img = image.load_img(path_name, target_size=(size, size))
        path_name = os.path.join(output_dir, file_name)
        img.save(path_name)


def rescale_img(src_dir, output_dir, size):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        os.makedirs(output_dir + '/train')
        os.makedirs(output_dir + '/test')
    format_img(src_dir + '/train', output_dir + '/train', size)
    format_img(src_dir + '/test', output_dir + '/test', size)


