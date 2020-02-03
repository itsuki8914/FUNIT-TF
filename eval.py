import os
import numpy as np
import argparse
import glob
import tensorflow as tf
import time
from utils import tileImage
from btgen import BatchGenerator
from FUNIT import FUNIT
import cv2

def main(arg):
    dataset = 'datasets'
    model_name = 'FUNIT'
    validation_output_dir = os.path.join('evaluation')
    os.makedirs(validation_output_dir, exist_ok=True)

    content_dir = arg.content_dir
    class_dir = arg.class_dir
    K = arg.K

    content_dir = os.path.join('datasets', content_dir)
    class_dir = os.path.join('datasets_val', class_dir)

    #print(glob.glob(content_dir + '/*'))
    #print(glob.glob(class_dir + '/*'))
    classes = os.listdir(dataset)
    num_classes = len(classes)
    print(num_classes)
    img_size = arg.img_size
    mini_batch_size = 1

    model = FUNIT(img_size=img_size, num_classes=num_classes, batch_size=mini_batch_size,
            mode='eval')

    ckpt = tf.train.get_checkpoint_state(os.path.join('experiments', model_name, 'checkpoints'))
    if ckpt:
        #last_model = ckpt.all_model_checkpoint_paths[0]
        last_model = ckpt.model_checkpoint_path
        print("loading {}".format(last_model))
        model.load(filepath=last_model)
    else:
        print("checkpoints are not found.")
        print("to inference must need a trained model.")
        return


    for file in glob.glob(content_dir + '/*'):
        print(file)
        class_images = np.random.choice(glob.glob(class_dir + '/*'), size=K, replace=False)
        class_K_batch = np.zeros( (K, img_size, img_size, 3), dtype=np.float32)
        for i in range(K):
            img = cv2.imread(class_images[i])
            img = cv2.resize(img,(img_size, img_size))
            class_K_batch[i,:,:,:] = (img - 127.5) / 127.5

        content_batch = np.zeros( (1, img_size, img_size, 3), dtype=np.float32)
        img = cv2.imread(file)
        img = cv2.resize(img,(img_size, img_size))
        content_batch[0,:,:,:] = (img - 127.5) / 127.5

        gen_img = model.eval(content_batch, class_K_batch)

        gen_img = np.array(gen_img)
        #gen_img = np.squeeze(gen_img)
        gen_img = tileImage(gen_img)
        #print(gen_img)
        out = (gen_img + 1)*127.5

        print(out.shape)
        path = os.path.join(validation_output_dir, "to{}_{}.png".format(os.path.splitext(os.path.basename(class_dir))[0], os.path.splitext(os.path.basename(file))[0]))
        print(path)
        cv2.imwrite(path, out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--content_dir',"-cnt", dest='content_dir', type=str, default=None, required=True, help='content dir')
    parser.add_argument('--class_dir',"-cls", dest='class_dir', type=str, default=None, required=True, help='class dir')
    parser.add_argument('--K',"-k", dest='K', type=int, default=1, help='test K shot')
    parser.add_argument('--img_size',"-i", dest='img_size', type=int, default=128, help='image size')
    args = parser.parse_args()
    start_time= time.time()
    main(args)
    end_time = time.time()
    time_elapsed = end_time - start_time

    print('Preprocessing Done.')

    print('Time Elapsed for Data Preprocessing: %02d:%02d:%02d' % (
        time_elapsed // 3600, (time_elapsed % 3600 // 60), (time_elapsed % 60 // 1)))
