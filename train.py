import os
import random
import numpy as np
import glob
import librosa
import tensorflow as tf
from utils import tileImage
from btgen import BatchGenerator
from FUNIT import FUNIT
import cv2
seed = 0

random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

def main():
    dataset = 'datasets'
    model_name = 'FUNIT'
    os.makedirs(os.path.join('experiments', model_name, 'checkpoints'), exist_ok=True)
    log_dir = os.path.join('logs', model_name)
    os.makedirs(log_dir, exist_ok=True)

    validation_output_dir = 'sample'
    os.makedirs(validation_output_dir, exist_ok=True)

    classes = os.listdir(dataset)
    num_classes = len(classes)
    print(num_classes)
    img_size = 128
    btGen = BatchGenerator(img_size=img_size, imgdir=dataset, num_classes=num_classes)

    num_iterations = 100000
    mini_batch_size = 8
    generator_learning_rate = 0.00010
    discriminator_learning_rate = 0.00010
    lambda_fm = 1
    lambda_rec = 0.1

    model = FUNIT(img_size=img_size, num_classes=num_classes, batch_size=mini_batch_size,
                rec_weight=lambda_rec, feature_weight=lambda_fm,log_dir=log_dir)

    ckpt = tf.train.get_checkpoint_state(os.path.join('experiments', model_name, 'checkpoints'))
    if ckpt:
        #last_model = ckpt.all_model_checkpoint_paths[1]
        last_model = ckpt.model_checkpoint_path
        print("loading {}".format(last_model))
        model.load(filepath=last_model)
    else:
        print("checkpoints are not found")

    iteration = 1
    while iteration <= num_iterations:
        generator_learning_rate *=0.99999
        discriminator_learning_rate *=0.99999

        cont_img, cont_label, cls_img, cls_label = btGen.getBatch(mini_batch_size)

        # to One-hot
        cont_labels = np.zeros([mini_batch_size, num_classes])
        cls_labels = np.zeros([mini_batch_size, num_classes])
        for b in range(mini_batch_size):
            cont_labels[b] = np.identity(num_classes)[cont_label[b]]
            cls_labels[b] = np.identity(num_classes)[cls_label[b]]


        gen_loss, dis_loss = model.train(content_image=cont_img, class_image=cls_img,
                    content_label=cont_labels, class_label=cls_labels,discriminator_learning_rate=discriminator_learning_rate,
                    generator_learning_rate=generator_learning_rate)

        print('Iteration: {:07d}, Generator Loss : {:.3f}, Discriminator Loss : {:.3f}'.format(iteration,
                                                                                               gen_loss,
                                                                                               dis_loss))

        if iteration % 5000 == 0:
            print('Checkpointing...')
            model.save(directory=os.path.join('experiments', model_name, 'checkpoints'),
                       filename='{}_{}.ckpt'.format(model_name, iteration))

        if iteration % 100 == 0 or iteration==1:
            cont_img, cont_label, cls_img, cls_label = btGen.getBatch(mini_batch_size)
            for b in range(mini_batch_size):
                cont_labels[b] = np.identity(num_classes)[cont_label[b]]
                cls_labels[b] = np.identity(num_classes)[cls_label[b]]
            gen_img = model.test(cont_img, cls_img)
            gen_img = np.array(gen_img)
            gen_img = np.squeeze(gen_img)
            print(gen_img.shape)
            contTiled = tileImage(cont_img)
            clsTiled = tileImage(cls_img)
            genTiled = tileImage(gen_img)

            out = np.concatenate([contTiled, clsTiled, genTiled], axis=1)
            out = (out+1)*127.5
            print(out.shape)
            cv2.imwrite("{}/{:07}.png".format(validation_output_dir, iteration), out)

        iteration +=1

if __name__ == '__main__':
    main()
