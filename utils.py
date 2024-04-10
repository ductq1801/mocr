import numpy as np
from model.mocr import MOCR
from model.vocab import Vocab
import math
from PIL import Image
import torch
import lmdb
from tqdm import tqdm 
import sys 
import cv2
from torchmetrics.functional.text import word_error_rate,char_error_rate

def checkImageIsValid(imageBin):
    isvalid = True
    imgH = None
    imgW = None

    imageBuf = np.fromstring(imageBin, dtype=np.uint8)
    try:
        img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)

        imgH, imgW = img.shape[0], img.shape[1]
        if imgH * imgW == 0:
            isvalid = False
    except Exception as e:
        isvalid = False

    return isvalid, imgH, imgW
def compute_accuracy_v1(ground_truth, predictions, mode='full_sequence'):
    """
    Computes accuracy
    :param ground_truth:
    :param predictions:
    :param display: Whether to print values to stdout
    :param mode: if 'per_char' is selected then
                 single_label_accuracy = correct_predicted_char_nums_of_single_sample / single_label_char_nums
                 avg_label_accuracy = sum(single_label_accuracy) / label_nums
                 if 'full_sequence' is selected then
                 single_label_accuracy = 1 if the prediction result is exactly the same as label else 0
                 avg_label_accuracy = sum(single_label_accuracy) / label_nums
    :return: avg_label_accuracy
    """
    if mode == 'per_char':

        accuracy = []

        for index, label in enumerate(ground_truth):
            prediction = predictions[index]
            total_count = len(label)
            correct_count = 0
            try:
                for i, tmp in enumerate(label):
                    if tmp == prediction[i]:
                        correct_count += 1
            except IndexError:
                continue
            finally:
                try:
                    accuracy.append(correct_count / total_count)
                except ZeroDivisionError:
                    if len(prediction) == 0:
                        accuracy.append(1)
                    else:
                        accuracy.append(0)
        avg_accuracy = np.mean(np.array(accuracy).astype(np.float32), axis=0)
    elif mode == 'full_sequence':
        try:
            correct_count = 0
            for index, label in enumerate(ground_truth):
                prediction = predictions[index]
                if prediction == label:
                    correct_count += 1
            avg_accuracy = correct_count / len(ground_truth)
        except ZeroDivisionError:
            if not predictions:
                avg_accuracy = 1
            else:
                avg_accuracy = 0
    else:
        raise NotImplementedError('Other accuracy compute mode has not been implemented')

    return avg_accuracy

def compute_accuracy(ground_truth, predictions, mode='full_sequence',type='acc'):
    if mode == 'full_sequence':
        err = word_error_rate(predictions,ground_truth)
        if type == 'acc':
            return max(1-err,0)
        return (err,0)
    err = char_error_rate(predictions,ground_truth)
    if type == 'acc':
        return max(1-err,0)
    return (err,0)
def build_model(config):
    vocab = Vocab(config['vocab'])
    device = config['device']
    
    model = MOCR(len(vocab),
            config['image_encoder'], 
            config['transformer'],
            )
    
    model = model.to(device)

    return model, vocab
import os


class Logger():
    def __init__(self, fname):
        path, _ = os.path.split(fname)
        os.makedirs(path, exist_ok=True)

        self.logger = open(fname, 'w')

    def log(self, string):
        self.logger.write(string+'\n')
        self.logger.flush()

    def close(self):
        self.logger.close()
def resize(w, h, expected_height, image_min_width, image_max_width):
    new_w = int(expected_height * float(w) / float(h))
    round_to = 10
    new_w = math.ceil(new_w/round_to)*round_to
    new_w = max(new_w, image_min_width)
    new_w = min(new_w, image_max_width)

    return new_w, expected_height

def process_image(img, image_height, image_min_width, image_max_width):
    img = np.array(img)
    img = Image.fromarray(img)
    w, h = img.size
    new_w, image_height = resize(w, h, image_height, image_min_width, image_max_width)

    img = img.resize((new_w, image_height), Image.LANCZOS)

    img = np.asarray(img).transpose(2,0, 1)
    img = img/255
    return img

def process_input(image, image_height, image_min_width, image_max_width):
    img = process_image(image, image_height, image_min_width, image_max_width)
    img = img[np.newaxis, ...]
    img = torch.FloatTensor(img)
    return img

def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k.encode(), v)

def createDataset(outputPath, root_dir, df_annotation):
    """
    Create LMDB dataset for CRNN training.
    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """
    annotations = [list(df_annotation.iloc[i].values) for i in range(len(df_annotation))]

    nSamples = len(annotations)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 0
    error = 0
    
    pbar = tqdm(range(nSamples), ncols = 100, desc='Create {}'.format(outputPath)) 
    for i in pbar:
        imageFile, label = annotations[i]
        imagePath = os.path.join(root_dir, imageFile)

        if not os.path.exists(imagePath):
            error += 1
            continue
        
        with open(imagePath, 'rb') as f:
            imageBin = f.read()
        isvalid, imgH, imgW = checkImageIsValid(imageBin)

        if not isvalid:
            error += 1
            continue

        imageKey = 'image-%09d' % cnt
        labelKey = 'label-%09d' % cnt
        pathKey = 'path-%09d' % cnt
        dimKey = 'dim-%09d' % cnt

        cache[imageKey] = imageBin
        cache[labelKey] = label.encode()
        cache[pathKey] = imageFile.encode()
        cache[dimKey] = np.array([imgH, imgW], dtype=np.int32).tobytes()

        cnt += 1

        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}

    nSamples = cnt-1
    cache['num-samples'] = str(nSamples).encode()
    writeCache(env, cache)

    if error > 0:
        print('Remove {} invalid images'.format(error))
    print('Created dataset with %d samples' % nSamples)
    sys.stdout.flush()