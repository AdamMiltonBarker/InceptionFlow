# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# *****************************************************************************
# InceptionFlow
# Copyright (c) 2018 Adam Milton-Barker - AdamMiltonBarker.com
# Based on Google's Tensorflow Imagenet Inception V3
# Uses TechBubble IoT JumpWay MQTT Client
# *****************************************************************************

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os.path
import random
import re
import glob
import hashlib
import tarfile
import json

from datetime import datetime

import numpy as np
from six.moves import urllib
import tensorflow as tf
import cv2

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

import struct

MODEL_DIR = os.getcwd()+"/model/imagenet"   
        
OBJECT_MODEL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz' 
OBJECT_MODEL_PATH = os.getcwd()+"/model/imagenet/classify_image_graph_def.pb"
OBJECT_MODEL_LABELS_PATH = os.getcwd()+"/model/imagenet/imagenet_2012_challenge_label_map_proto.pbtxt" 
OBJECT_MODEL_LABELSH_PATH = os.getcwd()+"/model/imagenet/imagenet_synset_to_human_label_map.txt"  

FACIAL_IMAGES_DIR = "model/training/facial"
FACIAL_MODEL_PATH = os.getcwd()+"/model/InceptionFlowFacial.pb"
FACIAL_MODEL_LABELS_PATH = os.getcwd()+"/model/InceptionFlowFacial.txt" 

CUSTOM_IMAGES_DIR = "model/training/custom"
CUSTOM_MODEL_PATH = os.getcwd()+"/model/InceptionFlowCustom.pb"
CUSTOM_MODEL_LABELS_PATH = os.getcwd()+"/model/InceptionFlowCustom.txt"

BOTTLENECK_DIR = 'model/bottleneck'
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
BOTTLENECK_TENSOR_SIZE = 2048

MODEL_INPUT_WIDTH = 299
MODEL_INPUT_HEIGHT = 299
MODEL_INPUT_DEPTH = 3

INTERMEDIATE_STORE_FREQ = 0

JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear:0'
FINAL_TENSOR_NAME = "final_result"

MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1 
NUM_TOP_PREDICTIONS = 5
TRAINING_STEPS = 4000
STEP_INTERVAL = 10

TEST_BATCH_SIZE = 5
TRAIN_BATCH_SIZE = 100
VALIDATION_BATCH_SIZE = 100

LEARNING_RATE = 0.01
SUMMARIES_DIR = "model/retrain_logs"
ARCHITECTURE = "inception_v3"
TESTING_PERCENTAGE = 10
VALIDATION_PERCENTAGE = 10
PRINT_MISC = True

FLIP_LEFT_RIGHT = False
RANDOM_CROP = 0
RANDOM_SCALE = 0
RANDOM_BRIGHTNESS = 0

class NodeLookup(object):
    
    """Converts integer node ID's to human readable labels."""
    
    def __init__(
        self,
        label_lookup_path=None,
        uid_lookup_path=None):
        
        if not label_lookup_path:
            
            label_lookup_path = OBJECT_MODEL_LABELS_PATH
                
        if not uid_lookup_path:
            
            uid_lookup_path = OBJECT_MODEL_LABELSH_PATH
                
        self.node_lookup = self.load(label_lookup_path, uid_lookup_path)
        
    def load(self, label_lookup_path, uid_lookup_path):

        """
        Loads a human readable English name for each softmax node.
        Args:
            label_lookup_path: string UID to integer node ID.
            uid_lookup_path: string UID to human-readable string.
        Returns:
            dict from integer node ID to human-readable string.
        """
        if not tf.gfile.Exists(uid_lookup_path):
            
            tf.logging.fatal('File does not exist %s', uid_lookup_path)
            print('File does not exist %s', uid_lookup_path)
            
        if not tf.gfile.Exists(label_lookup_path):
            
            tf.logging.fatal('File does not exist %s', label_lookup_path)
            print('File does not exist %s', label_lookup_path)

        #print('LOCATED MODEL LABELS')
            
        # Loads mapping from string UID to human-readable string
            
        proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
        uid_to_human = {}
        p = re.compile(r'[n\d]*[ \S,]*')

        for line in proto_as_ascii_lines:
            
            parsed_items = p.findall(line)
            #print(parsed_items)
            uid = parsed_items[0]
            human_string = parsed_items[2]
            uid_to_human[uid] = human_string
            
        # Loads mapping from string UID to integer node ID.
            
        node_id_to_uid = {}
        proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()

        for line in proto_as_ascii:
            
            if line.startswith('  target_class:'):
                
                target_class = int(line.split(': ')[1])
                
            if line.startswith('  target_class_string:'):
                
                target_class_string = line.split(': ')[1]
                node_id_to_uid[target_class] = target_class_string[1:-2]
                    
        # Loads the final mapping of integer node ID to human-readable string
        node_id_to_name = {}
        #print(uid_to_human)
        for key, val in node_id_to_uid.items():
            
            if val not in uid_to_human:
                
                tf.logging.fatal('Failed to locate: %s', val)
                print('Failed to locate: %s', val)
            
            name = uid_to_human[val]
            node_id_to_name[key] = name
            
        return node_id_to_name
    
    def id_to_string(self, node_id):
        
        if node_id not in self.node_lookup:
            return ''
        return self.node_lookup[node_id]

class InceptionFlow():
    
    def __init__(self):
        
        self.confs = {}

        self.BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
        self.BOTTLENECK_TENSOR_SIZE = 2048
        self.MODEL_INPUT_WIDTH = 299
        self.MODEL_INPUT_HEIGHT = 299
        self.MODEL_INPUT_DEPTH = 3
        self.JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
        self.RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear:0'

        self.MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1
        
        with open('data/confs.json') as confs:
            
            self.confs = json.loads(confs.read())
            
    def saveImage(self,frame):
        
        timeDirectory =  os.getcwd()+"/data/captures/"+datetime.now().strftime('%Y-%m-%d')+'/'+datetime.now().strftime('%H')
        
        if not os.path.exists(timeDirectory):
            os.makedirs(timeDirectory)

        currentImage=timeDirectory+'/'+datetime.now().strftime('%M-%S')+'.jpg'
        print(currentImage)
        print("")
        
        cv2.imwrite(currentImage, frame)

        return currentImage
            
    def captureAndDetect(self,frame):

        faceCascade = cv2.CascadeClassifier( os.getcwd()+'/model/'+self.confs["ClassifierSettings"]["HAAR_FACES"])
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(gray,
            scaleFactor=self.confs["ClassifierSettings"]["HAAR_SCALE_FACTOR"],
            minNeighbors=self.confs["ClassifierSettings"]["HAAR_MIN_NEIGHBORS"],
            minSize=(30,30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        if not len(faces):
            faceCascade = cv2.CascadeClassifier( os.getcwd()+'/model/'+self.confs["ClassifierSettings"]["HAAR_FACES2"])
            faces = faceCascade.detectMultiScale(gray,
                scaleFactor=self.confs["ClassifierSettings"]["HAAR_SCALE_FACTOR"],
                minNeighbors=self.confs["ClassifierSettings"]["HAAR_MIN_NEIGHBORS"],
                minSize=(30,30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

        if not len(faces):
            faceCascade = cv2.CascadeClassifier( os.getcwd()+'/model/'+self.confs["ClassifierSettings"]["HAAR_FACES3"])
            faces = faceCascade.detectMultiScale(gray,
                scaleFactor=self.confs["ClassifierSettings"]["HAAR_SCALE_FACTOR"],
                minNeighbors=self.confs["ClassifierSettings"]["HAAR_MIN_NEIGHBORS"],
                minSize=(30,30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

        if not len(faces):
            faceCascade = cv2.CascadeClassifier( os.getcwd()+'/model/'+self.confs["ClassifierSettings"]["HAAR_PROFILES"])
            faces = faceCascade.detectMultiScale(gray,
                scaleFactor=self.confs["ClassifierSettings"]["HAAR_SCALE_FACTOR"],
                minNeighbors=self.confs["ClassifierSettings"]["HAAR_MIN_NEIGHBORS"],
                minSize=(30,30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
        
        timeDirectory =  os.getcwd()+"/data/captures/"+datetime.now().strftime('%Y-%m-%d')+'/'+datetime.now().strftime('%H')
        
        if not os.path.exists(timeDirectory):
            os.makedirs(timeDirectory)

        currentImage=timeDirectory+'/'+datetime.now().strftime('%M-%S')+'-raw.jpg'
        inceptionImage=timeDirectory+'/'+datetime.now().strftime('%M-%S')+'-Processed.jpg'

        if len(faces):

            x, y, w, h = faces[0]
            print("Cropping image")
            cropped = self.crop(frame, x, y, w, h)
            print("Writing image")
            cv2.imwrite(inceptionImage, cropped)

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame,datetime.now().strftime("%Y-%m-%d %H:%M"),(10,450), font, 1,(255,255,255),2)
            cv2.putText(frame,"InceptionFlow - AdamMiltonBarker.com",(10,40), font, 1,(255,255,255),2)

            for (x, y, w, h) in faces:

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
            cv2.imwrite(currentImage, frame)
            print("Found " + str(len(faces)))
            return currentImage, inceptionImage, frame, faces[0]

        else:

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame,datetime.now().strftime("%Y-%m-%d %H:%M"),(10,450), font, 1,(255,255,255),2)
            cv2.putText(frame,"InceptionFlow - AdamMiltonBarker.com",(10,40), font, 1,(255,255,255),2)

            print("Found " + str(len(faces)))
            return currentImage, None, frame, None  

    def resize(self,image):

            return cv2.resize(image,(self.confs["ClassifierSettings"]["INCEPTION_SIZE"], self.confs["ClassifierSettings"]["INCEPTION_SIZE"]),interpolation=cv2.INTER_LANCZOS4)

    def crop(self,image, x, y, w, h):
        crop_height = int((self.confs["ClassifierSettings"]["INCEPTION_SIZE"] / float(self.confs["ClassifierSettings"]["INCEPTION_SIZE"]))*w)
        midy = y + h/2
        y1 = max(0, midy-crop_height/2)
        y2 = min(image.shape[0]-1, midy+crop_height/2)
        return image[int(y1):int(y2), x:x+w]
            
    def createGraph(self,graphType="ObjectCam"):
        
        if graphType == "ObjectCam" or graphType == "ObjectTest":
            
            print("Creating Default Object Graph")
            
            with tf.gfile.FastGFile(OBJECT_MODEL_PATH, 'rb') as f:
                
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                _ = tf.import_graph_def(graph_def, name='')
                
        elif graphType == "FacialCam" or graphType == "FacialTest":
            
            if os.path.exists(FACIAL_MODEL_PATH):
                
                print("Creating Facial Graph")
                
                with tf.gfile.FastGFile(FACIAL_MODEL_PATH, 'rb') as f:
                
                    graph_def = tf.GraphDef()
                    graph_def.ParseFromString(f.read())
                    _ = tf.import_graph_def(graph_def, name='')
                    
        elif  graphType == "CustomCam" or graphType == "CustomTest":
            
            if os.path.exists(CUSTOM_MODEL_PATH):
                
                print("Creating Custom Graph")

                with tf.gfile.FastGFile(CUSTOM_MODEL_PATH, 'rb') as f:

                    graph_def = tf.GraphDef()
                    graph_def.ParseFromString(f.read())
                    _ = tf.import_graph_def(graph_def, name='')
                    graph_def = tf.GraphDef()
                    graph_def.ParseFromString(f.read())
                    _ = tf.import_graph_def(graph_def, name='') 
            
    def classifyObject(self,image):
        
        """
        Runs inference on an image.
        
        Args:
            image: Image file name.
        Returns:
            Nothing
        """
        
        if not tf.gfile.Exists(image):
            
            tf.logging.fatal('File does not exist %s', image)
            
        image_data = tf.gfile.FastGFile(image, 'rb').read()
            
        with tf.Session() as sess:
            
            # Some useful tensors:
            # 'softmax:0': A tensor containing the normalized prediction across
            # 1000 labels.
            # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
            # float description of the image.
            # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
            # encoding of the image.
            # # Runs the softmax tensor by feeding the image_data as input to the graph.
            
            softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
            predictions = sess.run(
                softmax_tensor,
                {'DecodeJpeg/contents:0': image_data})
            
            predictions = np.squeeze(predictions)
            
            # Creates node ID --> English string lookup.
            node_lookup = NodeLookup()
            
            top_k = predictions.argsort()[-NUM_TOP_PREDICTIONS:][::-1]
            
            print('')
            print('TOP PREDICTIONS:')
            for node_id in top_k:
                
                human_string = node_lookup.id_to_string(node_id)
                score = predictions[node_id]
                print('%s (score = %.5f)' % (human_string, score))
            print('')

            human_string = node_lookup.id_to_string(top_k[0])
            score = predictions[top_k[0]]
            return human_string, score 
            
    def classifyCustom(self,image):
        
        """
        Runs inference on an image.
        
        Args:
            image: Image file name.
        Returns:
            Nothing
        """

        classification = None
        image = tf.gfile.FastGFile(image,'rb').read()

        with tf.Session() as sess:

            softmaxTensor = sess.graph.get_tensor_by_name('final_result:0')
            predictions = sess.run(softmaxTensor,{'DecodeJpeg/contents:0': image})
            predictions = np.squeeze(predictions)
            topPredictions = predictions.argsort()[-5:][::-1]
            f = open(CUSTOM_MODEL_LABELS_PATH, 'rb')
            lines = f.readlines()
            labels = [str(w).replace("\n","") for w in lines]

            for node in topPredictions:
                classification = labels[node]
                score = predictions[node]
                print('%s (score = %.5f)' % (classification, score))

            print("")
            topClassification = labels[topPredictions[0]]
            topClassification = topClassification.replace("b'", "")
            topClassification = topClassification.replace("'", "")
            topClassification = topClassification.replace("\\n", "")
            score = predictions[topPredictions[0]]
            topScore = predictions[topPredictions[0]]

            return topClassification, topScore 
            
    def classifyFace(self,image):
        
        """
        Runs inference on an image.
        
        Args:
            image: Image file name.
        Returns:
            Nothing
        """

        classification = None
        image = tf.gfile.FastGFile(image,'rb').read()

        with tf.Session() as sess:

            softmaxTensor = sess.graph.get_tensor_by_name('final_result:0')
            predictions = sess.run(softmaxTensor,{'DecodeJpeg/contents:0': image})
            predictions = np.squeeze(predictions)
            topPredictions = predictions.argsort()[-5:][::-1]
            f = open(FACIAL_MODEL_LABELS_PATH, 'rb')
            lines = f.readlines()
            labels = [str(w).replace("\n","") for w in lines]

            for node in topPredictions:
                classification = labels[node]
                score = predictions[node]
                print('%s (score = %.5f)' % (classification, score))

            print("")
            topClassification = labels[topPredictions[0]]
            topClassification = topClassification.replace("b'", "")
            topClassification = topClassification.replace("'", "")
            topClassification = topClassification.replace("\\n", "")
            score = predictions[topPredictions[0]]
            topScore = predictions[topPredictions[0]]

            return topClassification, topScore

    def checkModelDownload(self):
        
        """Download and extract model tar file."""
        
        dest_directory = MODEL_DIR
        
        if not os.path.exists(dest_directory):
            
            os.makedirs(dest_directory)
            
        filename = OBJECT_MODEL.split('/')[-1]
        filepath = os.path.join(dest_directory, filename)
        
        if not os.path.exists(filepath):
            
            def _progress(count, block_size, total_size):
                
                sys.stdout.write('\r>> Downloading %s %.1f%%' % (
                    filename,
                    float(count * block_size) / float(total_size) * 100.0))
                    
                sys.stdout.flush()

            filepath, _ = urllib.request.urlretrieve(OBJECT_MODEL, filepath, _progress)
            print()
            statinfo = os.stat(filepath)
            print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

        tarfile.open(filepath, 'r:gz').extractall(dest_directory)

    def create_image_lists(self, image_dir, testing_percentage, validation_percentage):
        if not gfile.Exists(image_dir):
            print("Image directory '" + image_dir + "' not found.")
            return None
        result = {}
        sub_dirs = [x[0] for x in os.walk(image_dir)]
        

        is_root_dir = True
        for sub_dir in sub_dirs:
            if is_root_dir:
                is_root_dir = False
                continue
            extensions = ['jpg', 'Jpg', 'jpeg', 'JPG', 'JPEG']
            file_list = []
            dir_name = os.path.basename(sub_dir)
            if dir_name == image_dir:
                continue
            print("Looking for images in '" + dir_name + "'")
            for extension in extensions:
                file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
                file_list.extend(glob.glob(file_glob))
            if not file_list:
                print('No files found')
                continue
            if len(file_list) < 20:
                print('WARNING: Folder has less than 20 images, which may cause issues.')
            elif len(file_list) > self.MAX_NUM_IMAGES_PER_CLASS:
                print('WARNING: Folder {} has more than {} images. Some images will ' 'never be selected.'.format(dir_name, self.MAX_NUM_IMAGES_PER_CLASS))
            label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
            training_images = []
            testing_images = []
            validation_images = []
            for file_name in file_list:
                base_name = os.path.basename(file_name)
                hash_name = re.sub(r'_nohash_.*$', '', file_name)
                hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
                percentage_hash = ((int(hash_name_hashed, 16) %
                                    (self.MAX_NUM_IMAGES_PER_CLASS + 1)) *
                                    (100.0 / self.MAX_NUM_IMAGES_PER_CLASS))
                if percentage_hash < validation_percentage:
                    validation_images.append(base_name)
                elif percentage_hash < (testing_percentage + validation_percentage):
                    testing_images.append(base_name)
                else:
                    training_images.append(base_name)
            result[label_name] = {
                'dir': dir_name,
                'training': training_images,
                'testing': testing_images,
                'validation': validation_images,
            }
        return result


    def get_image_path(self, image_lists, label_name, index, image_dir, category):
        if label_name not in image_lists:
            tf.logging.fatal('Label does not exist %s.', label_name)
        label_lists = image_lists[label_name]
        if category not in label_lists:
            tf.logging.fatal('Category does not exist %s.', category)
        category_list = label_lists[category]
        if not category_list:
            tf.logging.fatal('Label %s has no images in the category %s.',
                            label_name, category)
        mod_index = index % len(category_list)
        base_name = category_list[mod_index]
        sub_dir = label_lists['dir']
        full_path = os.path.join(image_dir, sub_dir, base_name)
        return full_path


    def get_bottleneck_path(self, image_lists, label_name, index, bottleneck_dir,
                            category):
        return self.get_image_path(image_lists, label_name, index, bottleneck_dir,category) + '.txt'


    def create_inception_graph(self):
        with tf.Session() as sess:
            model_filename = os.path.join(
                MODEL_DIR, 'classify_image_graph_def.pb')
            with gfile.FastGFile(model_filename, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                bottleneck_tensor, jpeg_data_tensor, resized_input_tensor = (
                    tf.import_graph_def(graph_def, name='', return_elements=[
                        self.BOTTLENECK_TENSOR_NAME, self.JPEG_DATA_TENSOR_NAME,
                        self.RESIZED_INPUT_TENSOR_NAME]))
        return sess.graph, bottleneck_tensor, jpeg_data_tensor, resized_input_tensor


    def run_bottleneck_on_image(self, sess, image_data, image_data_tensor,
                                bottleneck_tensor):
        bottleneck_values = sess.run(
            bottleneck_tensor,
            {image_data_tensor: image_data})
        bottleneck_values = np.squeeze(bottleneck_values)
        return bottleneck_values

    def maybeDownload(self):

        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)

        filename = OBJECT_MODEL.split('/')[-1]
        filepath = os.path.join(MODEL_DIR, filename)

        if not os.path.exists(filepath):

            def _progress(count, block_size, total_size):
                
                sys.stdout.write('\r>> Downloading %s %.1f%%' %(
                    filename,
                    float(count * block_size) / float(total_size) * 100.0))

                sys.stdout.flush()

            filepath, _ = urllib.request.urlretrieve(
                OBJECT_MODEL,
                filepath,
                _progress)

            print()

            statinfo = os.stat(filepath)

            print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

        tarfile.open(filepath,'r:gz').extractall(MODEL_DIR)


    def ensure_dir_exists(self, dir_name):
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)


    def write_list_of_floats_to_file(self, list_of_floats , file_path):

        s = struct.pack('d' * self.BOTTLENECK_TENSOR_SIZE, *list_of_floats)
        with open(file_path, 'wb') as f:
            f.write(s)


    def read_list_of_floats_from_file(self, file_path):

        with open(file_path, 'rb') as f:
            s = struct.unpack('d' * self.BOTTLENECK_TENSOR_SIZE, f.read())
            return list(s)


        bottleneck_path_2_bottleneck_values = {}


    def get_or_create_bottleneck(self, sess, image_lists, label_name, index, image_dir,
                                category, bottleneck_dir, jpeg_data_tensor,
                                bottleneck_tensor):
        label_lists = image_lists[label_name]
        sub_dir = label_lists['dir']
        sub_dir_path = os.path.join(bottleneck_dir, sub_dir)
        self.ensure_dir_exists(sub_dir_path)
        bottleneck_path = self.get_bottleneck_path(image_lists, label_name, index,
                                                bottleneck_dir, category)
        if not os.path.exists(bottleneck_path):
            print('Creating bottleneck at ' + bottleneck_path)
            image_path = self.get_image_path(image_lists, label_name, index, image_dir,
                                        category)
            if not gfile.Exists(image_path):
                tf.logging.fatal('File does not exist %s', image_path)
            image_data = gfile.FastGFile(image_path, 'rb').read()
            bottleneck_values = self.run_bottleneck_on_image(sess, image_data,
                                                        jpeg_data_tensor,
                                                        bottleneck_tensor)
            bottleneck_string = ','.join(str(x) for x in bottleneck_values)
            with open(bottleneck_path, 'w') as bottleneck_file:
                bottleneck_file.write(bottleneck_string)

        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
        return bottleneck_values


    def cache_bottlenecks(self, sess, image_lists, image_dir, bottleneck_dir,
                        jpeg_data_tensor, bottleneck_tensor):
        how_many_bottlenecks = 0
        self.ensure_dir_exists(bottleneck_dir)
        for label_name, label_lists in image_lists.items():
            for category in ['training', 'testing', 'validation']:
                category_list = label_lists[category]
                for index, unused_base_name in enumerate(category_list):
                    self.get_or_create_bottleneck(sess, image_lists, label_name, index,
                                            image_dir, category, bottleneck_dir,
                                            jpeg_data_tensor, bottleneck_tensor)
                    how_many_bottlenecks += 1
                    if how_many_bottlenecks % 100 == 0:
                        print(str(how_many_bottlenecks) + ' bottleneck files created.')


    def get_random_cached_bottlenecks(self, sess, image_lists, how_many, category,
                                    bottleneck_dir, image_dir, jpeg_data_tensor,
                                    bottleneck_tensor):
        class_count = len(image_lists.keys())
        bottlenecks = []
        ground_truths = []
        for unused_i in range(how_many):
            label_index = random.randrange(class_count)
            label_name = list(image_lists.keys())[label_index]
            image_index = random.randrange(self.MAX_NUM_IMAGES_PER_CLASS + 1)
            bottleneck = self.get_or_create_bottleneck(sess, image_lists, label_name,image_index, image_dir, category, bottleneck_dir, jpeg_data_tensor,bottleneck_tensor)
            ground_truth = np.zeros(class_count, dtype=np.float32)
            ground_truth[label_index] = 1.0
            bottlenecks.append(bottleneck)
            ground_truths.append(ground_truth)
        return bottlenecks, ground_truths


    def get_random_distorted_bottlenecks(
        self, sess, image_lists, how_many, category, image_dir, input_jpeg_tensor,
        distorted_image, resized_input_tensor, bottleneck_tensor):
        class_count = len(image_lists.keys())
        bottlenecks = []
        ground_truths = []
        for unused_i in range(how_many):
            label_index = random.randrange(class_count)
            label_name = list(image_lists.keys())[label_index]
            image_index = random.randrange(self.MAX_NUM_IMAGES_PER_CLASS + 1)
            image_path = self.get_image_path(image_lists, label_name, image_index, image_dir,
                                        category)
            if not gfile.Exists(image_path):
                tf.logging.fatal('File does not exist %s', image_path)
            jpeg_data = gfile.FastGFile(image_path, 'rb').read()
            
            distorted_image_data = sess.run(distorted_image,
                                            {input_jpeg_tensor: jpeg_data})
            bottleneck = self.run_bottleneck_on_image(sess, distorted_image_data,
                                                resized_input_tensor,
                                                bottleneck_tensor)
            ground_truth = np.zeros(class_count, dtype=np.float32)
            ground_truth[label_index] = 1.0
            bottlenecks.append(bottleneck)
            ground_truths.append(ground_truth)
        return bottlenecks, ground_truths


    def should_distort_images(self, flip_left_right, random_crop, random_scale,
                            random_brightness):
        return (flip_left_right or (random_crop != 0) or (random_scale != 0) or (random_brightness != 0))


    def add_input_distortions(self, flip_left_right, random_crop, random_scale,
                            random_brightness):

        jpeg_data = tf.placeholder(tf.string, name='DistortJPGInput')
        decoded_image = tf.image.decode_jpeg(jpeg_data, channels=self.MODEL_INPUT_DEPTH)
        decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
        decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
        margin_scale = 1.0 + (random_crop / 100.0)
        resize_scale = 1.0 + (random_scale / 100.0)
        margin_scale_value = tf.constant(margin_scale)
        resize_scale_value = tf.random_uniform(tensor_shape.scalar(),
                                                minval=1.0,
                                                maxval=resize_scale)
        scale_value = tf.mul(margin_scale_value, resize_scale_value)
        precrop_width = tf.mul(scale_value, self.MODEL_INPUT_WIDTH)
        precrop_height = tf.mul(scale_value, self.MODEL_INPUT_HEIGHT)
        precrop_shape = tf.pack([precrop_height, precrop_width])
        precrop_shape_as_int = tf.cast(precrop_shape, dtype=tf.int32)
        precropped_image = tf.image.resize_bilinear(decoded_image_4d,
                                                    precrop_shape_as_int)
        precropped_image_3d = tf.squeeze(precropped_image, squeeze_dims=[0])
        cropped_image = tf.random_crop(precropped_image_3d,
                                        [self.MODEL_INPUT_HEIGHT, self.MODEL_INPUT_WIDTH,
                                        self.MODEL_INPUT_DEPTH])
        if flip_left_right:
            flipped_image = tf.image.random_flip_left_right(cropped_image)
        else:
            flipped_image = cropped_image
        brightness_min = 1.0 - (random_brightness / 100.0)
        brightness_max = 1.0 + (random_brightness / 100.0)
        brightness_value = tf.random_uniform(tensor_shape.scalar(),
                                            minval=brightness_min,
                                            maxval=brightness_max)
        brightened_image = tf.mul(flipped_image, brightness_value)
        distort_result = tf.expand_dims(brightened_image, 0, name='DistortResult')
        return jpeg_data, distort_result


    def variable_summaries(self, var, name):
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean/' + name, mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev/' + name, stddev)
            tf.summary.scalar('max/' + name, tf.reduce_max(var))
            tf.summary.scalar('min/' + name, tf.reduce_min(var))
            tf.summary.histogram(name, var)


    def add_final_training_ops(self, class_count, final_tensor_name, bottleneck_tensor):
        with tf.name_scope('input'):
            bottleneck_input = tf.placeholder_with_default(
                bottleneck_tensor, shape=[None, self.BOTTLENECK_TENSOR_SIZE],
                name='BottleneckInputPlaceholder')

            ground_truth_input = tf.placeholder(tf.float32,
                                                [None, class_count],
                                                name='GroundTruthInput')

        layer_name = 'final_training_ops'
        with tf.name_scope(layer_name):
            with tf.name_scope('weights'):
                layer_weights = tf.Variable(tf.truncated_normal([self.BOTTLENECK_TENSOR_SIZE, class_count], stddev=0.001), name='final_weights')
                self.variable_summaries(layer_weights, layer_name + '/weights')
            with tf.name_scope('biases'):
                layer_biases = tf.Variable(tf.zeros([class_count]), name='final_biases')
                self.variable_summaries(layer_biases, layer_name + '/biases')
            with tf.name_scope('Wx_plus_b'):
                logits = tf.matmul(bottleneck_input, layer_weights) + layer_biases
                tf.summary.histogram(layer_name + '/pre_activations', logits)

        final_tensor = tf.nn.sigmoid(logits, name=final_tensor_name)
        tf.summary.histogram(final_tensor_name + '/activations', final_tensor)

        with tf.name_scope('cross_entropy'):
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits, labels=ground_truth_input)
            with tf.name_scope('total'):
                cross_entropy_mean = tf.reduce_mean(cross_entropy)
            tf.summary.scalar('cross entropy', cross_entropy_mean)

        with tf.name_scope('train'):
            train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(
                cross_entropy_mean)

        return (train_step, cross_entropy_mean, bottleneck_input, ground_truth_input,
                final_tensor)


    def add_evaluation_step(self, result_tensor, ground_truth_tensor):
        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                correct_prediction = tf.equal(tf.argmax(result_tensor, 1), \
                tf.argmax(ground_truth_tensor, 1))
            with tf.name_scope('accuracy'):
                evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('accuracy', evaluation_step)
        return evaluation_step


    def trainModel(self,classifier):
        
        if classifier == "Custom":
            
            trainingImageDir = CUSTOM_IMAGES_DIR
            trainingModelPath = CUSTOM_MODEL_PATH
            trainingLabelsPath = CUSTOM_MODEL_LABELS_PATH
            
        else:
            
            trainingImageDir = FACIAL_IMAGES_DIR
            trainingModelPath = FACIAL_MODEL_PATH
            trainingLabelsPath = FACIAL_MODEL_LABELS_PATH
        
        if tf.gfile.Exists(SUMMARIES_DIR):
            tf.gfile.DeleteRecursively(SUMMARIES_DIR)
        tf.gfile.MakeDirs(SUMMARIES_DIR)

        self.maybeDownload()
        graph, bottleneck_tensor, jpeg_data_tensor, resized_image_tensor = (
            self.create_inception_graph())
            
        image_lists = self.create_image_lists(trainingImageDir, TESTING_PERCENTAGE,
                                        VALIDATION_PERCENTAGE)
        class_count = len(image_lists.keys())
        if class_count == 0:
            print('No valid folders of images found at ' + trainingImageDir)
            return -1
        if class_count == 1:
            print('Only one valid folder of images found at ' + trainingImageDir +
                ' - multiple classes are needed for classification.')
            return -1

        do_distort_images = self.should_distort_images(
            FLIP_LEFT_RIGHT, RANDOM_CROP, RANDOM_SCALE,
            RANDOM_BRIGHTNESS)
        sess = tf.Session()

        if do_distort_images:
            
            distorted_jpeg_data_tensor, distorted_image_tensor = self.add_input_distortions(
                FLIP_LEFT_RIGHT, RANDOM_CROP, RANDOM_SCALE,
                RANDOM_BRIGHTNESS)
        else:
            
            self.cache_bottlenecks(sess, image_lists, trainingImageDir, BOTTLENECK_DIR,
                            jpeg_data_tensor, bottleneck_tensor)


        (train_step, cross_entropy, bottleneck_input, ground_truth_input,
        final_tensor) = self.add_final_training_ops(len(image_lists.keys()),
                                                FINAL_TENSOR_NAME,
                                                bottleneck_tensor)
                                                
        evaluation_step = self.add_evaluation_step(final_tensor, ground_truth_input)

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(SUMMARIES_DIR + '/train',
                                                sess.graph)
        validation_writer = tf.summary.FileWriter(SUMMARIES_DIR + '/validation')

        init = tf.initialize_all_variables()
        sess.run(init)

        for i in range(TRAINING_STEPS):
            if do_distort_images:
                train_bottlenecks, train_ground_truth = self.get_random_distorted_bottlenecks(
                sess, image_lists, TRAIN_BATCH_SIZE, 'training',
                trainingImageDir, distorted_jpeg_data_tensor,
                distorted_image_tensor, resized_image_tensor, bottleneck_tensor)
            else:
                train_bottlenecks, train_ground_truth = self.get_random_cached_bottlenecks(
                sess, image_lists, TRAIN_BATCH_SIZE, 'training',
                BOTTLENECK_DIR, trainingImageDir, jpeg_data_tensor,
                bottleneck_tensor)
                
            train_summary, _ = sess.run([merged, train_step],
                    feed_dict={bottleneck_input: train_bottlenecks,
                                ground_truth_input: train_ground_truth})
            train_writer.add_summary(train_summary, i)
            
            is_last_step = (i + 1 == TRAINING_STEPS)
            if (i % STEP_INTERVAL) == 0 or is_last_step:
                train_accuracy, cross_entropy_value = sess.run(
                [evaluation_step, cross_entropy],
                feed_dict={bottleneck_input: train_bottlenecks,
                            ground_truth_input: train_ground_truth})
            print('%s: Step %d: Train accuracy = %.1f%%' % (datetime.now(), i,
                                                            train_accuracy * 100))
            print('%s: Step %d: Cross entropy = %f' % (datetime.now(), i,
                                                        cross_entropy_value))
            validation_bottlenecks, validation_ground_truth = (
                self.get_random_cached_bottlenecks(
                    sess, image_lists, VALIDATION_BATCH_SIZE, 'validation',
                    BOTTLENECK_DIR, trainingImageDir, jpeg_data_tensor,
                    bottleneck_tensor))
                    
            validation_summary, validation_accuracy = sess.run(
                [merged, evaluation_step],
                feed_dict={bottleneck_input: validation_bottlenecks,
                            ground_truth_input: validation_ground_truth})
            validation_writer.add_summary(validation_summary, i)
            print('%s: Step %d: Validation accuracy = %.1f%%' %
                    (datetime.now(), i, validation_accuracy * 100))
                    
        test_bottlenecks, test_ground_truth = self.get_random_cached_bottlenecks(
            sess, image_lists, TEST_BATCH_SIZE, 'testing',
            BOTTLENECK_DIR, trainingImageDir, jpeg_data_tensor,
            bottleneck_tensor)
        test_accuracy = sess.run(
            evaluation_step,
            feed_dict={bottleneck_input: test_bottlenecks,
                        ground_truth_input: test_ground_truth})
        print('Final test accuracy = %.1f%%' % (test_accuracy * 100))

        output_graph_def = graph_util.convert_variables_to_constants(
            sess, graph.as_graph_def(), [FINAL_TENSOR_NAME])
        with gfile.FastGFile(trainingModelPath, 'wb') as f:
            f.write(output_graph_def.SerializeToString())
        with gfile.FastGFile(trainingLabelsPath, 'w') as f:
            f.write('\n'.join(image_lists.keys()) + '\n')
                
    def testModel(self,classifier):
        
        print("TESTING")
        print("")
        
        if classifier == "Custom":
            
            rootdir=os.getcwd()+"/model/testing/Custom/"
            thresh = "CUSTOM_THRESHOLD"
            
        elif classifier == "Facial":
            
            rootdir=os.getcwd()+"/model/testing/Facial/"
            thresh = "FACIAL_THRESHOLD"
            
        else:
            
            rootdir=os.getcwd()+"/model/testing/Objects/"
            thresh = "OBJECT_THRESHOLD"
            
        identified = 0
        
        for file in os.listdir(rootdir):
            
            if file.endswith(".jpg"):
                
                print("FILE: "+file)
                
                fileName = rootdir+"/"+file
                
                if classifier == "Custom":
                    
                    Object,Confidence = self.classifyCustom(fileName)
                    
                elif classifier == "Facial":
                    
                    Object,Confidence = self.classifyFace(fileName)
                    
                else:
                    
                    Object,Confidence = self.classifyObject(fileName)
                
                if Confidence > self.confs["ClassifierSettings"][thresh]:
                    
                    identified = identified + 1
                    
                    print("")
                    print("^^^IDENTIFIED^^^")
                    print("PROVIDED IMAGE: "+file)
                    print("DETECTED: "+str(Object))
                    print("CONFIDENCE: "+str(Confidence))
                    print("...")
                    print("")
                    
        print("COMPLETED TESTING")
        print(str(identified) + " IDENTIFIED")
        print("")