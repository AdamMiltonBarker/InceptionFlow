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
import struct

from datetime import datetime

import numpy as np
from six.moves import urllib
import tensorflow as tf
import cv2

from tensorflow.contrib.quantize.python import quant_ops
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat
        
OBJECT_MODEL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz' 
OBJECT_MODEL_DIR = os.getcwd()+"/model/imagenet"   
OBJECT_MODEL_PATH = os.getcwd()+"/model/imagenet/classify_image_graph_def.pb"
OBJECT_MODEL_LABELS_PATH = os.getcwd()+"/model/imagenet/imagenet_2012_challenge_label_map_proto.pbtxt" 
OBJECT_MODEL_LABELSH_PATH = os.getcwd()+"/model/imagenet/imagenet_synset_to_human_label_map.txt"  

FACIAL_MODEL_DIR = os.getcwd()+"/model"
FACIAL_MODEL_PATH = os.getcwd()+"/model/InceptionFlow.pb"
FACIAL_MODEL_LABELS_PATH = os.getcwd()+"/model/InceptionFlow.txt"

BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
BOTTLENECK_TENSOR_SIZE = 2048
MODEL_INPUT_WIDTH = 299
MODEL_INPUT_HEIGHT = 299
MODEL_INPUT_DEPTH = 3
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear:0'
MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1 
NUM_TOP_PREDICTIONS = 5

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
        self.bottleneck_path_2_bottleneck_values = {}
        
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
            return currentImage, inceptionImage, faces[0]

        else:

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame,datetime.now().strftime("%Y-%m-%d %H:%M"),(10,450), font, 1,(255,255,255),2)
            cv2.putText(frame,"InceptionFlow - AdamMiltonBarker.com",(10,40), font, 1,(255,255,255),2)

            print("Found " + str(len(faces)))
            return currentImage, None, None  

    def resize(self,image):

            return cv2.resize(image,(self.confs["ClassifierSettings"]["INCEPTION_SIZE"], self.confs["ClassifierSettings"]["INCEPTION_SIZE"]),interpolation=cv2.INTER_LANCZOS4)

    def crop(self,image, x, y, w, h):
        crop_height = int((self.confs["ClassifierSettings"]["INCEPTION_SIZE"] / float(self.confs["ClassifierSettings"]["INCEPTION_SIZE"]))*w)
        midy = y + h/2
        y1 = max(0, midy-crop_height/2)
        y2 = min(image.shape[0]-1, midy+crop_height/2)
        return image[int(y1):int(y2), x:x+w]
            
    def createGraph(self,graphType="ObjectTest"):
    
        if graphType == "ObjectTest" or graphType == "ObjectCam":
            
            print("Creating Default Object Graph")

            with tf.gfile.FastGFile(OBJECT_MODEL_PATH, 'rb') as f:

                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                _ = tf.import_graph_def(graph_def, name='') 
        
        elif graphType == "FacialTest" or graphType == "FacialCam":
            
            if os.path.exists(FACIAL_MODEL_PATH):
                
                print("Creating Facial Graph")

                with tf.gfile.FastGFile(FACIAL_MODEL_PATH, 'rb') as f:

                    graph_def = tf.GraphDef()
                    graph_def.ParseFromString(f.read())
                    _ = tf.import_graph_def(graph_def, name='') 
            
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

    def checkModelDownload(self):
        
        """Download and extract model tar file."""
        
        dest_directory = OBJECT_MODEL_DIR
        
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
        
        """
        Builds a list of training images from the file system.
        
        Analyzes the sub folders in the image directory, splits them into stable
        training, testing, and validation sets, and returns a data structure
        describing the lists of images for each label and their paths.
        
        Args:
            image_dir: String path to a folder containing subfolders of images.
            testing_percentage: Integer percentage of the images to reserve for tests.
            validation_percentage: Integer percentage of images reserved for validation.
        
        Returns:
            A dictionary containing an entry for each label subfolder, with images split
            into training, testing, and validation sets within each label.
        """
        
        if not gfile.Exists(image_dir):
            
            tf.logging.error("Image directory '" + image_dir + "' not found.")
            return None
        
        result = {}
        sub_dirs = [x[0] for x in gfile.Walk(image_dir)]
        
        # The root directory comes first, so skip it.
        is_root_dir = True
        
        for sub_dir in sub_dirs:
            
            if is_root_dir:
                
                is_root_dir = False
                continue
            
            extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
            file_list = []
            
            dir_name = os.path.basename(sub_dir)

            if dir_name == image_dir:
                continue
            
            tf.logging.info("Looking for images in '" + dir_name + "'")
            
            for extension in extensions:
                
                file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
                file_list.extend(gfile.Glob(file_glob))
                
            if not file_list:
                
                tf.logging.warning('No files found')
                continue
            
            if len(file_list) < 20:
                
                tf.logging.warning(
                    'WARNING: Folder has less than 20 images, which may cause issues.')
                    
            elif len(file_list) > MAX_NUM_IMAGES_PER_CLASS:
                
                tf.logging.warning(
                    'WARNING: Folder {} has more than {} images. Some images will '
                    'never be selected.'.format(dir_name, MAX_NUM_IMAGES_PER_CLASS))
                    
            label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
            training_images = []
            testing_images = []
            validation_images = []
            
            for file_name in file_list:
                
                base_name = os.path.basename(file_name)
                # We want to ignore anything after '_nohash_' in the file name when
                # deciding which set to put an image in, the data set creator has a way of
                # grouping photos that are close variations of each other. For example
                # this is used in the plant disease data set to group multiple pictures of
                # the same leaf.
                hash_name = re.sub(r'_nohash_.*$', '', file_name)
                # This looks a bit magical, but we need to decide whether this file should
                # go into the training, testing, or validation sets, and we want to keep
                # existing files in the same set even if more files are subsequently
                # added.
                # To do that, we need a stable way of deciding based on just the file name
                # itself, so we do a hash of that and then use that to generate a
                # probability value that we use to assign it.
                hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
                percentage_hash = ((int(hash_name_hashed, 16) %
                          (MAX_NUM_IMAGES_PER_CLASS + 1)) *
                         (100.0 / MAX_NUM_IMAGES_PER_CLASS))
                         
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
        
        """"
        Returns a path to an image for a label at the given index.
        Args:
            image_lists: Dictionary of training images for each label.
            label_name: Label string we want to get an image for.
            index: Int offset of the image we want. This will be moduloed by the
            available number of images for the label, so it can be arbitrarily large.
            image_dir: Root folder string of the subfolders containing the training
            images.
            category: Name string of set to pull images from - training, testing, or
            validation.
        Returns:
            File system path string to an image that meets the requested parameters.
        """
        
        if label_name not in image_lists:
            
            tf.logging.fatal('Label does not exist %s.', label_name)
            
        label_lists = image_lists[label_name]
        
        if category not in label_lists:
            
            tf.logging.fatal('Category does not exist %s.', category)
            category_list = label_lists[category]
            
        if not category_list:

            tf.logging.fatal(
                'Label %s has no images in the category %s.',
                label_name, 
                category)

        mod_index = index % len(category_list)
        base_name = category_list[mod_index]
        sub_dir = label_lists['dir']
        full_path = os.path.join(image_dir, sub_dir, base_name)
        return full_path

    def get_bottleneck_path(self, image_lists, label_name, index, bottleneck_dir, category, architecture):
        
        """"
        Returns a path to a bottleneck file for a label at the given index.
        Args:
            image_lists: Dictionary of training images for each label.
            label_name: Label string we want to get an image for.
            index: Integer offset of the image we want. This will be moduloed by the
            available number of images for the label, so it can be arbitrarily large.
            bottleneck_dir: Folder string holding cached files of bottleneck values.
            category: Name string of set to pull images from - training, testing, or
            validation.
            architecture: The name of the model architecture.
        Returns:
            File system path string to an image that meets the requested parameters.
        """
        return get_image_path(
            image_lists, 
            label_name, 
            index, 
            bottleneck_dir,
            category) + '_' + architecture + '.txt'
            
    def create_model_graph(self, model_info):
        
        """"
        Creates a graph from saved GraphDef file and returns a Graph object.
        Args:
            model_info: Dictionary containing information about the model architecture.
        Returns:
            Graph holding the trained Inception network, and various tensors we'll be
            manipulating.
        """
        
        with tf.Graph().as_default() as graph:
            
            model_path = os.path.join(FACIAL_MODEL_DIR, model_info['model_file_name'])
            print('Model path: ', model_path)
            
            with gfile.FastGFile(model_path, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                bottleneck_tensor, resized_input_tensor = (
                    tf.import_graph_def(
                        graph_def,
                        name='',
                        return_elements=[
                            model_info['bottleneck_tensor_name'],
                            model_info['resized_input_tensor_name'],
                ]))
                
        return graph, bottleneck_tensor, resized_input_tensor

    def run_bottleneck_on_image(self, sess, image_data, image_data_tensor,
                            decoded_image_tensor, resized_input_tensor,
                            bottleneck_tensor):
                            
        """
        Runs inference on an image to extract the 'bottleneck' summary layer.
        Args:
            sess: Current active TensorFlow Session.
            image_data: String of raw JPEG data.
            image_data_tensor: Input data layer in the graph.
            decoded_image_tensor: Output of initial image resizing and preprocessing.
            resized_input_tensor: The input node of the recognition graph.
            bottleneck_tensor: Layer before the final softmax.
        Returns:
            Numpy array of bottleneck values.
        """
        # First decode the JPEG image, resize it, and rescale the pixel values.
        resized_input_values = sess.run(
            decoded_image_tensor,
             {image_data_tensor: image_data})
             
        # Then run it through the recognition network.
        bottleneck_values = sess.run(
            bottleneck_tensor,
            {resized_input_tensor: resized_input_values})
            
        bottleneck_values = np.squeeze(bottleneck_values)
        return bottleneck_values

    def ensure_dir_exists(self, dir_name):
        
        """Makes sure the folder exists on disk.
        Args:
            dir_name: Path string to the folder we want to create.
        """
        
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

    def create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                           image_dir, category, sess, jpeg_data_tensor,
                           decoded_image_tensor, resized_input_tensor,
                           bottleneck_tensor):
                           
        """Create a single bottleneck file."""
        tf.logging.info('Creating bottleneck at ' + bottleneck_path)
        image_path = self.get_image_path(
            image_lists, 
            label_name, 
            index,
            image_dir, 
            category)
            
        if not gfile.Exists(image_path):
            
            tf.logging.fatal('File does not exist %s', image_path)
            
        image_data = gfile.FastGFile(image_path, 'rb').read()
        
        try:
            
            bottleneck_values = self.run_bottleneck_on_image(
                sess, 
                image_data, 
                jpeg_data_tensor, 
                decoded_image_tensor,
                resized_input_tensor,
                bottleneck_tensor)
                
        except Exception as e:
            
            raise RuntimeError('Error during processing file %s (%s)' % (image_path,str(e)))
            
        bottleneck_string = ','.join(str(x) for x in bottleneck_values)

        with open(bottleneck_path, 'w') as bottleneck_file:
            
            bottleneck_file.write(bottleneck_string)

    def get_or_create_bottleneck(sess, image_lists, label_name, index, image_dir,
                             category, bottleneck_dir, jpeg_data_tensor,
                             decoded_image_tensor, resized_input_tensor,
                             bottleneck_tensor, architecture):
                             
        """
        Retrieves or calculates bottleneck values for an image.
        If a cached version of the bottleneck data exists on-disk, return that,
        otherwise calculate the data and save it to disk for future use.
        Args:
            sess: The current active TensorFlow Session.
            image_lists: Dictionary of training images for each label.
            label_name: Label string we want to get an image for.
            index: Integer offset of the image we want. This will be modulo-ed by the
            available number of images for the label, so it can be arbitrarily large.
            image_dir: Root folder string of the subfolders containing the training
            images.
            category: Name string of which set to pull images from - training, testing,
            or validation.
            bottleneck_dir: Folder string holding cached files of bottleneck values.
            jpeg_data_tensor: The tensor to feed loaded jpeg data into.
            decoded_image_tensor: The output of decoding and resizing the image.
            resized_input_tensor: The input node of the recognition graph.
            bottleneck_tensor: The output tensor for the bottleneck values.
            architecture: The name of the model architecture.
        Returns:
            Numpy array of values produced by the bottleneck layer for the image.
        """
        label_lists = image_lists[label_name]
        sub_dir = label_lists['dir']
        sub_dir_path = os.path.join(bottleneck_dir, sub_dir)
        self.ensure_dir_exists(sub_dir_path)
        bottleneck_path = self.get_bottleneck_path(
            image_lists, 
            label_name, 
            index,
            bottleneck_dir,
            category,
            architecture)
            
        if not os.path.exists(bottleneck_path):
            
            self.create_bottleneck_file(
                bottleneck_path, 
                image_lists,
                label_name,
                index,
                image_dir,
                category,
                sess,
                jpeg_data_tensor,
                decoded_image_tensor,
                resized_input_tensor,
                bottleneck_tensor)
                
        with open(bottleneck_path, 'r') as bottleneck_file:
            
            bottleneck_string = bottleneck_file.read()
        did_hit_error = False
        
        try:
            
            bottleneck_values = [float(x) for x in bottleneck_string.split(',')]

        except ValueError:
            
            tf.logging.warning('Invalid float found, recreating bottleneck')
            did_hit_error = True
            
        if did_hit_error:
            
            self.create_bottleneck_file(
                bottleneck_path, 
                image_lists, 
                label_name, 
                index,
                image_dir,
                category,
                sess,
                jpeg_data_tensor,
                decoded_image_tensor,
                resized_input_tensor,
                bottleneck_tensor)
                
            with open(bottleneck_path, 'r') as bottleneck_file:
                
                bottleneck_string = bottleneck_file.read()

            # Allow exceptions to propagate here, since they shouldn't happen after a fresh creation
            bottleneck_values = [float(x) for x in bottleneck_string.split(',')]

        return bottleneck_values

    def cache_bottlenecks(sess, image_lists, image_dir, bottleneck_dir,
                      jpeg_data_tensor, decoded_image_tensor,
                      resized_input_tensor, bottleneck_tensor, architecture):
                      
        """
        Ensures all the training, testing, and validation bottlenecks are cached.
        Because we're likely to read the same image multiple times (if there are no
        distortions applied during training) it can speed things up a lot if we
        calculate the bottleneck layer values once for each image during
        preprocessing, and then just read those cached values repeatedly during
        training. Here we go through all the images we've found, calculate those
        values, and save them off.
        Args:
            sess: The current active TensorFlow Session.
            image_lists: Dictionary of training images for each label.
            image_dir: Root folder string of the subfolders containing the training
            images.
            bottleneck_dir: Folder string holding cached files of bottleneck values.
            jpeg_data_tensor: Input tensor for jpeg data from file.
            decoded_image_tensor: The output of decoding and resizing the image.
            resized_input_tensor: The input node of the recognition graph.
            bottleneck_tensor: The penultimate output layer of the graph.
            architecture: The name of the model architecture.
        Returns:
            Nothing.
        """
        
        how_many_bottlenecks = 0
        self.ensure_dir_exists(bottleneck_dir)
        
        for label_name, label_lists in image_lists.items():
            
            for category in ['training', 'testing', 'validation']:
                
                category_list = label_lists[category]
                
                for index, unused_base_name in enumerate(category_list):
                    
                    self.get_or_create_bottleneck(
                        sess, 
                        image_lists, 
                        label_name, 
                        index, 
                        image_dir,
                        category,
                        bottleneck_dir, 
                        jpeg_data_tensor, 
                        decoded_image_tensor,
                        resized_input_tensor, 
                        bottleneck_tensor, 
                        architecture)
                        
                    how_many_bottlenecks += 1
                    if how_many_bottlenecks % 100 == 0:
                        
                        tf.logging.info(
                            str(how_many_bottlenecks) + ' bottleneck files created.')