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

import argparse
import os.path
import re
import sys
import tarfile
import json

from datetime import datetime

import numpy as np
from six.moves import urllib
import tensorflow as tf
import cv2
        
OBJECT_MODEL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz' 
OBJECT_MODEL_DIR = os.getcwd()+"/model/imagenet"   
OBJECT_MODEL_PATH = os.getcwd()+"/model/imagenet/classify_image_graph_def.pb"
OBJECT_MODEL_LABELS_PATH = os.getcwd()+"/model/imagenet/imagenet_2012_challenge_label_map_proto.pbtxt" 
OBJECT_MODEL_LABELSH_PATH = os.getcwd()+"/model/imagenet/imagenet_synset_to_human_label_map.txt"  

FACIAL_MODEL_PATH = os.getcwd()+"/model/InceptionFlow.pb"
FACIAL_MODEL_LABELS = os.getcwd()+"/model/InceptionFlow.txt"

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
        
        with open('data/confs.json') as confs:
            
            self.confs = json.loads(confs.read())
            
    def saveImage(self,frame):
        
        timeDirectory = os.path.dirname(os.path.abspath(__file__))+"/images/"+datetime.now().strftime('%Y-%m-%d')+'/'+datetime.now().strftime('%H')
        
        if not os.path.exists(timeDirectory):
            os.makedirs(timeDirectory)

        currentImage=timeDirectory+'/'+datetime.now().strftime('%M-%S')+'.jpg'
        print(currentImage)
        print("")
        
        cv2.imwrite(currentImage, frame)

        return currentImage
            
    def createGraph(self,graphType="Object"):
        
        if graphType == "Object":
            
            print("Creating Default Object Graph")

            with tf.gfile.FastGFile(OBJECT_MODEL_PATH, 'rb') as f:

                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                _ = tf.import_graph_def(graph_def, name='') 
        
        elif graphType == "Facial":
            
            print("Creating Facial Graph")

            with tf.gfile.FastGFile(FACIAL_MODEL_PATH, 'rb') as f:

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

            filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
            print()
            statinfo = os.stat(filepath)
            print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

        tarfile.open(filepath, 'r:gz').extractall(dest_directory)