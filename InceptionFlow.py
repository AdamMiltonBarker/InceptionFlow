# *****************************************************************************
# InceptionFlow
# Copyright (c) 2018 Adam Milton-Barker - AdamMiltonBarker.com
# Based on Google's Tensorflow Imagenet Inception V3
# Uses TechBubble IoT JumpWay MQTT Client
# *****************************************************************************

import sys
import os
import time
import json

from datetime import datetime

import techbubbleiotjumpwaymqtt.device as IoTJumpWayDevice
import techbubbleiotjumpwaymqtt.application as IoTJumpWayApplication
import InceptionFlow
import cv2

class InceptionFlowCore():
    	
	def __init__(self):
    		
		"""	
			CLASSIFIER MODE:
			
			Classifier configuration can be found in data/confs.json
			
				- CustomCam: This mode sets the program to monitoring the camera for custom objects
				- CustomTrain: This mode sets the program to to train using the provided custom training data
				- CustomTest: This mode sets the program to test custom object detection using the provided custom testing data
				- FacialCam: This mode sets the program to monitoring the camera for faces
				- FacialTrain: This mode sets the program to to train using the provided faces training data
				- FacialTest: This mode sets the program to test custom object detection using the provided custom testing data
				- ObjectCam: This mode sets the program to monitoring the camera for objects
				- ObjectTest: This mode sets the program to test object detection using images in the objects testing folder
		"""
		
		self.confs = {}
		
		with open('data/confs.json') as confs:
			
			self.confs = json.loads(confs.read())
			
		self.startMQTT()
		self.InceptionFlow = InceptionFlow.InceptionFlow()
		self.InceptionFlow.checkModelDownload()
		
		if self.confs["ClassifierSettings"]["MODE"] == "ObjectCam" or self.confs["ClassifierSettings"]["MODE"] == "FacialCam" or self.confs["ClassifierSettings"]["MODE"] == "CustomCam":
		
			self.InceptionFlow.createGraph(self.confs["ClassifierSettings"]["MODE"])
        
			try:
    			
				self.OpenCVCapture = cv2.VideoCapture(self.confs["Cameras"][0]["URL"])

			except Exception as e:
				print("FAILED TO CONNECT TO WEBCAM")
				print(str(e))
				sys.exit()
		
		elif self.confs["ClassifierSettings"]["MODE"] == "ObjectTest" or self.confs["ClassifierSettings"]["MODE"] == "FacialTest" or self.confs["ClassifierSettings"]["MODE"] == "CustomTest":
		
			self.InceptionFlow.createGraph(self.confs["ClassifierSettings"]["MODE"])
		
	def startMQTT(self):
        
		try:

			self.JumpWayMQTTClient = IoTJumpWayApplication.JumpWayPythonMQTTApplicationConnection({
				"locationID": self.confs["IoTJumpWaySettings"]["SystemLocation"],
				"applicationID": self.confs["IoTJumpWaySettings"]["SystemApplicationID"],
				"applicationName": self.confs["IoTJumpWaySettings"]["SystemApplicationName"],
				"username": self.confs["IoTJumpWayMQTTSettings"]["MQTTUsername"],
				"password": self.confs["IoTJumpWayMQTTSettings"]["MQTTPassword"]
			})

		except Exception as e:
			print(str(e))
			sys.exit()

		self.JumpWayMQTTClient.connectToApplication()
		
InceptionFlowCore = InceptionFlowCore()

while True:
    
    if InceptionFlowCore.confs["ClassifierSettings"]["MODE"] == "CustomTrain":
        
        InceptionFlowCore.InceptionFlow.trainModel("Custom")
        
        print("TRAINING COMPLETED")
        print("")
        
        sys.exit(0)
        
    elif InceptionFlowCore.confs["ClassifierSettings"]["MODE"] == "FacialTrain":
        
        InceptionFlowCore.InceptionFlow.trainModel("Facial")
        
        print("TRAINING COMPLETED")
        print("")
        
        sys.exit(0)
        
    elif InceptionFlowCore.confs["ClassifierSettings"]["MODE"] == "ObjectTest":
        
        InceptionFlowCore.InceptionFlow.testModel("Object")
        
        print("TESTING COMPLETED")
        print("")
        
        sys.exit(0)
        
    elif InceptionFlowCore.confs["ClassifierSettings"]["MODE"] == "CustomTest":
        
        InceptionFlowCore.InceptionFlow.testModel("Custom")
        
        print("TESTING COMPLETED")
        print("")
        
        sys.exit(0)
        
    elif InceptionFlowCore.confs["ClassifierSettings"]["MODE"] == "FacialTest":
        
        InceptionFlowCore.InceptionFlow.testModel("Facial")
        
        print("TESTING COMPLETED")
        print("")
        
        sys.exit(0)
        
    else:
        
        if InceptionFlowCore.confs["ClassifierSettings"]["MODE"] == "ObjectCam":
            
            try:
                
                ret, frame = InceptionFlowCore.OpenCVCapture.read()
                if not ret: continue
                    
                savedFrame = InceptionFlowCore.InceptionFlow.saveImage(frame)
                Object,Confidence = InceptionFlowCore.InceptionFlow.classifyObject(savedFrame)
                
                if Confidence > InceptionFlowCore.confs["ClassifierSettings"]["OBJECT_THRESHOLD"]:
                    
                    print("Object: "+str(Object))
                    print("Confidence: "+str(Confidence))
                    print("")
                    
                    InceptionFlowCore.JumpWayMQTTClient.publishToDeviceChannel(
							"Sensors",
							InceptionFlowCore.confs["IoTJumpWaySettings"]["SystemZone"],
							InceptionFlowCore.confs["IoTJumpWaySettings"]["SystemDeviceID"],
							{
								"Sensor":"CCTV",
								"SensorID": InceptionFlowCore.confs["Cameras"][0]["ID"],
								"SensorValue":"OBJECT: " + str(Object) + " (Confidence: " + str(Confidence) + ")"
							}
						)
                        
                else:
                    
                    print("")
                    print("NOTHING IDENTIFIED")
                    print("")
                    
                cv2.imshow("InceptionFlow Viewer", frame)
                if cv2.waitKey(1) == 27:
                    break
                
            except cv2.error as e:
                print(e)
                
        elif InceptionFlowCore.confs["ClassifierSettings"]["MODE"] == "FacialCam":
            
            try:
                
                ret, frame = InceptionFlowCore.OpenCVCapture.read()
                if not ret: continue
                    
                currentImage,inceptionImage, marked, detected = InceptionFlowCore.InceptionFlow.captureAndDetect(frame)
                
                if detected is None:
                    continue
                    
                Object,Confidence = InceptionFlowCore.InceptionFlow.classifyFace(inceptionImage)
                
                if Confidence > InceptionFlowCore.confs["ClassifierSettings"]["FACIAL_THRESHOLD"]:
                    
                    print("Person: "+str(Object))
                    print("Confidence: "+str(Confidence))
                    print("")
                    
                    InceptionFlowCore.JumpWayMQTTClient.publishToDeviceChannel(
							"Sensors",
							InceptionFlowCore.confs["IoTJumpWaySettings"]["SystemZone"],
							InceptionFlowCore.confs["IoTJumpWaySettings"]["SystemDeviceID"],
							{
								"Sensor":"CCTV",
								"SensorID": InceptionFlowCore.confs["Cameras"][0]["ID"],
								"SensorValue":"PERSON: " + str(Object) + " (Confidence: " + str(Confidence) + ")"
							}
						)
                        
                else:
                    
                    print("")
                    print("NOTHING IDENTIFIED")
                    print("")
                    
                cv2.imshow("InceptionFlow Viewer", marked)
                if cv2.waitKey(1) == 27:
                    break
                
            except cv2.error as e:
                print(e)
                
InceptionFlowCore.OpenCVCapture.release()
InceptionFlowCore.JumpWayMQTTClient.disconnectFromApplication()