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
			
				- ObjectTest: This mode sets the program to test object detection using images in the testing folder
				- ObjectCam: This mode sets the program to monitoring the camera for objects
				- FacialTest: This mode sets the program to test face detection using images in the testing folder
				- FacialTrain: This mode sets the program to to train using the provided training data
				- FacialCam: This mode sets the program to monitoring the camera for faces		
		"""
		
		self.confs = {}
		
		with open('data/confs.json') as confs:
			
			self.confs = json.loads(confs.read())
			
		self.startMQTT()
		self.InceptionFlow = InceptionFlow.InceptionFlow()
		self.InceptionFlow.checkModelDownload()
		self.InceptionFlow.createGraph(self.confs["ClassifierSettings"]["MODE"])
		
		if self.confs["ClassifierSettings"]["MODE"] == "ObjectCam" or self.confs["ClassifierSettings"]["MODE"] == "FacialCam":
        
			try:
    			
				self.OpenCVCapture = cv2.VideoCapture(self.confs["Cameras"][0]["URL"])

			except Exception as e:
				print("FAILED TO CONNECT TO WEBCAM")
				print(str(e))
				sys.exit()
		
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
		
	def objectTesting(self):
		
		print("TESTING OBJECTS")
		print("")
		rootdir=os.getcwd()+"/model/testing/Objects/"
		
		identified = 0

		for file in os.listdir(rootdir):
    			
			if file.endswith(".jpg"): 

				print("FILE: "+file)

				fileName = rootdir+"/"+file
				Object,Confidence = self.InceptionFlow.classifyObject(fileName)

				if Confidence > InceptionFlowCore.confs["ClassifierSettings"]["OBJECT_THRESHOLD"]:
    					
					identified = identified + 1
    					
					print("")
					print("^^^IDENTIFIED^^^")
					print("PROVIDED IMAGE: "+file)
					print("OBJECT DETECTED: "+str(Object))
					print("CONFIDENCE: "+str(Confidence))
					print("...")
					print("")

		print("COMPLETED TESTING OBJECTS")
		print(str(identified) + " IDENTIFIED OBJECTS")
		print("")
		
	def facialTesting(self):
		
		print("TESTING FACIAL REC")
		print("")
		rootdir=os.getcwd()+"/model/testing/Facial/"
		
		for file in os.listdir(rootdir):

			if file.endswith(".jpg"): 

				print("FILE: "+file)

				fileName = rootdir+"/"+file
				newPayload = cv2.imread(fileName,1)
				currentImage,inceptionImage,detected = self.InceptionFlow.captureAndDetect(newPayload)

				if detected is None:
					continue

				Label,Confidence = self.InceptionFlow.classifyFace(inceptionImage)

				if Confidence > self.confs["ClassifierSettings"]["FACIAL_THRESHOLD"]:
					
					print("")
					print("PROVIDED IMAGE: "+file)
					print("OBJECT DETECTED: "+str(Label))
					print("CONFIDENCE: "+str(Confidence))
					print("...")
					print("")

		print("COMPLETED TESTING FACIAL RECOGNITION")
		print("")
		
InceptionFlowCore = InceptionFlowCore()

while True:
    		
	if InceptionFlowCore.confs["ClassifierSettings"]["MODE"] == "FacialTrain":
		
		InceptionFlowCore.InceptionFlow.trainModel()
		InceptionFlowCore.Train=False
		
		print("TRAINING COMPLETED")
		print("")
    		
	elif InceptionFlowCore.confs["ClassifierSettings"]["MODE"] == "ObjectTest":	
		
			InceptionFlowCore.objectTesting()
			
			print("TESTING COMPLETED")
			print("")
    		
	elif InceptionFlowCore.confs["ClassifierSettings"]["MODE"] == "FacialTest":	
	
		InceptionFlowCore.facialTesting()
		
		print("TESTING COMPLETED")
		print("")
    		
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

				time.sleep(5)
			
			except cv2.error as e:
				print(e)     
				
		elif InceptionFlowCore.confs["ClassifierSettings"]["MODE"] == "FacialCam":	
    	
			try:
				
				ret, frame = InceptionFlowCore.OpenCVCapture.read()
				if not ret: continue
				
				currentImage,inceptionImage,detected = InceptionFlowCore.InceptionFlow.captureAndDetect(frame)

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

				time.sleep(5)
			
			except cv2.error as e:
				print(e) 

InceptionFlowCore.OpenCVCapture.release()
InceptionFlowCore.JumpWayMQTTClient.disconnectFromApplication()
