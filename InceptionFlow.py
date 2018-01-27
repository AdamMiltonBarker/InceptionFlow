# *****************************************************************************
# InceptionFlow
# Copyright (c) 2018 Adam Milton-Barker - AdamMiltonBarker.com
# Based on Google's Tensorflow Imagenet Inception V3
# Uses TechBubble IoT JumpWay MQTT Client
# *****************************************************************************

import sys
import os
import time
import cv2
import json
import techbubbleiotjumpwaymqtt.device
import InceptionFlow
from datetime import datetime

class InceptionFlowCore():
    	
	def __init__(self):
    		
		self.Train=False
		self.Test=True
    		
		self.confs = {}
		self.threshold = 0.75

		self.InceptionFlow = InceptionFlow.InceptionFlow()
		
		with open('data/confs.json') as confs:
			
			self.confs = json.loads(confs.read())
			
		self.startMQTT()
		self.InceptionFlow.checkModelDownload()
		self.InceptionFlow.createGraph("Object")
		
	def startMQTT(self):
        
		try:

			self.JumpWayMQTTClient = techbubbleiotjumpwaymqtt.device.JumpWayPythonMQTTDeviceConnection({
				"locationID": self.confs["IoTJumpWaySettings"]["SystemLocation"],
				"zoneID": self.confs["IoTJumpWaySettings"]["SystemZone"],
				"deviceId": self.confs["IoTJumpWaySettings"]["SystemDeviceID"],
				"deviceName": self.confs["IoTJumpWaySettings"]["SystemDeviceName"],
				"username": self.confs["IoTJumpWayMQTTSettings"]["MQTTUsername"],
				"password": self.confs["IoTJumpWayMQTTSettings"]["MQTTPassword"]
			})

		except Exception as e:
			print(str(e))
			sys.exit()

		self.JumpWayMQTTClient.connectToDevice()
		
	def testing(self):
		
		print("TESTING OBJECTS")
		print("")
		rootdir=os.getcwd()+"/model/testing/Objects/"
		
		for file in os.listdir(rootdir):

			if file.endswith(".jpg"): 

				print("FILE: "+file)

				fileName = rootdir+"/"+file
				label,confidence = self.InceptionFlow.classifyObject(fileName)
				
				if label:
					
					print("IMAGE: "+file)
					print("OBJECT: "+str(label))
					print("Confidence: "+str(confidence))
					print("")

		print("")
		print("COMPLETING TESTING OBJECTS")
		
InceptionFlowCore = InceptionFlowCore()

while True:
    	
	if InceptionFlowCore.Train==True:
    		
		pass
    		
	elif InceptionFlowCore.Test==True:
    		
		InceptionFlowCore.testing()
		InceptionFlowCore.Test=False
		print("TESTING DEACTIVATED")
    		
	else:
    		
		pass