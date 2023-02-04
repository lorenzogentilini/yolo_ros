#!/usr/bin/env python3
import sys

import rospy as rp
import numpy as np
import math as mt
import torch
import cv2

from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseArray, Pose

class YoloWrapper:
    def __init__(self):
        self.ready = False
        self.executing = False
        self.processed = False
        self.img = Image()

        self.CVDepthToNumpy = {cv2.CV_8U: 'uint8', cv2.CV_8S: 'int8', cv2.CV_16U: 'uint16',
                               cv2.CV_16S: 'int16', cv2.CV_32S:'int32', cv2.CV_32F:'float32',
                               cv2.CV_64F: 'float64'}
        self.NumpyToCVType  = {'uint8': '8U', 'int8': '8S', 'uint16': '16U',
                               'int16': '16S', 'int32': '32S', 'float32': '32F',
                               'float64': '64F'}
        # Subscribers
        self.imageSub = rp.Subscriber("/axis/image_raw_out", Image, self.imageCallback, queue_size=1)

        # Publishers
        self.detectionImgPub = rp.Publisher('/yolo_detection', Image, queue_size=1)
        self.bbCentrePub = rp.Publisher('/obj_detected', PoseArray, queue_size=1)

        # Model
        self.modelFile = rp.get_param('yolo_ros_node/weightsfile', './yoloWeights.pt')
        print(self.modelFile)

        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.modelFile)
        self.ready = True
        print("Model Loaded!")
        
    def imageCallback(self, msg):
        if not self.executing:
            self.img = self.msgToCVImg(msg)
            self.processed = False

    def execute(self):
        if not self.ready:
            return
        
        if self.processed:
            return
        
        self.executing = True

        # Run Model
        results = self.model(self.img)
        detected = results.xyxy[0].cpu().numpy()

        objs = PoseArray()
        for npresults in detected:
            if len(npresults) != 0:
                print(npresults)

                start_point = (int(npresults[0]), int(npresults[1]))
                end_point = (int(npresults[2]), int(npresults[3]))
                cv2.rectangle(self.img, start_point, end_point, (255,0,0), 2)
                self.detectionImgPub.publish(self.CVImgToMsg(self.img, "bgr8"))

                obj = Pose()
                obj.position.x = (npresults[0] + npresults[2])/2
                obj.position.y = (npresults[1] + npresults[3])/2
                objs.poses.append(obj)

        if len(objs.poses) != 0:
            self.bbCentrePub.publish(objs)

        self.executing = False
        self.processed = True

    def getTorchImage(self, surce_img):
        img = torch.from_numpy(np.array(surce_img)).permute(2, 0, 1).float()
        return img[None].cuda()

    #### MSG/CV Conversion ################################
    def CVImgToMsg(self, img, enc = "passthrough"):
        msg = Image()
        msg.height = img.shape[0]
        msg.width = img.shape[1]

        if len(img.shape) < 3:
            cv_type = self.fromDTypeToCVType(img.dtype, 1)
        else:
            cv_type = self.fromDTypeToCVType(img.dtype, img.shape[2])

        if enc == "passthrough":
            msg.encoding = cv_type
        else:
            msg.encoding = enc

        if img.dtype.byteorder == '>':
            msg.is_bigendian = True

        msg.data = img.tostring()
        msg.step = len(msg.data) // msg.height

        return msg

    def msgToCVImg(self, msg):
        dtype, n_channels = self.fromEcodingToDType(msg.encoding)
        dtype = np.dtype(dtype)
        dtype = dtype.newbyteorder('>' if msg.is_bigendian else '<')

        if n_channels == 1:
            im = np.ndarray(shape=(msg.height, msg.width), dtype=dtype, buffer=msg.data)
        else:
            if(type(msg.data) == str):
                im = np.ndarray(shape=(msg.height, msg.width, n_channels), dtype=dtype, buffer=msg.data.encode())
            else:
                im = np.ndarray(shape=(msg.height, msg.width, n_channels), dtype=dtype, buffer=msg.data)

        # Chech Byteorder
        if msg.is_bigendian == (sys.byteorder == 'little'):
            im = im.byteswap().newbyteorder()
        
        return im

    def fromDTypeToCVType(self, dtype, channels):
        return '%sC%d' % (self.NumpyToCVType[dtype.name], channels)

    def fromEcodingToDType(self, enc):
        return self.fromCVTypeToDType(self.fromEncodingToCVType(enc))

    def fromCVTypeToDType(self, cvtype):
        return self.CVDepthToNumpy[self.CvMatDepth(cvtype)], self.CvMatChannels(cvtype)

    def CvMatDepth(self, flag):
        return flag & ((1 << 3) - 1)

    def CvMatChannels(self, flag):
        return ((flag & (511 << 3)) >> 3) + 1

    def fromEncodingToCVType(self, enc):
        if(enc == "bgr8"):
            return cv2.CV_8UC3
        if(enc == "mono8"):
            return cv2.CV_8UC1
        if(enc == "rgb8"):
            return cv2.CV_8UC3
        if(enc == "mono16"):
            return cv2.CV_16UC1
        if(enc == "bgr16"):
            return cv2.CV_16UC3
        if(enc == "rgb16"):
            return cv2.CV_16UC3
        if(enc == "bgra8"):
            return cv2.CV_8UC4
        if(enc == "rgba8"):
            return cv2.CV_8UC4
        if(enc == "bgra16"):
            return cv2.CV_16UC4
        if(enc == "rgba16"):
            return cv2.CV_16UC4
        if(enc == "bayer_rggb8"):
            return cv2.CV_8UC1
        if(enc == "bayer_bggr8"):
            return cv2.CV_8UC1
        if(enc == "bayer_gbrg8"):
            return cv2.CV_8UC1
        if(enc == "bayer_grbg8"):
            return cv2.CV_8UC1
        if(enc == "bayer_rggb16"):
            return cv2.CV_16UC1
        if(enc == "bayer_bggr16"):
            return cv2.CV_16UC1
        if(enc == "bayer_gbrg16"):
            return cv2.CV_16UC1
        if(enc == "bayer_grbg16"):
            return cv2.CV_16UC1
        if(enc == "yuv422"):
            return cv2.CV_8UC2
    #### END ################################
    
# Main Function
if __name__ == '__main__':
    rp.init_node('yolo_ros_node', anonymous = False)
    
    # Initialize Object
    YoloRos = YoloWrapper()

    # Spin
    while not rp.is_shutdown():
        YoloRos.execute()
