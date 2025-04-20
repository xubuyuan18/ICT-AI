import sys
import cv2
import numpy 
import os
import time
import matplotlib.pyplot as plt 
import acl

import acllite_utils as ut 
import constants as constants
from acllite_module import aclliteModule
from acllite_resource import resource_list


#定义类
class ACL:
    def __init__(self,device_id=0):
        self.device_id = device_id
        self.context = None
        self.srteam = None
        self.run_mode = None

    def init(self):
        "初始化资源"
        print("")
        ret  = acl.init()
        