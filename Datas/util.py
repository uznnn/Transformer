# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 18:50:23 2022

@author: weizh
"""
import  sys,os
import numpy as np
def doc_path(doc_name):
    __file__ = sys.argv[0]
    __root__ = os.path.dirname(os.path.realpath(__file__))     #获得所在脚本路径
    __path__ = os.path.realpath(os.path.join(__root__,"{}".format(doc_name)))
    return __path__
