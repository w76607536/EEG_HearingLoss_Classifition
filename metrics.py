'''
Author: w76607536 wyb76607635@gmail.com
Date: 2022-05-16 23:39:14
LastEditors:  
LastEditTime: 2022-05-19 21:13:45
FilePath: \EEG_HearingLoss_Classifition\metrics.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# -*- coding: utf-8 -*-


'''
@Time    : 16/5/2021 4:04 下午
@Author  : bobo
@FileName: metrics.py
@Software: PyCharm
 
'''

from medpy import metric
import numpy as np
import  torch

def classification_accuracy(output, target):
    

    result = (output.argmax(dim=1) ==target).sum().item()/target.size(0)
    #print("acc:",output,output.argmax(dim=1),(output.argmax(dim=1) ==target).sum().item(),target,result)

    return  result 



