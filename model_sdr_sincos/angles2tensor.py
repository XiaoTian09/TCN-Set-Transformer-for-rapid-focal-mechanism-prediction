# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 17:59:12 2024

@author: tianx
"""

import numpy as np

def angles2tensor(strike, dip, rake):
    '''
    将震源机制的角度参数转换为震源机制矩阵
    '''
    # 计算震源机制矩阵的分量
    strike=strike*np.pi/180;
    dip=dip*np.pi/180;
    rake=rake*np.pi/180;

    m11 = - ( np.sin(dip)*np.cos(rake)*np.sin(2*strike) + np.sin(2*dip)*np.sin(rake)*np.sin(strike)*np.sin(strike) ) ;
    m22 = + ( np.sin(dip)*np.cos(rake)*np.sin(2*strike) - np.sin(2*dip)*np.sin(rake)*np.cos(strike)*np.cos(strike) ) ;
    m33 = + np.sin(2*dip)*np.sin(rake);
    m12 = + (np.sin(dip)*np.cos(rake)*np.cos(2*strike) + np.sin(2*dip)*np.sin(rake)*np.sin(2*strike)/2) ;
    m13 =  - (np.cos(dip)*np.cos(rake)*np.cos(strike) + np.cos(2*dip)*np.sin(rake)*np.sin(strike));
    m23 =  - (np.cos(dip)*np.cos(rake)*np.sin(strike) - np.cos(2*dip)*np.sin(rake)*np.cos(strike));

    M = np.array([[m11, m12, m13], [m12, m22, m23], [m13, m23, m33]]);
    return M