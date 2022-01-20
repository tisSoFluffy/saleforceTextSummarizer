# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 17:33:03 2022

@author: joshu
"""

from text_processor import *
import pathlib

files = pathlib.Path('data')

titles = []
text = []

for file in files.iterdir():
    titles.append(file.stem)
    with open(file, 'r',encoding = 'utf-8') as f:
        text.append(f.read())
    
        