import numpy as np
import pandas as pd
import os
import csv
from pathlib import Path
import cv2

ids = pd.read_csv('./data/perfectly_detected_ears/annotations/recognition/ids.csv')
data_path = 'data/perfectly_detected_ears/'

# Sort each file in own folder
with open('data/perfectly_detected_ears/annotations/recognition/ids.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        set = row[0].split('/')[0]
        imag = row[0].split('/')[1]
        id = row[1]
        img_path = data_path + row[0]
        converted_path = data_path + 'converted/' + set + '/' + id
        img=cv2.imread(img_path)
        img_resized = cv2.resize(img, (200, 200), interpolation = cv2.INTER_LINEAR)
        # cv2.imshow('image',img_resized)
        # cv2.waitKey(0)

        # print(converted_path+'/'+img)

        if not os.path.exists(converted_path):
            os.makedirs(converted_path)
        print(converted_path+'/'+imag)
        cv2.imwrite(converted_path+'/'+imag, img_resized)
        #break
