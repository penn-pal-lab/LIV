#!/bin/bash

echo "Downloading LIV RealRobot Dataset!"
gdown --fuzzy  https://drive.google.com/file/d/1bRsokm0FNmHnfMgCzpiuhrhEHH2VON-s/view?usp=sharing
unzip liv_realrobot.zip && rm liv_realrobot.zip 
