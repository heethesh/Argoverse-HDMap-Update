#!/bin/bash

if [ $# -eq 4 ] 
then
    echo "Received required args"
else
    echo "Invalid args: arg1: Path to Data Folder, 
                        arg2: Mode (Options: mono, stereo), 
                        arg3: Descriptor type (Options: sift, surf, brisk, orb)
                        arg4: Top n matches"

    exit -1
fi

python3 sfm_main.py "$1" $2 $3 $4
