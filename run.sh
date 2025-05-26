#!/bin/bash

# for some reason need to install again groundingdino
# see here https://github.com/IDEA-Research/Grounded-SAM-2/issues/56#issuecomment-2471647093
python -m pip install --no-build-isolation -e grounding_dino

# run the demo: assumes /tmp has been mounted/shared with host
mkdir -p /tmp/test
python detection.py --video AICandidateTest-FINAL.mp4 --output /tmp/test