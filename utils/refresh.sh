#!/bin/bash

# Dataloader class
scp data/loader.py jonathantan@linux.cs.uchicago.edu:~/cv/data

# Classifiers
scp nets/zoomout.py jonathantan@linux.cs.uchicago.edu:~/cv/nets
scp nets/classifier.py jonathantan@linux.cs.uchicago.edu:~/cv/nets

# Main .py files
scp *.py jonathantan@linux.cs.uchicago.edu:~/cv
