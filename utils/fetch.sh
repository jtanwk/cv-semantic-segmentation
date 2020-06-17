#!/bin/bash

# Clear out current results
rm -r results
mkdir results

# Fetch stdout log
scp jonathantan@linux.cs.uchicago.edu:~/cv/project/*.stdout .
scp jonathantan@linux.cs.uchicago.edu:~/cv/project/*.stderr .

# Fetch learning curves
scp jonathantan@linux.cs.uchicago.edu:~/cv/project/*.png .

# Fetch image chips and predictions
scp jonathantan@linux.cs.uchicago.edu:/scratch/jonathantan/results/* results
