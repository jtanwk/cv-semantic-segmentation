# cv-semantic-segmentation

## Directory Structure

```
.
├── data: holds definition for dataloader class
├── features: holds intermediate features (e.g. embeddings)
├── nets: holds definitions for neural net classifiers
├── run.sh: script submitted to SBATCH on server
├── sampling.py
├── train_cls.py
├── train_seg.py
└── utils: other useful preprocessing scripts
```

## How to run

Run `sampling.py` to extract intermediate features from images.

Run `train_cls.py` to train the fully-connected classifier on the intermediate features.

Run `train_seg.py` to load pre-trained weights from the fully-connected classifier and use them to generate semantic segmentation predictions.
