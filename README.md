# CNVision

## Contributors
Sina Majidian (👑 Group Lead), Archit Kalra (👑 Group Lead), Ya Cui, Philippe Sanio, Muteeba Azhar, Thomas Krannich, Zhihan Zhou, Cliff Lun, Narges SangaraniPour, Jasmine Baker, Gerald McCollam, Kavya Vaddadi, Jinhao Wang, Fazel Amirvahedi, Hanan Wees

## Introduction
Our goal is to enhance copy number (CNV) calling and filtering using advanced machine learning techniques. We focus on leveraging bidirectional encoder models, such as **DNABERT-2**, to improve the classification of CNV breakpoints from a reference genome. For instance, Sniffles2 calls for a Genome in a Bottle (GIAB) benchmark dataset can be cross-referenced with CNV breakpoints on the GRCh38 reference genome. These breakpoints can then be used to derive sequences for training models to identify insertions, deletions, inversions, and false positives. We will also incorporate image-based classification of CNVs (using a similar technique to Xia et al. (2024)), utilizing DINOv3: Self-supervised models with downstream multi-class classification pipelines.

## Methods

<img width="1256" height="1462" alt="image" src="https://github.com/user-attachments/assets/06ff29ad-e052-480e-ae0d-50a13b41d93a" />



Figure 1. Methods Flowchart. Descriptive flow chart of methodology.  

## Relevant Papers
- [Smolka et al., 2024 - Detection of mosaic and population-level structural variants with Sniffles2](https://doi.org/10.1038/s41587-023-02024-y)
- [Xia et al., 2024 - CSV-Filter: a deep learning-based comprehensive structural variant filtering method](https://doi.org/10.1093/bioinformatics/btae539)
- [Zhou et al., 2024 - DNABERT-2: Efficient foundation model and benchmark for multi-species genome](https://iclr.cc/media/iclr-2024/Slides/17823.pdf)
- [Poplin et al., 2018 - A universal SNP and small-indel variant caller using deep neural networks](https://www.nature.com/articles/nbt.4235)
