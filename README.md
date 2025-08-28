# CNVision

## Contributors
Sina Majidian (👑 Group Lead), Archit Kalra (👑 Group Lead), Ya Cui, Philippe Sanio, Muteeba Azhar, Thomas Krannich, Zhihan Zhou, Cliff Lun, Narges SangaraniPour, Jasmine Baker, Gerald McCollam, Kavya Vaddadi, Jinhao Wang, Fazel Amirvahedi, Hanan Wees

## Introduction
Our goal is to enhance copy number (CNV) calling and filtering using advanced machine learning techniques. We focus on leveraging bidirectional encoder models, such as **DNABERT-2**, to improve the classification of CNV breakpoints from a reference genome. For instance, Sniffles2 calls for a Genome in a Bottle (GIAB) benchmark dataset can be cross-referenced with CNV breakpoints on the GRCh38 reference genome. These breakpoints can then be used to derive sequences for training models to identify insertions, deletions, inversions, and false positives. We will also incorporate image-based classification of CNVs (using a similar technique to Xia et al. (2024)), utilizing DINOv3: Self-supervised models with downstream multi-class classification pipelines.

## Methods

<img width="1256" height="1462" alt="image" src="https://github.com/user-attachments/assets/06ff29ad-e052-480e-ae0d-50a13b41d93a" />


Figure 1. Methods Flowchart. Descriptive flow chart of methodology.  





### Build environment
```
# create and activate virtual python environment
conda create -n dna python=3.8
conda activate dna

# install required packages
pip install -r requirements.txt
pip uninstall triton
````



## Module 1 (data preprocessing)

Downloading data

```
#HG002_sup_PAW70337_based_alignment_and_call_files: 
aws s3 cp s3://ont-open-data/giab_2025.01/analysis/wf-human-variation/sup/HG002/PAW70337/output/SAMPLE.haplotagged.cram   ./ --no-sign-request
aws s3 cp s3://ont-open-data/giab_2025.01/analysis/wf-human-variation/sup/HG002/PAW70337/output/SAMPLE.haplotagged.cram.crai ./ --no-sign-request
echo "done downloading samples-cram files"



#reference fasta - GRCh38
wget ftp://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/000/001/405/GCA_000001405.15_GRCh38/seqs_for_alignment_pipelines.ucsc_ids/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna.gz
echo "done downloading cram specific-fasta file"
gunzip GCA_000001405.15_GRCh38_no_alt_analysis_set.fna.gz 
samtools faidx GCA_000001405.15_GRCh38_no_alt_analysis_set.fna
echo "done indexing fasta"
```



```
# CNV (Spectre)
aws s3 cp s3://ont-open-data/giab_2025.01/analysis/wf-human-variation/sup/HG002/PAW70337/output/SAMPLE.wf_cnv.vcf.gz   ./ --no-sign-request
aws s3 cp s3://ont-open-data/giab_2025.01/analysis/wf-human-variation/sup/HG002/PAW70337/output/SAMPLE.wf_cnv.vcf.gz.tbi ./ --no-sign-request
echo "done downloading Spectre VCF files"
```
(We are generating CNVs from the newest version of software.)




## Module 2 (LLM )

### Scripts for Genome Foundation Model encoding, embedding and fine-tuning

Here are two scripts:
1. encoding_embedding.py that provide examples of using DNABERT-2 (or any similar model like Nucleotide Transformers) to encode and embed DNA sequences.
2. finetune.py that allow finetuning of DNABERT-2 and similar models on a sequence classfication or regression dataset.


```
dna_sequences = ["CAGTACGTACGATCGATCG", "CAGTCAGTCGATCGATCGATCG"]
model_name = "zhihan1996/DNABERT-2-117M"
encoding = encode_sequence(dna_sequences, model_name)
print(encoding)
```


```
python finetune.py \
    --model_name_or_path $model_name \
    --data_path  $data_path \
    --kmer -1 \
    --run_name DNABERT2_run \
    --model_max_length $max_length \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --learning_rate 3e-5 \
    --num_train_epochs 3 \
    --fp16 \
    --save_steps 200 \
    --output_dir output/dnabert2 \
    --evaluation_strategy steps \
    --eval_steps 200 \
    --warmup_ratio 0.1 \
    --logging_steps 100000 \
    --find_unused_parameters False
```



## Module 3 ( Image Encoding)

BAM data preprocess from CSV-Filter
```
python bam2depth.py
```




## Module 4 ( Image Embeddings )

Self-supervised learning for vision at unprecedented scale

https://ai.meta.com/dinov3/




## Relevant Papers
- [Smolka et al., 2024 - Detection of mosaic and population-level structural variants with Sniffles2](https://doi.org/10.1038/s41587-023-02024-y)
- [Xia et al., 2024 - CSV-Filter: a deep learning-based comprehensive structural variant filtering method](https://doi.org/10.1093/bioinformatics/btae539)
- [Zhou et al., 2024 - DNABERT-2: Efficient foundation model and benchmark for multi-species genome](https://iclr.cc/media/iclr-2024/Slides/17823.pdf)
- [Poplin et al., 2018 - A universal SNP and small-indel variant caller using deep neural networks](https://www.nature.com/articles/nbt.4235)
