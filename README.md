# CNVision

<img width="1256" height="1462" alt="image" src="main/CNV.png" />

## Contributors
Sina Majidian (👑 Group Lead), Archit Kalra (👑 Group Lead), Philippe Sanio, Muteeba Azhar, Thomas Krannich, Zhihan Zhou, Narges SangaraniPour, Jasmine Baker, Gerald McCollam, Kavya Vaddadi, Jinhao Wang, Fazel Amirvahedi, Hanan Wees

## Introduction
Our goal is to enhance copy number (CNV) calling and filtering using advanced machine learning techniques. We focus on leveraging bidirectional encoder models, such as **DNABERT-2**, to improve the classification of CNV breakpoints from a reference genome, as well as image-based techniques for CNV classification. For instance, Sniffles2 calls for a Genome in a Bottle (GIAB) benchmark dataset can be cross-referenced with CNV breakpoints on the GRCh38 reference genome. These breakpoints can then be used to derive sequences for training models to identify insertions, deletions, inversions, and false positives. We will also incorporate image-based classification of CNVs (using a similar technique to Xia et al. (2024)), utilizing DINOv3: Self-supervised models with downstream multi-class classification pipelines (Siméoni et al., 2025).

## Methods

The benchmarked human genome datasets HG002 and HG008 from the Genome in a Bottle (GIAB) consortium was used. The input data included long-read sequencing datasets generated using PacBio HiFi and Oxford Nanopore ultra-long (ONT-UL) platforms (McDaniel et al., 2025). Reads were aligned against GRCh38 and T2T-CHM13 reference assemblies using standard long read alignment strategies to obtain high-quality BAM files. CNV were called using Sniffles2 (v2.0.7) and Spectre (v0.2.2) which represent state-of-the-art CNV caller optimized for long read sequencing, supplemented with optional third CNV caller for cross validation (De Clercq et al., 2024) (Smolka et al., 2024). Breakpoints from called CNVs were extracted and used as the basis for both sequence-based and image-based downstream analysis. We extracted flanking regions from each predicted CNV breakpoint to capture specific sequence signature. The genomic segments were tokenized and encoded using DNABERT-2, a transformer based language model trained on large scale genomic data. We fine tuned DNABERT-2 embeddings for CNV classification tasks. Training used the GIAB HG002/HG008 benchmark truth sets and clinically validated CNV datasets with a train/test split to enable robust model fine-tuning and evaluation (Consens et al., 2025). 

In parallel with sequence embedding, we generated image-based representations of CNVs from aligned read data. Using CSV-filter and NPSV-deep, we encoded read pileups into tensor-like images that reflect allele balance, sequencing depth and zygosity status (e.g., homozygous reference [0/0] vs heterozygous variant [0/1]) (Xia et al., 2024). The resulting image tensors were processed using DINOv3, a self-supervised vision transformer model developed for large-scale image representation learning. DINOv3 produces latent image embeddings that capture structural features distinguishing true CNVs from sequencing artifacts (Siméoni et al., 2025). As with sequence embeddings, supervised fine-tuning was conducted against GIAB truth sets and clinical CNVs using an identical training/test split strategy. Both the DNABERT-2 (sequence-based) and DINOv3 (image based) models were fine-tumed independently. Outputs were integrated into a filtering framework designed to classify CNV candidates as true or false positives. Finally, a filtering and evaluation framework was implemented to integrate outputs from both models. Model performance was assessed by comparison against truth sets (GIAB and clinical CNVs), false positive reduction efficiency, precision and recall metrics which provide a balanced measure of classification accuracy in CNV detection. This multi model approach, integrating sequence language models with vision transformers provides a robust and scalable framework for accurate CNV detection in long read sequencing data. 

<img width="1256" height="1462" alt="image" src="https://github.com/user-attachments/assets/06ff29ad-e052-480e-ae0d-50a13b41d93a" />

<div align="center" style="text-align: center"> <b> Figure </b>: Methods Flowchart. Descriptive flow chart of methodology.   </div>






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

Data generation for the pipeline.
<img align="center" style="text-align: center" width="1000" alt="image" src="Data_Module/datageneration.png" />

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
#converting the CRAM file to BAM file to solve file issues
samtools view -@ 12 -T <GCA_000001405.15_GRCh38_no_alt_analysis_set.fa> -b -o SAMPLE.haplotagged.bam SAMPLE.haplotagged.cram
samtools index SAMPLE.haplotagged.bam
```


Deriving large SVs using Sniffles:
```
# Running Sniffles 2.6.3
sniffles --input SAMPLE.haplotagged.bam --vcf HG002/sniffles_out/sniffles.vcf.gz --threads 12 --snf HG002/sniffles_out/sniffels.snf
```
Deriving large CNVs using Spectre
```
# CNV (Spectre)
aws s3 cp s3://ont-open-data/giab_2025.01/analysis/wf-human-variation/sup/HG002/PAW70337/output/SAMPLE.wf_cnv.vcf.gz   ./ --no-sign-request
aws s3 cp s3://ont-open-data/giab_2025.01/analysis/wf-human-variation/sup/HG002/PAW70337/output/SAMPLE.wf_cnv.vcf.gz.tbi ./ --no-sign-request
echo "done downloading Spectre VCF files"

# Running Spectre
spectre CNVCaller --coverage cov/coverage.regions.bed.gz --sample-id HG002 --output-dir ./spectre_out --reference GCA_000001405.15_GRCh38_no_alt_analysis_set.fna.gz --blacklist grch38_blacklist_spectre.bed  --ploidy-chr chrX:1 --min-cnv-len 25000 --metadata spectre_out/metadata.mdr
```


### Visualizing data

Use the `plot-svlen.R` to plot all SV lengths (>100nt) from a given VCF file. Requires `bcftools` and `Rscript` in your working environment.

Example command to plot deletions (DEL) from a given VCF file: 
```bash
bcftools query -i 'SVTYPE="DEL"' -f '%INFO/SVLEN\n' <input>.vcf.gz | awk '{print $1}' | Rscript plot-svlen.R <output-prefix>
```


<img align="center" style="text-align: center" width="800" alt="image" src="Data_Module/del-length-histo.png" />



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
python Image_Encoding_Module/scripts/gen_RGB_images.py --input_dir tensors_data --output_dir images_data
```
</br>

For each candidate CNV:
- All reads are gathered overlapping the variant locus within a ±500 bp window.
- Alignment patterns are extracted from CIGAR strings.
- 224×224 pixel images are directly generated with four separate channels, where each channel captures specific CIGAR alignment operations: matches, deletions, insertions and soft-clips.
- To enable compatibility with image models, these 4-channel images are converted to 3-channel RGB format using a grayscale encoding where Match=1, Deletion=2, Insertion=3, and Soft-clip=4, then replicating this grayscale across all three RGB channels. 
</br>

<div align="center">

<img width="1000" alt="image" src="Image_Encoding_Module/Screenshot%201404-06-07%20at%2018.37.22.png" />
</div>
<div align="center" style="text-align: center"> <b> Figure </b>: Image generation from CIGAR strings as developed by Xia et al. (2024). Alignment patterns are extracted from CIGAR strings and developed into RGB images for convolutional neural network training. </div>

## Image Generation Workflow

Candidate CNVs were converted into image representations for deep learning analysis:

1. **Read extraction:** Reads overlapping ±500 bp of each CNV breakpoint were collected from haplotagged CRAM/BAM files (restricted to chr22).  
2. **CIGAR parsing:** Alignment operations (matches, deletions, insertions, soft-clips) were extracted.  
3. **Image encoding:**  
   - 224×224 grayscale tensors with four channels (M, D, I, S).  
   - Converted into RGB format via grayscale encoding:  
     - Match = 1, Deletion = 2, Insertion = 3, Soft-clip = 4  
     (replicated across RGB channels).  
4. **Output:** RGB images (PNG) for downstream CNN and vision-transformer models.  

---

## Benchmarking Results

From the chr22 subset of HG002, we generated a total of **427 RGB CNV images**:  
- **250 true positives (TPs)** (128 insertions, 122 deletions)  
- **177 false positives (FPs)** (50 insertions, 127 deletions)  

---

## Figures

We highlight representative examples of both true and false positive CNV predictions.  
These illustrate the alignment patterns around CNV loci and the visual separability of signal vs noise.

<p align="center">
  <img src="Image_Encoding_Module/figures/HG002_GRCh38_TP_chr22_20075554_20076402_INS_848bp.png" alt="TP Insertion (848 bp)" width="45%">
  <img src="Image_Encoding_Module/figures/HG002_GRCh38_TP_chr22_17289461_17298218_DEL_8757bp.png" alt="TP Deletion (8.7 kb)" width="45%">
</p>
<p align="center">
  <img src="Image_Encoding_Module/figures/HG002_GRCh38_FP_chr22_19907696_19907774_INS_78bp.png" alt="FP Insertion (78 bp)" width="45%">
  <img src="Image_Encoding_Module/figures/HG002_GRCh38_FP_chr22_23941517_23943818_DEL_2301bp.png" alt="FP Deletion (2.3 kb)" width="45%">
</p>

**Figure 1.** Examples of CNV pileup images. Top row: true positives (TP-INS, TP-DEL). Bottom row: false positives (FP-INS, FP-DEL).

---



#### Image Embedding from NPSV-deep 

<img align="center" style="text-align: center" width="1000" alt="image" src="Image_Embedding_Module/variant0.png" />

<img align="center" style="text-align: center" width="1000" alt="image" src="Image_Embedding_Module/variant2.png" />



<p align="center">
  <b>Image embeddings from NPSV-deep (Fazel Amirvahedi)</b><br>
  Encodes nine input channels used for structural variant classification:<br>
  1. Read allele assignment &nbsp;
  2. Haplotag &nbsp;
  3. Base quality &nbsp;
  4. Strand &nbsp;
  5. Mapping quality &nbsp;
  6. Fragment allele assignment &nbsp;
  7. Alternate insert size &nbsp;
  8. Reference insert size &nbsp;
  9. Aligned bases (matched or soft-clipped)
</p>



## Module 4 ( Image Embeddings )

Self-supervised learning for vision at unprecedented scale

The images are embedded using DINOv3, a self-supervised vision transformer, which is also fine-tuned on a train/test split to enhance classification accuracy.
#### Training metadata (image tensors under DIR)
```
python make_meta.py /path/to/DIR meta.csv
```

#### Inference metadata
```
python make_meta_infer.py /path/to/DIR meta_infer.csv
```
#### Finetune (ViT + LoRA)
```
python finetune_lora_vit.py \
  --csv meta.csv \
  --group_by event_id \
  --workers 2 \
  --batch 256
```

</br>
<img align="center" style="text-align: center" width="1000" alt="image" src="Image_Embedding_Module/dinov3.png" />
</br>
<div align="center" style="text-align: center"> <b> Figure </b>: DINOv3 data curation and model development stages. </div>

</br>


## Training loss function
```
"epoch": 0, "train_loss": 0.789
"epoch": 1, "train_loss": 0.407
"epoch": 2, "train_loss": 0.376
"epoch": 3, "train_loss": 0.386
"epoch": 4, "train_loss": 0.373
"epoch": 5, "train_loss": 0.371
"epoch": 6, "train_loss": 0.388
"epoch": 7, "train_loss": 0.374
"epoch": 8, "train_loss": 0.391
"epoch": 9, "train_loss": 0.367


Newest epoch:  train_loss: 0.231
```




## Module 5 ( Filtering and Evaluation )


Each sub-command contains help documentation. Start with `truvari -h` to see available commands.

The current most common Truvari use case is for structural variation benchmarking:
```
  truvari bench -b base.vcf.gz -c comp.vcf.gz -f reference.fa -o output_dir/
```

Find more matches by harmonizing phased variants using refine:
```
   truvari refine output_dir/
```

Use Truvari's comparison engine to consolidate redundant variants in a merged multi-sample VCF:
```
    bcftools merge -m none sampleA.vcf.gz sampleB.vcf.gz | bgzip > merge.vcf.gz
    tabix merge.vcf.gz
    truvari collapse -i merge.vcf.gz -o truvari_merge.vcf
```


## Relevant Papers
- [Smolka et al., 2024 - Detection of mosaic and population-level structural variants with Sniffles2](https://doi.org/10.1038/s41587-023-02024-y)
- [Xia et al., 2024 - CSV-Filter: a deep learning-based comprehensive structural variant filtering method](https://doi.org/10.1093/bioinformatics/btae539)
- [Zhou et al., 2024 - DNABERT-2: Efficient foundation model and benchmark for multi-species genome](https://iclr.cc/media/iclr-2024/Slides/17823.pdf)
- [Poplin et al., 2018 - A universal SNP and small-indel variant caller using deep neural networks](https://www.nature.com/articles/nbt.4235)
- [Siméoni, O., Vo, H. V., Seitzer, M., Baldassarre, F., Oquab, M., Jose, C., ... & Bojanowski, P. (2025). DINOv3.](https://ai.meta.com/research/publications/dinov3/)


<img width="1352" height="878" alt="Team" src="https://github.com/user-attachments/assets/f9951778-e819-4a98-b38b-4280d15ab21d" />
(+ Narges, Zhihan, Jasmine, Gerald)


