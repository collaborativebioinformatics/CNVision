
## Module 1 (data preprocessing)

### Downloading data

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

### Visualizing data

Use the `plot-svlen.R` to plot all SV lengths (>100nt) from a given VCF file. Requires `bcftools` and `Rscript` in your working environment.

Example command to plot deletions (DEL) from a given VCF file: 
```bash
bcftools query -i 'SVTYPE="DEL"' -f '%INFO/SVLEN\n' <input>.vcf.gz | awk '{print $1}' | Rscript plot-svlen.R <output-prefix>
```


