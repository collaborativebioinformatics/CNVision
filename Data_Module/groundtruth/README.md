# Ground truth SV data sets

### HG002 w.r.t. T2T

- CHM13v2.0_HG2-T2TQ100-V1.1_stvar.vcf.gz
- CHM13v2.0_HG2-T2TQ100-V1.1_stvar.vcf.gz.tbi
- CHM13v2.0_HG2-T2TQ100-V1.1_stvar.onlyDEL.vcf.gz
- CHM13v2.0_HG2-T2TQ100-V1.1_stvar.onlyDEL.vcf.gz.tbi

### HG002 w.r.t. GRCh38

- GRCh38_HG2-T2TQ100-V1.1_stvar.vcf.gz
- GRCh38_HG2-T2TQ100-V1.1_stvar.vcf.gz.tbi
- GRCh38_HG2-T2TQ100-V1.1_stvar.onlyDEL.vcf.gz
- GRCh38_HG2-T2TQ100-V1.1_stvar.onlyDEL.vcf.gz.tbi

# Commands

DEL filtered files were generated via

```bash
bcftools view -i 'INFO/SVTYPE="DEL"' <input>.vcf.gz -Oz -o <input>.onlyDEL.vcf.gz
tabix -p vcf <input>.onlyDEL.vcf.gz
```

# Sources

NCBI

```
https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/data/AshkenazimTrio/analysis/NIST_HG002_DraftBenchmark_defrabbV0.019-20241113/
```

# Notes

- Mind that the current DEL filtered subsets do not contain DUPs. It requires a slightly more elaborate filtering, e.g. getting all variants with `TRFdiff>=1`.