# Module 3 – CNV Image Generation and Benchmarking (HG002 chr22)

## Data & Tools

We analyzed the **GIAB 2025.01 release (PAW70337)** from Oxford Nanopore Technologies  
([Detailed Data & Description Page](https://epi2me.nanoporetech.com/giab-2025.01/)),  
using the **HG002 Ashkenazi Trio son** dataset.  

- **Sample:** HG002 (Ashkenazi Trio son)  
- **Platform:** Oxford Nanopore **sup (super accuracy)** basecalling model  
- **Basecaller:** Dorado v0.8.2  
- **Reference genome:** GRCh38 (no alt analysis set)  
- **Variant callers:**  
  - **Sniffles2 v2.0.7** (SV/CNV detection)  
  - **Spectre v0.2.2** (long-read CNV detection)  
- **Benchmarking:** **Truvari v4.1.0** against the **GIAB HG002 SV truth set v0.6 (GRCh38 no-alt)**  

---

## Work in Progress – Channel Improvisation (Testing Logic)
To extend beyond the four CIGAR-based channels, we are experimenting with adding **coverage, MAPQ, and strand channels**:

- **Coverage (per-column read depth)**  
  - *Why:* CNVs are fundamentally depth events; depth shifts highlight copy number changes.  
  - *How:* Count reads covering each reference column, normalize 0–255, and broadcast to a square image.  

- **Mapping Quality (MAPQ)**  
  - *Why:* Low-MAPQ piles often occur near repeats and ambiguous regions; useful to down-weight noisy evidence.  
  - *How:* Paint each read’s aligned span with its MAPQ (scaled 0–255, capped at 60).  

- **Strand Orientation**  
  - *Why:* Strand imbalance and orientation flips frequently occur across CNV breakpoints.  
  - *How:* Paint read spans with 255 for reverse-strand, 0 for forward.  

These additions are to enhance the separability of true vs false CNVs. Future extensions may also include mismatch density and GC% tracks.

---


