

# Generate image from CIGAR string based on 
https://github.com/xcxw127/CSV-Filter/tree/main

```
        # Process Match operations (Channel 0)
        cigars_img = torch.zeros([1, len(r_start), ref_max - ref_min])
        for i, read in enumerate(sam_file.fetch(chromosome, begin, end)):
            max_terminal = read.reference_start - ref_min
            for operation, length in read.cigar:
                if operation == 0:  # Match
                    cigars_img[0, i, max_terminal:max_terminal+length] = 255
                    max_terminal += length
                elif operation == 2:  # Deletion
                    max_terminal += length
                elif operation == 3 or operation == 7 or operation == 8:  # N, =, X
                    max_terminal += length
        cigars_img1 = resize(cigars_img)
```

## Training metadata (image tensors under DIR)
```
python make_meta.py /path/to/DIR meta.csv
```

## Inference metadata
```
python make_meta_infer.py /path/to/DIR meta_infer.csv
```
## Finetune (ViT + LoRA)
```
python finetune_lora_vit.py \
  --csv meta.csv \
  --group_by event_id \
  --workers 2 \
  --batch 256
```
