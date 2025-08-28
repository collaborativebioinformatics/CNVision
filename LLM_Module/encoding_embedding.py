"""
This scripts provide examples of using DNABERT-2 (or any similar model like Nucleotide Transformers) to encode and embed DNA sequences.

The encoding is done by the tokenizer, and the embedding is done by the model.
The tokenizer will tokenize the DNA sequence into a list of tokens, and the model will embed the tokens into a vector space.

Input:
- DNA sequences, list[str], e.g. ["ATCG", "ATCG"]
- Model name, str, e.g. "zhihan1996/DNABERT-2-117M"
    supported models (I only tested the following ones but it should work for any models that are compatible with transformers):
        - "zhihan1996/DNABERT-2-117M"
        - ""

Output:
- Encoding vector:
    shape: (vocab_size)
    meaning: each element is the count of the corresponding token in the DNA sequence
- Embedding
    shape: (hidden_size)
    meaning: the embedding of the DNA sequence produced by the model

"""

import tqdm
import torch
import transformers
import numpy as np

def encode_sequence(sequence, model_name):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = transformers.AutoModel.from_pretrained(model_name, trust_remote_code=True)

    print(f"Producing {len(tokenizer)}-length encoding vector for {len(sequence)} sequences with {model_name}")

    # Tokenize the DNA sequence
    sequence_input_ids = tokenizer(sequence, add_special_tokens=False)["input_ids"]

    encodings = np.zeros((len(sequence), len(tokenizer)))
    for i,seq_ids in enumerate(sequence_input_ids):
        for token_id in seq_ids:
            encodings[i, token_id] += 1

    return encodings





def embed_sequence(dna_sequences, model_name_or_path, model_max_length=400, batch_size=20):
    # reorder the sequences by length
    lengths = [len(seq) for seq in dna_sequences]
    idx = np.argsort(lengths)
    dna_sequences = [dna_sequences[i] for i in idx]
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name_or_path,
            cache_dir=None,
            model_max_length=model_max_length,
            padding_side="right",
            use_fast=True,
            trust_remote_code=True,
        )

    is_hyenadna = "hyenadna" in model_name_or_path
    is_nt = "nucleotide-transformer" in model_name_or_path
    
    if is_nt:
        model = transformers.AutoModelForMaskedLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        ) 
    else:
        model = transformers.AutoModel.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
            )
    

    n_gpu = torch.cuda.device_count()
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
        
    model.to("cuda")


    train_loader = torch.utils.data.DataLoader(dna_sequences, batch_size=batch_size*n_gpu, shuffle=False, num_workers=2*n_gpu)
    for j, batch in enumerate(tqdm.tqdm(train_loader)):
        with torch.no_grad():
            token_feat = tokenizer.batch_encode_plus(
                    batch, 
                    max_length=model_max_length, 
                    return_tensors='pt', 
                    padding='longest', 
                    truncation=True
                )
            input_ids = token_feat['input_ids'].cuda()
            attention_mask = token_feat['attention_mask'].cuda()
            if is_hyenadna:
                model_output = model.forward(input_ids=input_ids)[0].detach().cpu()
            else:
                model_output = model.forward(input_ids=input_ids, attention_mask=attention_mask)[0].detach().cpu()
                
            attention_mask = attention_mask.unsqueeze(-1).detach().cpu()
            embedding = torch.sum(model_output*attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
            
            if j==0:
                embeddings = embedding
            else:
                
                embeddings = torch.cat((embeddings, embedding), dim=0)

    embeddings = np.array(embeddings.detach().cpu())
    
    # reorder the embeddings
    embeddings = embeddings[np.argsort(idx)]

    return embeddings


if __name__ == "__main__":
    dna_sequences = ["CAGTACGTACGATCGATCG", "CAGTCAGTCGATCGATCGATCG"]
    model_name = "zhihan1996/DNABERT-2-117M"
    encoding = encode_sequence(dna_sequences, model_name)
    print(encoding)

    embeddings = embed_sequence(dna_sequences, model_name)
    print(embeddings)

