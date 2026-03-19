Fine-tune gemma3:27b based on labeled data

1. Add source to dataloader.py -> in load function
Add to the list the location of the folder containing the new data as well as eventually preprocessing necessary to get 
the data as a list of `{'processed': anonimized-text, 'text': original text}` entries

2. LORA weight generation + saving model 
```python finetune_gemma3_27b.py```

3. Quantize model using Q5_K_M
```/llama.cpp/build/bin/llama-quantize gemma3-v2-medical-anonymizer-f16.gguf gemma3-27b-v2_q5_k_m.gguf Q5_K_M```

4. Create model - will already include the prompt based on the Modelfile
```ollama create gemma3-v2-anon:latest -f Modelfile_gemma27b```

-------
Old commands that might be useful in the future
3python llama.cpp/convert_hf_to_gguf.py gemma3-v2-merged --outfile gemma3-v2-merged.gguf --outtype f16