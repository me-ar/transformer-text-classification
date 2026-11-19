| model                       |   accuracy |   macro_f1 |    ece |   latency_ms_per_example |   train_seconds |   trainable_params |   ece_after_temp_scaling |   temperature |
|:----------------------------|-----------:|-----------:|-------:|-------------------------:|----------------:|-------------------:|-------------------------:|--------------:|
| TF-IDF + LogReg             |     0.9088 |     0.9089 | 0.1391 |                    0.038 |            28.8 |            2730420 |                 nan      |        nan    |
| MiniLM embeddings + LogReg  |     0.9269 |     0.9267 | 0.0752 |                    2.427 |            28   |              29568 |                 nan      |        nan    |
| DistilBERT (full fine-tune) |     0.9218 |     0.9218 | 0.0394 |                   13.995 |          2427.9 |           67012685 |                   0.0151 |          0.85 |
| DistilBERT + LoRA (r=8)     |     0.9153 |     0.9155 | 0.0117 |                   14.797 |          2445.1 |             797261 |                   0.0117 |          1.15 |
