# AIdea2020_NER

The NER system for AIcup2020 醫病訊息決策與對話語料分析競賽 - 秋季賽：醫病資料去識別化

#### Train/inference from single transformer model for NER
``` 
python NER_transformer.py 
```
#### Ensemble models from prob output
```
python transformer_ensemble.py 
```
#### Train/inference the sequence classification model for one of post-processing strategies
```
python privacy_classification.py 
```
#### Post-processing
```
python second_retrieval.py 
```
---------------------------------
**Detailed document(Chinese)**: https://drive.google.com/file/d/1j997zzBYe2Vn5gWGjWC_614zQPWHb7RY/view?usp=sharing
