# ner-bert-crf
- Use `BERT+CRF` model to complete **NER(Named Entity Recognition)** task
  - for example, if you give a sentence like **"I am a student from China"**, the model will output the following result: **`['O','O','O','O','B-LOCATION','I-LOCATION','O']`**
  - if you give a sentence like **"I am Lisa from NewJersey"**, the model will output the following result: **`['O','O','B-PERSON','I-PERSON','O','B-LOCATION']`**


## Basic NER Models (4 models in total, check `model.py`)
1. BiLSTM+CRF: `TorchModel`
2. Bert + CRF: `BertCRFModel`
3. RegexExpression-only NER Model: `RegularExpressionModel`
   1. it only uses regular expression without any deeplearning methods to do the NER task.
4. Whole-Sentence NER Model (BERT+LSTM/GRU): `WholeSentenceNERModel`

---

## Additional models (you can try and play)
1. a self-defined tokenizer to replace the AutoTokenizer in BERT.
2. Bert+self-defined-CRF+**verterbi**
3. Bert+self-defined-CRF+**BeamSearch**


## Self-Defined-CRF
- we copy the CRF code from `torchcrf` package.
- we add a `decode_beam_search` function to the CRF model, which allows us to use the CRF model to do beam search decoding
- we add a `decode_viterbi` function to the CRF model, which allows us to use the CRF model to do viterbi decoding
- you can check the modified  CRF code in `crf.py`
  
---

## Important Project Components



---

## How to Run?
1. before running, fill your **local BERT model path** in `config.py`
2. you can also modify other hyperparameters as your need.
```python
Config = {
    "model_path": "model_output",
    "schema_path": "ner_data/schema.json",
    "train_data_path": "ner_data/train",
    "valid_data_path": "ner_data/test",
    "vocab_path":"chars.txt",
    "max_length": 100,
    "hidden_size": 256,
    "num_layers": 2,
    "epoch": 5,
    "batch_size": 16,
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "use_crf": True,
    "class_num": 9,
    "bert_path": r"D:\pre-trained-models\bert-base-chinese",
    ...
}
```

```shell
pip install -r requirements.txt
```

1. run `main.py` 
2. in `main.py` :
   1. the `main()` function is used to train only 1 model, and see performance
   2. the `batch_train()` function is used to train all the models and see all the performance of NER.
   3. you can check `metrics.csv` for all the models' performance 

---

## Project Structure
- in `model.py`, you can see there is a ModelHub that allows you to select 1 of the 4 models. you can do the selection in `main()` function from `main.py`.

![image](https://github.com/user-attachments/assets/ac416b36-0b80-44a2-89aa-3af5a084aa33)

![image](https://github.com/user-attachments/assets/31774bf1-59ba-4123-9793-a423f333b9fb)


## BERT
![image](https://github.com/user-attachments/assets/3720173b-90ae-4c3b-813a-ea0f90443f58)

![image](https://github.com/user-attachments/assets/521360a1-a08a-440f-b74d-82060360ce92)


## CRF
![image](https://github.com/user-attachments/assets/b07c5576-1c44-4a3c-9a9e-6946bb725a00)

![image](https://github.com/user-attachments/assets/db1539bc-650e-4502-8f33-8740911ea399)




## Model Test Results
### CRF + BiLSTM output
![image](https://github.com/user-attachments/assets/67ce9f4a-2bba-4a79-b2c2-f5b17326c5bf)




### RegexNERModel output
![image](https://github.com/user-attachments/assets/2926ef88-507d-4a78-ade6-5ac8cee16da0)


### Bert + CRF output
![image](https://github.com/user-attachments/assets/190bd57a-e7d0-4e88-b9e5-0283224a0d74)



### Whole Sentence NER output
![image](https://github.com/user-attachments/assets/80a1cb7d-450f-45e4-aef1-ccbd22bff750)

---

## Training Results
1. you can check the complete training metrics in `metrics.csv`
![image](https://github.com/user-attachments/assets/aaa94f18-cc29-4910-a6c3-4699178f1ae4)


---

## Performance Comparison
| Model | Avg-Precision | Avg-Recall |   Macro-F1   |  Micro-F1 |  
|----------|----------|----------|----------|------------|
| BiLSTM+CRF |0.634867 | 0.489829 |  0.548224   | 0.657138 |
| Bert+CRF | 0.416666 | 0.229813 | 0.282876 | 0.406775 |
| Regex-Only |     |    |         |     |
| BERT+RNN |    |    |         |       |
