# 20 news group classification using distil-bert

## Objective
The objective is to build classification models on 20 news group dataset. 
There are over 20k documents with each document representing one of the 20 news categories like 'alt.atheism', 'comp.graphics' etc.


In this repo, we show how to setup and run machine learning models on 20 news group data.

Notebooks overview:
1. exploratory_data_analysis.ipynb which performs EDA to get overview of the data.
2. traditional_ml_models.ipynb which has CBOW+logistic regression, 
tf-idf+logistic regression and tf-idf+SVM models.
3. distilbert_classification.ipynb consists of baseline distil-bert model and docbert implementations.


## Setup environment

This model requires PyTorch 1.6 or later. We recommend running the code in a virtual environment with Python 3.7:
```
virtualenv -p python3.7 venv
source venv/bin/activate
pip3 install -r requirements.txt
```

deactivate `source deactivate`


## Steps to run the model

**GPU with minimum 8GB RAM is required to run distil-bert model ** 

1. Clone this repo and setup the virtual environment.
2. Install required libraries.
3. Open jupyter notebook and run the notebooks.


## Run model on google colab
You can also run `distilbert_classification.ipynb` and `traditional_ml_models.ipynb` notebooks using google colab. 
To run models on google colab, Upload 'data.zip' present in git repository to colab Files, uncomment below code present in notebooks and run it.

```
# !pip install transformers
# !pip install pkbar
# !unzip data.zip
```

## Fine tuning
Fine tuning can be done according to the model requirements. 
Some hyperparameters which are used to train this model and can be explored further are:
```
max_seq_len = 512
do_lower_case = True
batch_size = 16
epochs = 13
warmup_proportion = 0.1
gradient_accumulation_steps = 1
learning_rate = 1e-5
```

## Results

The results are as follows:

| Model                          | F1    | accuracy |
|:------------------------------:|:-----:|:--------:|
| CBOW +Logistic regression      | 0.83  | 0.82     |
| Tf-idf + Logistic regression   | 0.79  | 0.79     | 
| Tf-idf + SVM                   | 0.74  | 0.73     |
| Distilbert baseline            | 0.834 | 0.831    |
| Docbert baseline               | 0.87  | 0.85     |
| Docbert after parameter tuning | <b>0.9</b> | <b>0.886</b>|


## References
1. [BERT paper](https://arxiv.org/abs/1810.04805)
2. Hugging face's pretrained pytorch models: [pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT)
3. [Docbert](https://arxiv.org/abs/1904.08398) 



# 2. Extract information from paystubs.

I discussed my approach to the problem in [report.pdf](https://github.com/suhas-chowdary/20newsgroup/blob/master/report.pdf).
