## nlp-tutorial

<p align="center"><img width="100" src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/11/TensorFlowLogo.svg/225px-TensorFlowLogo.svg.png" />  <img width="100" src="https://media-thumbs.golden.com/OLqzmrmwAzY1P7Sl29k2T9WjJdM=/200x200/smart/golden-storage-production.s3.amazonaws.com/topic_images/e08914afa10a4179893eeb07cb5e4713.png" /></p>

`nlp-tutorial` is a tutorial of NLP(Natural Language Processing) using **Pytorch**. Most of the models in NLP were implemented with less than **100 lines** of code.(except comments or blank lines)


## Curriculum - (Example Purpose)

#### 1. Basic Embedding Model

- 1-1. [NNLM(Neural Network Language Model)](1-1.NNLM) - **Predict Next Word**
  - Content: Predict next word
  - Paper -  [A Neural Probabilistic Language Model(2003)](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
- 1-2. [Word2Vec(Skip-gram)](1-2.Word2Vec) - **Embedding Words and Show Graph**
  - Content: Word embedding by predicting context words around a central word (target word). Use Softmax loss function (CrossEntropyLoss) 
    to optimize the prediction of context words.
  - Paper - [Distributed Representations of Words and Phrases
    and their Compositionality(2013)](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)
- 1-3. [FastText(Application Level)](1-3.FastText) - **Sentence Classification**
  - Content: An open-source library developed by Facebook AI Research for text classification and word embeddings. It extends the Word2Vec model by considering character n-grams, which helps to better understand word structure and handle out-of-vocabulary words.
  - Paper - [Bag of Tricks for Efficient Text Classification(2016)](https://arxiv.org/pdf/1607.01759.pdf)


#### 2. CNN(Convolutional Neural Network)

- 2-1. [TextCNN](2-1.TextCNN) - **Binary Sentiment Classification - Convolutional Neural Network for text**
  - Paper - [Convolutional Neural Networks for Sentence Classification(2014)](http://www.aclweb.org/anthology/D14-1181)
  - Text classification, using convolution layers to extract features from text.
    The model is trained with simple sentences to classify sentiments.

#### 3. RNN(Recurrent Neural Network)

- 3-1. [TextRNN](3-1.TextRNN) - **Predict Next Step - Recurrent Neural Network**
  - Paper - [Finding Structure in Time(1990)](http://psych.colorado.edu/~kimlab/Elman1990.pdf)
  - Predicting the next word in a sentence. 
    The model uses one-hot encoding for words and is trained on a small dataset to predict the last word in a sentence. For nlp 

- 3-2. [TextLSTM](https://github.com/graykode/nlp-tutorial/tree/master/3-2.TextLSTM) - **Autocomplete - predict the next character**
  - Paper - [LONG SHORT-TERM MEMORY(1997)](https://www.bioinf.jku.at/publications/older/2604.pdf)
  - The LSTM model in this code is used to predict the next character based on a sequence of input characters.
  - The process includes: Data preparation, LSTM model building, training with data, and finally predicting the result.
  - This technique can be extended to handle more complex problems in NLP such as text generation, sentence completion, or semantic analysis.
- 3-3. [Bi-LSTM](3-3.Bi-LSTM) - **Predict Next Word in Long Sentence**
  - This model learns to predict the next word in a sentence based on the previous words.
  - Bi-LSTM is used to leverage information from both the past and future in the data sequence.

#### 4. Attention Mechanism

- 4-1. [Seq2Seq](4-1.Seq2Seq) - **Change Word - translate pairs of synonyms or antonyms**
  - Paper - [Learning Phrase Representations using RNN Encoder–Decoder
    for Statistical Machine Translation(2014)](https://arxiv.org/pdf/1406.1078.pdf)
  - Uses RNN to learn word pairs.
    The model is capable of translating from an input sequence to an output sequence by learning from the training data.
    This technique can be extended to more complex problems such as natural language translation or text summarization.
- 4-2. [Seq2Seq with Attention](4-2.Seq2Seq(Attention)) - **Translate**
  - Paper - [Neural Machine Translation by Jointly Learning to Align and Translate(2014)](https://arxiv.org/abs/1409.0473)
  - The Seq2Seq model with Attention in this code helps translate a sentence from German to English. The Attention mechanism makes the model able to "pay attention" to important parts of  the input sentence when translating, which helps improve accuracy and efficiency.

- 4-3. [Bi-LSTM with Attention](4-3.Bi-LSTM(Attention)) - **Binary Sentiment Classification**
  - classify sentences as good or bad based on using BiLSTM combined with Attention mechanism to find out which words are most important to influence the prediction.

#### 5. Model based on Transformer

- 5-1.  [The Transformer](5-1.Transformer) - **Translate**
  - Paper - [Attention Is All You Need(2017)](https://arxiv.org/abs/1706.03762)

  - ()
- 5-2. [BERT](5-2.BERT) - **Classification Next Sentence & Predict Masked Tokens**
  - Paper - [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding(2018)](https://arxiv.org/abs/1810.04805)


## Dependencies

- Python 3.5+
- Pytorch 1.0.0+


