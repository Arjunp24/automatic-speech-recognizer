# Automatic Speech Recognizer
---
This project aims to build a deep neural network that functions as part of an end-to-end automatic speech recognition (ASR) pipeline. <br>

The [LibriSpeech dataset](http://www.openslr.org/12/) is used to train and evaluate the models. The pipeline will first convert any raw audio to feature representations that are commonly used for ASR. It will then move on to building neural networks that can map these audio features to transcribed text. Different audio features taken into consideration are MFCC features and Spectorgrams. <br>
The various models that were implemented include: <br>
1) Deep RNN + TimeDistributed Dense
2) CNN + RNN + TimeDistributed Dense
3) Bidirectional RNN + TimeDistributed Dense
4) RNN + TimeDistributed Dense
5) Vanilla RNN <br>
(List is presented in decrasing order of validation accuracy) <br><br>
*[Project](https://www.udacity.com/course/natural-language-processing-nanodegree--nd892) done as part of Udacity Natural Language Processing Nanodegree Program*
