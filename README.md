# Deep Machine Learning Knowledge Exchange

### Hello, this is winnerineast. I believe the better future is Human Being + Machine and I'm working on it in order to make it happen. Here is the inventory for all kinds of knowledges I collected from internet without any sign-in.

# Special and equivalent thanks to (I just appended the name at the tail when I leverage or borrow his/her information)
#### [Andrew Thomas](https://github.com/andrewt3000)  
#### [Keon Kim](https://github.com/keonkim)
#### [Nam Vu](https://github.com/ZuzooVn)
#### [Denny Britz](https://github.com/dennybritz) 
#### [Flood Sung](https://github.com/songrotek)

# Each category becomes longer and longer, I have to use multiple files to contain them. And here I just show you the structure and linkages to the respective files.

## [Notes and Tutorials]()
- NLP
  - Word Vectors
    - [Word2Vec tutorial](http://tensorflow.org/tutorials/word2vec/index.html) in [TensorFlow](http://tensorflow.org/)  
    - [Andrew Thomas notes on neural networks](https://github.com/andrewt3000/MachineLearning/blob/master/neuralNets.md)  
    - [Word2vec Parameter Learning Explained](http://www-personal.umich.edu/~ronxin/pdf/w2vexp.pdf)  
    - [Deep Learning, NLP, and Representations](http://colah.github.io/posts/2014-07-NLP-RNNs-Representations/) and 
    - [The amazing power of word vectors](https://blog.acolyer.org/2016/04/21/the-amazing-power-of-word-vectors/)
    - [GloVe: Global vectors for word representation](http://nlp.stanford.edu/projects/glove/glove.pdf)  
    - [Evalutaion section led to controversy](http://rare-technologies.com/making-sense-of-word2vec/) 
    - [Glove source code and training data](http://nlp.stanford.edu/projects/glove/) 

## [Courses]()
- NLP
  - [Kyunghyun Cho's NLP course in NYU](http://www.kyunghyuncho.me/home/courses/ds-ga-3001-fall-2015)
  - [Stanford Natural Language Processing](https://www.coursera.org/learn/nlp)  Intro NLP course with videos. This has **no deep learning**. But it is a good primer for traditional nlp.
  - [Stanford CS 224D: Deep Learning for NLP class](http://cs224d.stanford.edu/syllabus.html)  
  - [Richard Socher](https://scholar.google.com/citations?user=FaOcyfMAAAAJ&hl=en). (2016)  Class with syllabus, and slides. Videos: [2015 lectures](https://www.youtube.com/channel/UCsGC3XXF1ThHwtDo18d7WVw/videos) / [2016 lectures](https://www.youtube.com/playlist?list=PLcGUo322oqu9n4i0X3cRJgKyVy7OkDdoi)   

## [People]()
- [Mikolov](https://scholar.google.com/citations?user=oBu8kMMAAAAJ&hl=en) et al. 2013. Performs well on word similarity and analogy task.  Expands on famous example: King – Man + Woman = Queen  
- [Yoav Goldberg](https://plus.google.com/114479713299850783539/posts/BYvhAbgG8T2)  

## [Source Codes]()
- NLP
  - Word Vectors
    - [Word2Vec source code](https://code.google.com/p/word2vec/)  
  
## [Books]()

## [Papers]()
- NLP
  - Word Vectors
    - [A Primer on Neural Network Models for Natural Language Processing](http://u.cs.biu.ac.il/~yogo/nnlp.pdf) Yoav Goldberg. October 2015. No new info, 75 page summary of state of the art.  
    - [A neural probabilistic language model](http://papers.nips.cc/paper/1839-a-neural-probabilistic-language-model.pdf) Bengio 2003. Seminal paper on word vectors.  
    - [Efficient Estimation of Word Representations in Vector Space](http://arxiv.org/pdf/1301.3781v3.pdf)  
    - [Distributed Representations of Words and Phrases and their Compositionality](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)  
    - [Linguistic Regularities in Continuous Space Word Representations](http://www.aclweb.org/anthology/N13-1090)  
    - [Enriching Word Vectors with Subword Information](https://arxiv.org/pdf/1607.04606v1.pdf)  






=================
The following are not yet consolidated part.
=================

___
[FastText Code](https://github.com/facebookresearch/fastText)  


## Sentiment Analysis
Thought vectors are numeric representations for sentences, paragraphs, and documents.  This concept is used for many text classification tasks such as sentiment analysis.      

[Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank](http://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf)  
Socher et al. 2013.  Introduces Recursive Neural Tensor Network and dataset: "sentiment treebank."  Includes [demo site](http://nlp.stanford.edu/sentiment/
). Uses a parse tree.

[Distributed Representations of Sentences and Documents](http://cs.stanford.edu/~quocle/paragraph_vector.pdf)  
[Le](https://scholar.google.com/citations?user=vfT6-XIAAAAJ), Mikolov. 2014.  Introduces Paragraph Vector. Concatenates and averages pretrained, fixed word vectors to create vectors for sentences, paragraphs and documents. Also known as paragraph2vec.  Doesn't use a parse tree.  
Implemented in [gensim](https://github.com/piskvorky/gensim/).  See [doc2vec tutorial](http://rare-technologies.com/doc2vec-tutorial/)

[Deep Recursive Neural Networks for Compositionality in Language](http://www.cs.cornell.edu/~oirsoy/files/nips14drsv.pdf)  
Irsoy & Cardie. 2014.  Uses Deep Recursive Neural Networks. Uses a parse tree.

[Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks](https://aclweb.org/anthology/P/P15/P15-1150.pdf)  
Tai et al. 2015  Introduces Tree LSTM. Uses a parse tree.

[Semi-supervised Sequence Learning](http://arxiv.org/pdf/1511.01432.pdf)  
Dai, Le 2015  
Approach: "We present two approaches that use unlabeled data to improve sequence learning with recurrent networks. The first approach is to predict what comes next in a sequence, which is a conventional language model in natural language processing.
The second approach is to use a sequence autoencoder..."  
Result: "With pretraining, we are able to train long short term memory recurrent networks up to a few hundred
timesteps, thereby achieving strong performance in many text classification tasks, such as IMDB, DBpedia and 20 Newsgroups."

[Bag of Tricks for Efficient Text Classification](https://arxiv.org/abs/1607.01759)  
Joulin, Grave, Bojanowski, Mikolov 2016 Facebook AI Research.  
"Our experiments show that our fast text classifier fastText is often on par with deep learning classifiers in terms of accuracy, and many orders of magnitude faster for training and evaluation."  
[FastText blog](https://research.facebook.com/blog/fasttext/)  
[FastText Code](https://github.com/facebookresearch/fastText)  

##Neural Machine Translation
In 2014, neural machine translation (NMT) performance became comprable to state of the art statistical machine translation(SMT).  

[Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](http://arxiv.org/pdf/1406.1078v3.pdf) ([abstract](https://arxiv.org/abs/1406.1078))    
Cho et al. 2014 Breakthrough deep learning paper on machine translation. Introduces basic sequence to sequence model which includes two rnns, an encoder for input and a decoder for output.  

[Neural Machine Translation by jointly learning to align and translate](http://arxiv.org/pdf/1409.0473v6.pdf) ([abstract](https://arxiv.org/abs/1409.0473))     
Bahdanau, Cho, Bengio 2014.  
Implements attention mechanism. "Each time the proposed model generates a word in a translation, it
(soft-)searches for a set of positions in a source sentence where the most relevant information is
concentrated"  
Result: "comparable to the existing state-of-the-art phrase-based system on the task of English-to-French translation."  
[English to French Demo](http://104.131.78.120/)  

[On Using Very Large Target Vocabulary for Neural Machine Translation](https://arxiv.org/pdf/1412.2007v2.pdf)  
Jean, Cho, Memisevic, Bengio 2014.    
"we try replacing each [UNK] token with the aligned source word or its most likely translation determined by another word alignment model."  
Result: English -> German bleu score = 21.59 (target vocabulary of 50,000)    

[Sequence to Sequence Learning with Neural Networks](http://arxiv.org/pdf/1409.3215v3.pdf)  
Sutskever, Vinyals, Le 2014.  ([nips presentation](http://research.microsoft.com/apps/video/?id=239083)). Uses seq2seq to generate translations.  
Result: English -> French bleu score = 34.8 (WMT’14 dataset)  
A key contribution is improvements from reversing the source sentences.  
[seq2seq tutorial](http://tensorflow.org/tutorials/seq2seq/index.html) in [TensorFlow](http://tensorflow.org/).   

[Addressing the Rare Word Problem in Neural Machine Translation](https://arxiv.org/pdf/1410.8206v4.pdf) ([abstract](https://arxiv.org/abs/1410.8206))  
Luong, Sutskever, Le, Vinyals, Zaremba 2014    
Replace UNK words with dictionary lookup.  
Result: English -> French BLEU score = 37.5.  

[Effective Approaches to Attention-based Neural Machine Translation](http://stanford.edu/~lmthang/data/papers/emnlp15_attn.pdf)  
Luong, Pham, Manning. 2015  
2 models of attention: global and local.  
Result: English -> German 25.9 BLEU points  

[Context-Dependent Word Representation for Neural
Machine Translation](http://arxiv.org/pdf/1607.00578v1.pdf)  
Choi, Cho, Bengio 2016  
"we propose to contextualize the word embedding vectors using a nonlinear bag-of-words representation of the source sentence."  
"we propose to represent special tokens (such as numbers, proper nouns and acronyms) with typed symbols to facilitate translating those words that are not well-suited to be translated via continuous vectors."   

[Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation](http://arxiv.org/abs/1609.08144)  
Wu et al. 2016  
[blog post](https://research.googleblog.com/2016/09/a-neural-network-for-machine.html)  
"WMT’14 English-to-French, our single model scores 38.95 BLEU"  
"WMT’14 English-to-German, our single model scores 24.17 BLEU"  

## Image Captioning
[Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](http://arxiv.org/pdf/1502.03044v3.pdf)  
Xu et al. 2015 Creates captions by feeding image into a CNN which feeds into hidden state of an RNN that generates the caption. At each time step the RNN outputs next word and the next location to pay attention to via a probability over grid locations. Uses 2 types of attention soft and hard. Soft attention uses gradient descent and backprop and is deterministic. Hard attention selects the element with highest probability. Hard attention uses reinforcement learning, rather than backprop and is stochastic.  

[Open source implementation in TensorFlow](https://research.googleblog.com/2016/09/show-and-tell-image-captioning-open.html)  

##Conversation modeling / Dialog
[Neural Responding Machine for Short-Text Conversation](http://arxiv.org/pdf/1503.02364v2.pdf)  
Shang et al. 2015  Uses Neural Responding Machine.  Trained on Weibo dataset.  Achieves one round conversations with 75% appropriate responses.  

[A Neural Network Approach to Context-Sensitive Generation of Conversational Responses](http://arxiv.org/pdf/1506.06714v1.pdf)  
Sordoni et al. 2015.  Generates responses to tweets.   
Uses [Recurrent Neural Network Language Model (RLM) architecture
of (Mikolov et al., 2010).](http://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf)  source code: [RNNLM Toolkit](http://www.rnnlm.org/)

[Building End-To-End Dialogue Systems Using Generative Hierarchical Neural Network Models](http://arxiv.org/pdf/1507.04808v3.pdf)  
Serban, Sordoni, Bengio et al. 2015. Extends [hierarchical recurrent encoder-decoder](https://arxiv.org/abs/1507.02221) neural network (HRED).

[Attention with Intention for a Neural Network Conversation Model](http://arxiv.org/pdf/1510.08565v3.pdf)  
Yao et al. 2015 Architecture is three recurrent networks: an encoder, an intention network and a decoder.  

[A Hierarchical Latent Variable Encoder-Decoder Model for Generating Dialogues](http://arxiv.org/pdf/1605.06069v3.pdf)  
Serban, Sordoni, Lowe, Charlin, Pineau, Courville, Bengio 2016  
Proposes novel architecture: VHRED.  Latent Variable Hierarchical Recurrent Encoder-Decoder  
Compares favorably against LSTM and HRED.  
___
[A Neural Conversation Model](http://arxiv.org/pdf/1506.05869v3.pdf)  
Vinyals, [Le](https://scholar.google.com/citations?user=vfT6-XIAAAAJ) 2015.  Uses LSTM RNNs to generate conversational responses. Uses [seq2seq framework](http://tensorflow.org/tutorials/seq2seq/index.html).  Seq2Seq was originally designed for machine translation and it "translates" a single sentence, up to around 79 words, to a single sentence response, and has no memory of previous dialog exchanges.  Used in Google [Smart Reply feature for Inbox](http://googleresearch.blogspot.co.uk/2015/11/computer-respond-to-this-email.html)  

[Incorporating Copying Mechanism in Sequence-to-Sequence Learning](http://arxiv.org/pdf/1603.06393v3.pdf)  
Gu et al. 2016 Proposes CopyNet, builds on seq2seq.  

[A Persona-Based Neural Conversation Model](http://arxiv.org/pdf/1603.06155v2.pdf)  
Li et al. 2016  Proposes persona-based models for handling the issue of speaker consistency in neural response generation. Builds on seq2seq.  

[Deep Reinforcement Learning for Dialogue Generation](https://arxiv.org/pdf/1606.01541v3.pdf)  
Li et al. 2016. Uses reinforcement learing to generate diverse responses. Trains 2 agents to chat with each other. Builds on seq2seq.   

___
[Deep learning for chatbots](http://www.wildml.com/2016/04/deep-learning-for-chatbots-part-1-introduction/)  
Article summary of state of the art, and challenges for chatbots.  
[Deep learning for chatbots. part 2](http://www.wildml.com/2016/07/deep-learning-for-chatbots-2-retrieval-based-model-tensorflow/)  
Implements a retrieval based dialog agent using dual encoder lstm with TensorFlow, based on the Ubuntu dataset [[paper](http://arxiv.org/pdf/1506.08909v3.pdf)] includes [source code](https://github.com/dennybritz/chatbot-retrieval/)  

##Memory and Attention Models
Attention mechanisms allows the network to refer back to the input sequence, instead of forcing it to encode all information into one fixed-length vector.  - [Attention and Memory in Deep Learning and NLP](http://www.opendatascience.com/blog/attention-and-memory-in-deep-learning-and-nlp/)  

[Memory Networks](http://arxiv.org/pdf/1410.3916v10.pdf) Weston et. al 2014, and 
[End-To-End Memory Networks](http://arxiv.org/pdf/1503.08895v4.pdf) Sukhbaatar et. al 2015.  
Memory networks are implemented in [MemNN](https://github.com/facebook/MemNN).  Attempts to solve task of reason attention and memory.  
[Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks](http://arxiv.org/pdf/1502.05698v7.pdf)  
Weston 2015. Classifies QA tasks like single factoid, yes/no etc. Extends memory networks.  
[Evaluating prerequisite qualities for learning end to end dialog systems](http://arxiv.org/pdf/1511.06931.pdf)  
Dodge et. al 2015. Tests Memory Networks on 4 tasks including reddit dialog task.  
See [Jason Weston lecture on MemNN](https://www.youtube.com/watch?v=Xumy3Yjq4zk)  
  
[Neural Turing Machines](http://arxiv.org/pdf/1410.5401v2.pdf)  
Graves, Wayne, Danihelka 2014.  
We extend the capabilities of neural networks by coupling them to external memory resources, which they can interact with by attentional processes. The combined system is analogous to a Turing Machine or Von Neumann architecture but is differentiable end-toend, allowing it to be efficiently trained with gradient descent. Preliminary results demonstrate
that Neural Turing Machines can infer simple algorithms such as copying, sorting, and associative recall from input and output examples.
[Olah and Carter blog on NTM](http://distill.pub/2016/augmented-rnns/#neural-turing-machines)  

[Inferring Algorithmic Patterns with Stack-Augmented Recurrent Nets](http://arxiv.org/pdf/1503.01007v4.pdf)  
Joulin, Mikolov 2015. [Stack RNN source code](https://github.com/facebook/Stack-RNN) and [blog post](https://research.facebook.com/blog/1642778845966521/inferring-algorithmic-patterns-with-stack/)  


[Reasoning, Attention and Memory RAM workshop at NIPS 2015. slides included](http://www.thespermwhale.com/jaseweston/ram/)  

=====================================================================================

# awesome-nlp [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

> A curated list of resources dedicated to Natural Language Processing
>
> Maintainers - [Keon Kim](https://github.com/keonkim), [Martin Park](https://github.com/outpark)

*Please read the [contribution guidelines](contributing.md) before contributing.*

Please feel free to [pull requests](https://github.com/keonkim/awesome-nlp/pulls), or email Martin Park (sp3005@nyu.edu)/Keon Kim (keon.kim@nyu.edu) to add links.


## Table of Contents

 - [Tutorials and Courses](#tutorials-and-courses)
   - [videos](#videos)
 - [Deep Learning for NLP](#deep-learning-for-nlp)
 - [Packages](#packages)
   - [Implemendations](#implementations)
   - [Libraries](#libraries)
     - [Node.js](#user-content-node-js)
     - [Python](#user-content-python)
     - [C++](#user-content-c++)
     - [Java](#user-content-java)
     - [Clojure](#user-content-clojure)
     - [Ruby](#user-content-ruby)
   - [Services](#services)
 - [Articles](#articles)
   - [Review Articles](#review-articles)
   - [Word Vectors](#word-vectors)
   - [Thought Vectors](#thought-vectors)
   - [Machine Translation](#machine-translation)
   - [General Natural Language Processing](#general-natural-langauge-processing)
   - [Named Entity Recognition](#name-entity-recognition)
   - [Single Exchange Dialogs](#single-exchange-dialogs)
   - [Memory and Attention Models](#memory-and-attention-models)
   - [General Natural Language Processing](#general-natural-language-processing)
   - [Named Entity Recognition](#named-entity-recognition)
   - [Neural Network](#neural-network)
   - [Supplementary Materials](#supplementary-materials)
 - [Blogs](#blogs)
 - [Credits](#credits)


## Tutorials and Courses

* Tensor Flow Tutorial on [Seq2Seq](https://www.tensorflow.org/tutorials/seq2seq/index.html) Models
* Natural Language Understanding with Distributed Representation [Lecture Note](https://github.com/nyu-dl/NLP_DL_Lecture_Note) by Cho
* [Michael Collins](http://www.cs.columbia.edu/~mcollins/) - one of the best NLP teachers. Check out the material on the courses he is teaching.

### videos

* [Intro to Natural Language Processing](https://www.coursera.org/learn/natural-language-processing) on Coursera by U of Michigan
* [Intro to Artificial Intelligence](https://www.udacity.com/course/intro-to-artificial-intelligence--cs271) course on Udacity which also covers NLP
* [Deep Learning for Natural Language Processing (2015 classes)](https://www.youtube.com/playlist?list=PLmImxx8Char8dxWB9LRqdpCTmewaml96q) by Richard Socher
* [Deep Learning for Natural Language Processing (2016 classes)](https://www.youtube.com/playlist?list=PLmImxx8Char9Ig0ZHSyTqGsdhb9weEGam) by Richard Socher. Updated to make use of Tensorflow. Note that there are some lectures missing (lecture 9, and lectures 12 onwards). 
* [Natural Language Processing](https://www.coursera.org/learn/nlangp) - course on Coursera that was only done in 2013. The videos are not available at the moment. Also Mike Collins is a great professor and his notes and lectures are very good. 
* [Statistical Machine Translation](http://mt-class.org) - a Machine Translation course with great assignments and slides. 
* [Natural Language Processing SFU](http://www.cs.sfu.ca/~anoop/teaching/CMPT-413-Spring-2014/) - course by [Prof Anoop Sarkar](https://www.cs.sfu.ca/~anoop/) on Natural Language Processing. Good notes and some good lectures on youtube about HMM. 
* [Udacity Deep Learning](https://classroom.udacity.com/courses/ud730) Deep Learning course on Udacity (using Tensorflow) which covers a section on using deep learning for NLP tasks (covering Word2Vec, RNN's and LSTMs).
* [NLTK with Python 3 for Natural Language Processing](https://www.youtube.com/playlist?list=PLQVvvaa0QuDf2JswnfiGkliBInZnIC4HL) by Harrison Kinsley(sentdex). Good tutorials with NLTK code implementation.

## Deep Learning for NLP 

[Stanford CS 224D: Deep Learning for NLP class](http://cs224d.stanford.edu/syllabus.html)  
Class by [Richard Socher](https://scholar.google.com/citations?user=FaOcyfMAAAAJ&hl=en). 2016 content was updated to make use of Tensorflow. Lecture slides and reading materials for 2016 class [here](http://cs224d.stanford.edu/syllabus.html). Videos for 2016 class [here](https://www.youtube.com/playlist?list=PLmImxx8Char9Ig0ZHSyTqGsdhb9weEGam). Note that there are some lecture videos missing for 2016 (lecture 9, and lectures 12 onwards). All videos for 2015 class [here](https://www.youtube.com/playlist?list=PLmImxx8Char8dxWB9LRqdpCTmewaml96q)

[Udacity Deep Learning](https://classroom.udacity.com/courses/ud730)
Deep Learning course on Udacity (using Tensorflow) which covers a section on using deep learning for NLP tasks. This section covers how to implement Word2Vec, RNN's and LSTMs.

[A Primer on Neural Network Models for Natural Language Processing](http://u.cs.biu.ac.il/~yogo/nnlp.pdf)  
Yoav Goldberg. October 2015. No new info, 75 page summary of state of the art.  


## Packages

### Implementations
* [Pre-trained word embeddings for WSJ corpus](https://github.com/ai-ku/wvec) by Koc AI-Lab
* [Word2vec](https://code.google.com/archive/p/word2vec) by Mikolov
* [HLBL language model](http://metaoptimize.com/projects/wordreprs/) by Turian
* [Real-valued vector "embeddings"](http://www.cis.upenn.edu/~ungar/eigenwords/) by Dhillon
* [Improving Word Representations Via Global Context And Multiple Word Prototypes](http://www.socher.org/index.php/Main/ImprovingWordRepresentationsViaGlobalContextAndMultipleWordPrototypes) by Huang
* [Dependency based word embeddings](https://levyomer.wordpress.com/2014/04/25/dependency-based-word-embeddings/)
* [Global Vectors for Word Representations](http://nlp.stanford.edu/projects/glove/)

### Libraries
* [TwitIE: An Open-Source Information Extraction Pipeline for Microblog Text](http://www.anthology.aclweb.org/R/R13/R13-1011.pdf)

* <a id="node-js">**Node.js and Javascript** - Node.js Libaries for NLP</a>
  * [Twitter-text](https://github.com/twitter/twitter-text) - A JavaScript implementation of Twitter's text processing library
  * [Knwl.js](https://github.com/loadfive/Knwl.js) - A Natural Language Processor in JS
  * [Retext](https://github.com/wooorm/retext) - Extensible system for analyzing and manipulating natural language
  * [NLP Compromise](https://github.com/nlp-compromise/nlp_compromise) - Natural Language processing in the browser
  * [Natural](https://github.com/NaturalNode/natural) - general natural language facilities for node

* <a id="python">**Python** - Python NLP Libraries</a>
  * [Scikit-learn: Machine learning in Python](http://arxiv.org/pdf/1201.0490.pdf)
  * [Natural Language Toolkit (NLTK)](http://www.nltk.org/)
  * [Pattern](http://www.clips.ua.ac.be/pattern) - A web mining module for the Python programming language. It has tools for natural language processing, machine learning, among others.
  * [TextBlob](http://textblob.readthedocs.org/) - Providing a consistent API for diving into common natural language processing (NLP) tasks. Stands on the giant shoulders of NLTK and Pattern, and plays nicely with both.
  * [YAlign](https://github.com/machinalis/yalign) - A sentence aligner, a friendly tool for extracting parallel sentences from comparable corpora.
  * [jieba](https://github.com/fxsjy/jieba#jieba-1) - Chinese Words Segmentation Utilities.
  * [SnowNLP](https://github.com/isnowfy/snownlp) - A library for processing Chinese text.
  * [KoNLPy](http://konlpy.org) - A Python package for Korean natural language processing.
  * [Rosetta](https://github.com/columbia-applied-data-science/rosetta) - Text processing tools and wrappers (e.g. Vowpal Wabbit)
  * [BLLIP Parser](https://pypi.python.org/pypi/bllipparser/) - Python bindings for the BLLIP Natural Language Parser (also known as the Charniak-Johnson parser)
  * [PyNLPl](https://github.com/proycon/pynlpl) - Python Natural Language Processing Library. General purpose NLP library for Python. Also contains some specific modules for parsing common NLP formats, most notably for [FoLiA](http://proycon.github.io/folia/), but also ARPA language models, Moses phrasetables, GIZA++ alignments.
  * [python-ucto](https://github.com/proycon/python-ucto) - Python binding to ucto (a unicode-aware rule-based tokenizer for various languages)
  * [python-frog](https://github.com/proycon/python-frog) - Python binding to Frog, an NLP suite for Dutch. (pos tagging, lemmatisation, dependency parsing, NER)
  * [python-zpar](https://github.com/EducationalTestingService/python-zpar) - Python bindings for [ZPar](https://github.com/frcchang/zpar), a statistical part-of-speech-tagger, constiuency parser, and dependency parser for English.
  * [colibri-core](https://github.com/proycon/colibri-core) - Python binding to C++ library for extracting and working with with basic linguistic constructions such as n-grams and skipgrams in a quick and memory-efficient way.
  * [spaCy](https://github.com/spacy-io/spaCy) - Industrial strength NLP with Python and Cython.
  * [PyStanfordDependencies](https://github.com/dmcc/PyStanfordDependencies) - Python interface for converting Penn Treebank trees to Stanford Dependencies.

* <a id="c++">**C++** - C++ Libraries</a>
  * [MIT Information Extraction Toolkit](https://github.com/mit-nlp/MITIE) - C, C++, and Python tools for named entity recognition and relation extraction
  * [CRF++](https://taku910.github.io/crfpp/) - Open source implementation of Conditional Random Fields (CRFs) for segmenting/labeling sequential data & other Natural Language Processing tasks.
  * [CRFsuite](http://www.chokkan.org/software/crfsuite/) - CRFsuite is an implementation of Conditional Random Fields (CRFs) for labeling sequential data.
  * [BLLIP Parser](https://github.com/BLLIP/bllip-parser) - BLLIP Natural Language Parser (also known as the Charniak-Johnson parser)
  * [colibri-core](https://github.com/proycon/colibri-core) - C++ library, command line tools, and Python binding for extracting and working with basic linguistic constructions such as n-grams and skipgrams in a quick and memory-efficient way.
  * [ucto](https://github.com/LanguageMachines/ucto) - Unicode-aware regular-expression based tokenizer for various languages. Tool and C++ library. Supports FoLiA format.
  * [libfolia](https://github.com/LanguageMachines/libfolia) - C++ library for the [FoLiA format](http://proycon.github.io/folia/)
  * [frog](https://github.com/LanguageMachines/frog) - Memory-based NLP suite developed for Dutch: PoS tagger, lemmatiser, dependency parser, NER, shallow parser, morphological analyzer.
  * [MeTA](https://github.com/meta-toolkit/meta) - [MeTA : ModErn Text Analysis](https://meta-toolkit.org/) is a C++ Data Sciences Toolkit that facilitates mining big text data.
  * [Mecab (Japanese)](http://taku910.github.io/mecab/)
  * [Mecab (Korean)](http://eunjeon.blogspot.com/)
  * [Moses](http://statmt.org/moses/)

* <a id="java">**Java** - Java NLP Libraries</a>
  * [Stanford NLP](http://nlp.stanford.edu/software/index.shtml)
  * [OpenNLP](http://opennlp.apache.org/)
  * [ClearNLP](https://github.com/clir/clearnlp)
  * [Word2vec in Java](http://deeplearning4j.org/word2vec.html)
  * [ReVerb](https://github.com/knowitall/reverb/) Web-Scale Open Information Extraction
  * [OpenRegex](https://github.com/knowitall/openregex) An efficient and flexible token-based regular expression language and engine.  
  * [CogcompNLP](https://github.com/IllinoisCogComp/illinois-cogcomp-nlp) - Core libraries developed in the U of Illinois' Cognitive Computation Group. 
  
* <a id="scala">**Scala** - Scala NLP Libraries</a>
  * [Saul](https://github.com/IllinoisCogComp/saul) - Library for developing NLP systems, including built in modules like SRL, POS, etc. 

* <a id="clojure">**Clojure**</a>
  * [Clojure-openNLP](https://github.com/dakrone/clojure-opennlp) - Natural Language Processing in Clojure (opennlp)
  * [Infections-clj](https://github.com/r0man/inflections-clj) - Rails-like inflection library for Clojure and ClojureScript

* <a id="ruby">**Ruby**</a>
  * Kevin Dias's [A collection of Natural Language Processing (NLP) Ruby libraries, tools and software](https://github.com/diasks2/ruby-nlp)
  
### Services
* [Wit-ai](https://github.com/wit-ai/wit) - Natural Language Interface for apps and devices.
* [Iris](http://iris.lore.ai) - Free text search API over large public document collections.

## Articles

### Review Articles
* [Deep Learning for Web Search and Natural Language Processing](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/wsdm2015.v3.pdf)
* [Probabilistic topic models](https://www.cs.princeton.edu/~blei/papers/Blei2012.pdf)
* [Natural language processing: an introduction](http://jamia.oxfordjournals.org/content/18/5/544.short)
* [A unified architecture for natural language processing: Deep neural networks with multitask learning](http://arxiv.org/pdf/1201.0490.pdf)
* [A Critical Review of Recurrent Neural Networksfor Sequence Learning](http://arxiv.org/pdf/1506.00019v1.pdf)
* [Deep parsing in Watson](http://nlp.cs.rpi.edu/course/spring14/deepparsing.pdf)
* [Online named entity recognition method for microtexts in social networking services: A case study of twitter](http://arxiv.org/pdf/1301.2857.pdf)


### Word Vectors
Resources about word vectors, aka word embeddings, and distributed representations for words.  
Word vectors are numeric representations of words that are often used as input to deep learning systems. This process is sometimes called pretraining.  

[Efficient Estimation of Word Representations in Vector Space](http://arxiv.org/pdf/1301.3781v3.pdf)  
[Distributed Representations of Words and Phrases and their Compositionality]
(http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)  
[Mikolov](https://scholar.google.com/citations?user=oBu8kMMAAAAJ&hl=en) et al. 2013.  
Generate word and phrase vectors.  Performs well on word similarity and analogy task and includes [Word2Vec source code](https://code.google.com/p/word2vec/)  Subsamples frequent words. (i.e. frequent words like "the" are skipped periodically to speed things up and improve vector for less frequently used words)  
[Word2Vec tutorial](http://tensorflow.org/tutorials/word2vec/index.html) in [TensorFlow](http://tensorflow.org/)  

[Deep Learning, NLP, and Representations](http://colah.github.io/posts/2014-07-NLP-RNNs-Representations/)  
Chris Olah (2014)  Blog post explaining word2vec.  

[GloVe: Global vectors for word representation](http://nlp.stanford.edu/projects/glove/glove.pdf)  
Pennington, Socher, Manning. 2014. Creates word vectors and relates word2vec to matrix factorizations.  [Evalutaion section led to controversy](http://rare-technologies.com/making-sense-of-word2vec/) by [Yoav Goldberg](https://plus.google.com/114479713299850783539/posts/BYvhAbgG8T2)  
[Glove source code and training data](http://nlp.stanford.edu/projects/glove/)

* [word2vec](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) - on creating vectors to represent language, useful for RNN inputs
* [sense2vec](http://arxiv.org/abs/1511.06388) - on word sense disambiguation
* [Infinite Dimensional Word Embeddings](http://arxiv.org/abs/1511.05392) - new
* [Skip Thought Vectors](http://arxiv.org/abs/1506.06726) - word representation method
* [Adaptive skip-gram](http://arxiv.org/abs/1502.07257) - similar approach, with adaptive properties

### Thought Vectors
Thought vectors are numeric representations for sentences, paragraphs, and documents.  The following papers are listed in order of date published, each one replaces the last as the state of the art in sentiment analysis.  

[Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.383.1327&rep=rep1&type=pdf)  
Socher et al. 2013.  Introduces Recursive Neural Tensor Network.  Uses a parse tree.

[Distributed Representations of Sentences and Documents](http://cs.stanford.edu/~quocle/paragraph_vector.pdf)  
[Le](https://scholar.google.com/citations?user=vfT6-XIAAAAJ), Mikolov. 2014.  Introduces Paragraph Vector. Concatenates and averages pretrained, fixed word vectors to create vectors for sentences, paragraphs and documents. Also known as paragraph2vec.  Doesn't use a parse tree.  
Implemented in [gensim](https://github.com/piskvorky/gensim/).  See [doc2vec tutorial](http://rare-technologies.com/doc2vec-tutorial/)

[Deep Recursive Neural Networks for Compositionality in Language](http://www.cs.cornell.edu/~oirsoy/files/nips14drsv.pdf)  
Irsoy & Cardie. 2014.  Uses Deep Recursive Neural Networks. Uses a parse tree.

[Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks](https://aclweb.org/anthology/P/P15/P15-1150.pdf)  
Tai et al. 2015  Introduces Tree LSTM. Uses a parse tree.

[Semi-supervised Sequence Learning](http://arxiv.org/pdf/1511.01432.pdf)  
Dai, Le 2015 "With pretraining, we are able to train long short term memory recurrent networks up to a few hundred
timesteps, thereby achieving strong performance in many text classification tasks, such as IMDB, DBpedia and 20 Newsgroups."  
### Machine Translation
[Neural Machine Translation by jointly learning to align and translate](http://arxiv.org/pdf/1409.0473v6.pdf)
Bahdanau, Cho 2014.  "comparable to the existing state-of-the-art phrase-based system on the task of English-to-French translation."  Implements attention mechanism.  
[English to French Demo](http://104.131.78.120/)  

[Sequence to Sequence Learning with Neural Networks](http://arxiv.org/pdf/1409.3215v3.pdf)  
Sutskever, Vinyals, Le 2014.  ([nips presentation](http://research.microsoft.com/apps/video/?id=239083)). Uses LSTM RNNs to generate translations. " Our main result is that on an English to French translation task from the WMT’14 dataset, the translations produced by the LSTM achieve a BLEU score of 34.8"  
[seq2seq tutorial](http://tensorflow.org/tutorials/seq2seq/index.html) in 

* [Cross-lingual Pseudo-Projected Expectation Regularization for Weakly Supervised Learning](http://arxiv.org/pdf/1310.1597v1.pdf)
* [Generating Chinese Named Entity Data from a Parallel Corpus](http://www.mt-archive.info/IJCNLP-2011-Fu.pdf)
* [IXA pipeline: Efficient and Ready to Use Multilingual NLP tools](http://www.lrec-conf.org/proceedings/lrec2014/pdf/775_Paper.pdf)


### Single Exchange Dialogs
[A Neural Network Approach toContext-Sensitive Generation of Conversational Responses](http://arxiv.org/pdf/1506.06714v1.pdf)  
Sordoni 2015.  Generates responses to tweets.   
Uses [Recurrent Neural Network Language Model (RLM) architecture
of (Mikolov et al., 2010).](http://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf)  source code: [RNNLM Toolkit](http://www.rnnlm.org/)

[Neural Responding Machine for Short-Text Conversation](http://arxiv.org/pdf/1503.02364v2.pdf)  
Shang et al. 2015  Uses Neural Responding Machine.  Trained on Weibo dataset.  Achieves one round conversations with 75% appropriate responses.  

[A Neural Conversation Model](http://arxiv.org/pdf/1506.05869v3.pdf)  
Vinyals, [Le](https://scholar.google.com/citations?user=vfT6-XIAAAAJ) 2015.  Uses LSTM RNNs to generate conversational responses. Uses [seq2seq framework](http://tensorflow.org/tutorials/seq2seq/index.html).  Seq2Seq was originally designed for machine transation and it "translates" a single sentence, up to around 79 words, to a single sentence response, and has no memory of previous dialog exchanges.  Used in Google [Smart Reply feature for Inbox](http://googleresearch.blogspot.co.uk/2015/11/computer-respond-to-this-email.html)  

### Memory and Attention Models (from [DL4NLP](https://github.com/andrewt3000/DL4NLP))
[Reasoning, Attention and Memory RAM workshop at NIPS 2015. slides included](http://www.thespermwhale.com/jaseweston/ram/)  

[Memory Networks](http://arxiv.org/pdf/1410.3916v10.pdf) Weston et. al 2014, and 
[End-To-End Memory Networks](http://arxiv.org/pdf/1503.08895v4.pdf) Sukhbaatar et. al 2015.  
Memory networks are implemented in [MemNN](https://github.com/facebook/MemNN).  Attempts to solve task of reason attention and memory.  
[Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks](http://arxiv.org/pdf/1502.05698v7.pdf)  
Weston 2015. Classifies QA tasks like single factoid, yes/no etc. Extends memory networks.  
[Evaluating prerequisite qualities for learning end to end dialog systems](http://arxiv.org/pdf/1511.06931.pdf)  
Dodge et. al 2015. Tests Memory Networks on 4 tasks including reddit dialog task.  
See [Jason Weston lecture on MemNN](https://www.youtube.com/watch?v=Xumy3Yjq4zk)  
  
[Neural Turing Machines](http://arxiv.org/pdf/1410.5401v2.pdf)  
Graves et al. 2014.  

[Inferring Algorithmic Patterns with Stack-Augmented Recurrent Nets](http://arxiv.org/pdf/1503.01007v4.pdf)  
Joulin, Mikolov 2015. [Stack RNN source code](https://github.com/facebook/Stack-RNN) and [blog post](https://research.facebook.com/blog/1642778845966521/inferring-algorithmic-patterns-with-stack/)  

### General Natural Language Processing
* [Neural autocoder for paragraphs and documents](http://arxiv.org/abs/1506.01057) - LSTM representation
* [LSTM over tree structures](http://arxiv.org/abs/1503.04881)
* [Sequence to Sequence Learning](http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf) - word vectors for machine translation
* [Teaching Machines to Read and Comprehend](http://arxiv.org/abs/1506.03340) - DeepMind paper
* [Efficient Estimation of Word Representations in Vector Space](http://arxiv.org/pdf/1301.3781.pdf)
* [Improving distributional similarity with lessons learned from word embeddings](https://tacl2013.cs.columbia.edu/ojs/index.php/tacl/article/viewFile/570/124)
* [Low-Dimensional Embeddings of Logic](http://www.aclweb.org/anthology/W/W14/W14-2409.pdf)
* Tutorial on Markov Logic Networks ([based on this paper](http://homes.cs.washington.edu/~pedrod/papers/mlj05.pdf))
* [Markov Logic Networks for Natural Language Question Answering](http://arxiv.org/pdf/1507.03045v1.pdf)
* [Distant Supervision for Cancer Pathway Extraction From Text](http://research.microsoft.com/en-us/um/people/hoifung/papers/psb15.pdf)
* [Privee: An Architecture for Automatically Analyzing Web Privacy Policies](http://www.sebastianzimmeck.de/zimmeckAndBellovin2014Privee.pdf)
* [A Neural Probabilistic Language Model](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
* [Template-Based Information Extraction without the Templates](http://www.usna.edu/Users/cs/nchamber/pubs/acl2011-chambers-templates.pdf)
* [Retrofitting word vectors to semantic lexicons](http://www.cs.cmu.edu/~mfaruqui/papers/naacl15-retrofitting.pdf)
* [Unsupervised Learning of the Morphology of a Natural Language](http://www.mitpressjournals.org/doi/pdfplus/10.1162/089120101750300490)
* [Natural Language Processing (Almost) from Scratch](http://arxiv.org/pdf/1103.0398.pdf)
* [Computational Grounded Cognition: a new alliance between grounded cognition and computational modelling](http://journal.frontiersin.org/article/10.3389/fpsyg.2012.00612/full)
* [Learning the Structure of Biomedical Relation Extractions](http://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004216)
* [Relation extraction with matrix factorization and universal schemas](http://www.anthology.aclweb.org/N/N13/N13-1008.pdf)

### Named Entity Recognition
* [A survey of named entity recognition and classification](http://nlp.cs.nyu.edu/sekine/papers/li07.pdf)
* [Benchmarking the extraction and disambiguation of named entities on the semantic web](http://www.lrec-conf.org/proceedings/lrec2014/pdf/176_Paper.pdf)
* [Knowledge base population: Successful approaches and challenges](http://www.aclweb.org/anthology/P11-1115)
* [SpeedRead: A fast named entity recognition Pipeline](http://arxiv.org/pdf/1301.2857.pdf)

### Neural Network
* [The Unreasonable Effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness)
* [Statistical Language Models based on Neural Networks](http://www.fit.vutbr.cz/~imikolov/rnnlm/thesis.pdf)
* [Slides from Google Talk](http://www.fit.vutbr.cz/~imikolov/rnnlm/google.pdf)

### Supplementary Materials
* [Word2Vec](https://github.com/clulab/nlp-reading-group/wiki/Word2Vec-Resources)
* [Relation Extraction with Matrix Factorization and Universal Schemas](http://www.riedelcastro.org//publications/papers/riedel13relation.pdf)
* [Towards a Formal Distributional Semantics: Simulating Logical Calculi with Tensors](http://www.aclweb.org/anthology/S13-1001)
* [Presentation slides for MLN tutorial](https://github.com/clulab/nlp-reading-group/blob/master/fall-2015-resources/mln-summary-20150918.ppt)
* [Presentation slides for QA applications of MLNs](https://github.com/clulab/nlp-reading-group/blob/master/fall-2015-resources/Markov%20Logic%20Networks%20for%20Natural%20Language%20Question%20Answering.pdf)
* [Presentation slides](https://github.com/clulab/nlp-reading-group/blob/master/fall-2015-resources/poon-paper.pdf)
* [Knowledge-Based Weak Supervision for Information Extraction of Overlapping Relations](https://homes.cs.washington.edu/~clzhang/paper/acl2011.pdf)


## Blogs
* Blog Post on [Deep Learning, NLP, and Representations](http://colah.github.io/posts/2014-07-NLP-RNNs-Representations/)
* Blog Post on [NLP Tutorial](http://www.vikparuchuri.com/blog/natural-language-processing-tutorial/)
* [Natural Language Processing Blog](http://nlpers.blogspot.ch/) by Hal Daumé III
* [Machine Learning Blog](https://bmcfee.github.io/#home) by Brian McFee


## Credits
part of the lists are from 
* [ai-reading-list](https://github.com/m0nologuer/AI-reading-list) 
* [nlp-reading-group](https://github.com/clulab/nlp-reading-group/wiki/Fall-2015-Reading-Schedule/_edit)
* [awesome-spanish-nlp](https://github.com/dav009/awesome-spanish-nlp)
* [jjangsangy's awesome-nlp](https://gist.github.com/jjangsangy/8759f163bc3558779c46)
* [awesome-machine-learning](https://github.com/josephmisiti/awesome-machine-learning/edit/master/README.md)
* [DL4NLP](https://github.com/andrewt3000/DL4NLP)

==============================================================================================

# Top-down learning path: Machine Learning for Software Engineers

<p align="center">
  <a href="https://github.com/ZuzooVn/machine-learning-for-software-engineers">
    <img alt="Top-down learning path: Machine Learning for Software Engineers" src="https://img.shields.io/badge/Machine%20Learning-Software%20Engineers-blue.svg">
  </a>
  <a href="https://github.com/ZuzooVn/machine-learning-for-software-engineers/stargazers">
    <img alt="GitHub stars" src="https://img.shields.io/github/stars/ZuzooVn/machine-learning-for-software-engineers.svg">
  </a>
  <a href="https://github.com/ZuzooVn/machine-learning-for-software-engineers/network">
    <img alt="GitHub forks" src="https://img.shields.io/github/forks/ZuzooVn/machine-learning-for-software-engineers.svg">
  </a>
</p>

Inspired by [Google Interview University](https://github.com/jwasham/google-interview-university).

Translations: [Brazilian Portuguese](https://github.com/ZuzooVn/machine-learning-for-software-engineers/blob/master/README-pt-BR.md) | [中文版本](https://github.com/ZuzooVn/machine-learning-for-software-engineers/blob/master/README-zh-CN.md)

## What is it?

This is my multi-month study plan for going from mobile developer (self-taught, no CS degree) to machine learning engineer.

My main goal was to find an approach to studying Machine Learning that is mainly hands-on and abstracts most of the Math for the beginner. 
This approach is unconventional because it’s the top-down and results-first approach designed for software engineers.

Please, feel free to make any contributions you feel will make it better.

---

## Table of Contents

- [What is it?](#what-is-it)
- [Why use it?](#why-use-it)
- [How to use it](#how-to-use-it)
- [Follow me](#follow-me)
- [Don't feel you aren't smart enough](#dont-feel-you-arent-smart-enough)
- [About Video Resources](#about-video-resources)
- [Prerequisite Knowledge](#prerequisite-knowledge)
- [The Daily Plan](#the-daily-plan)
- [Motivation](#motivation)
- [Machine learning overview](#machine-learning-overview)
- [Machine learning mastery](#machine-learning-mastery)
- [Machine learning is fun](#machine-learning-is-fun)
- [Inky Machine Learning](#inky-machine-learning)
- [Machine learning: an in-depth, non-technical guide](#machine-learning-an-in-depth-non-technical-guide)
- [Stories and experiences](#stories-and-experiences)
- [Machine Learning Algorithms](#machine-learning-algorithms)
- [Beginner Books](#beginner-books)
- [Practical Books](#practical-books)
- [Kaggle knowledge competitions](#kaggle-knowledge-competitions)
- [Video Series](#video-series)
- [MOOC](#mooc)
- [Resources](#resources)
- [Becoming an Open Source Contributor](#becoming-an-open-source-contributor)
- [Games](#games)
- [Podcasts](#podcasts)
- [Communities](#communities)
- [Interview Questions](#interview-questions)
- [My admired companies](#my-admired-companies)

---

## Why use it?

I'm following this plan to prepare for my near-future job: Machine learning engineer. I've been building native mobile applications (Android/iOS/Blackberry) since 2011. I have a Software Engineering degree, not a Computer Science degree. I have an itty-bitty amount of basic knowledge about: Calculus, Linear Algebra, Discrete Mathematics, Probability & Statistics from university.
Think about my interest in machine learning:
- [Can I learn and get a job in Machine Learning without studying CS Master and PhD?](https://www.quora.com/Can-I-learn-and-get-a-job-in-Machine-Learning-without-studying-CS-Master-and-PhD)
    - You can, but it is far more difficult than when I got into the field.
- [How do I get a job in Machine Learning as a software programmer who self-studies Machine Learning, but  never has a chance to use it at work?](https://www.quora.com/How-do-I-get-a-job-in-Machine-Learning-as-a-software-programmer-who-self-studies-Machine-Learning-but-never-has-a-chance-to-use-it-at-work)
    - I'm hiring machine learning experts for my team and your MOOC will not get you the job (there is better news below). In fact, many people with a master's in machine learning will not get the job because they (and most who have taken MOOCs) do not have a deep understanding that will help me solve my problems
- [What skills are needed for machine learning jobs?](http://programmers.stackexchange.com/questions/79476/what-skills-are-needed-for-machine-learning-jobs)
    - First, you need to have a decent CS/Math background. ML is an advanced topic so most textbooks assume that you have that background. Second, machine learning is a very general topic with many sub-specialties requiring unique skills. You may want to browse the curriculum of an MS program in Machine Learning to see the course, curriculum and textbook.
    - Statistics, Probability, distributed computing, and Statistics.

I find myself in times of trouble.

AFAIK, [There are two sides to machine learning](http://machinelearningmastery.com/programmers-can-get-into-machine-learning/):
- Practical Machine Learning: This is about querying databases, cleaning data, writing scripts to transform data and gluing algorithm and libraries together and writing custom code to squeeze reliable answers from data to satisfy difficult and ill-defined questions. It’s the mess of reality.
- Theoretical Machine Learning: This is about math and abstraction and idealized scenarios and limits and beauty and informing what is possible. It is a whole lot neater and cleaner and removed from the mess of reality.

I think the best way for practice-focused methodology is something like ['practice — learning — practice'](http://machinelearningmastery.com/machine-learning-for-programmers/#comment-358985), that means where students first come with some existing projects with problems and solutions (practice) to get familiar with traditional methods in the area and perhaps also with their methodology. After practicing with some elementary experiences, they can go into the books and study the underlying theory, which serves to guide their future advanced practice and will enhance their toolbox of solving practical problems. Studying theory also further improves their understanding on the elementary experiences, and will help them acquire advanced experiences more quickly.

 It's a long plan. It's going to take me years. If you are familiar with a lot of this already it will take you a lot less time.

## How to use it
Everything below is an outline, and you should tackle the items in order from top to bottom.

I'm using Github's special markdown flavor, including tasks lists to check progress.

- [x] Create a new branch so you can check items like this, just put an x in the brackets: [x]

[More about Github-flavored markdown](https://guides.github.com/features/mastering-markdown/#GitHub-flavored-markdown)

## Follow me
I'm a Vietnamese Software Engineer who is really passionate and wants to work in the USA.

How much did I work during this plan? Roughly 4 hours/night after a long, hard day at work.

I'm on the journey. 

- Twitter: [@Nam Vu](https://twitter.com/zuzoovn)

| ![Nam Vu - Top-down learning path: machine learning for software engineers](http://sv1.upsieutoc.com/2016/10/08/331f241c8da44d0c43e9324d55440db6.md.jpg)|
|:---:|
| USA as heck | 

## Don't feel you aren't smart enough
I get discouraged from books and courses that tell me as soon as I open them that multivariate calculus, inferential statistics and linear algebra are prerequisites. I still don’t know how to get started…

- [What if I’m Not Good at Mathematics](http://machinelearningmastery.com/what-if-im-not-good-at-mathematics/)
- [5 Techniques To Understand Machine Learning Algorithms Without the Background in Mathematics](http://machinelearningmastery.com/techniques-to-understand-machine-learning-algorithms-without-the-background-in-mathematics/)
- [How do I learn machine learning?](https://www.quora.com/Machine-Learning/How-do-I-learn-machine-learning-1)

## About Video Resources

Some videos are available only by enrolling in a Coursera or EdX class. It is free to do so, but sometimes the classes
are no longer in session so you have to wait a couple of months, so you have no access. I'm going to be adding more videos
from public sources and replacing the online course videos over time. I like using university lectures.

## Prerequisite Knowledge

This short section were prerequisites/interesting info I wanted to learn before getting started on the daily plan.

- [ ] [What is the difference between Data Analytics, Data Analysis, Data Mining, Data Science, Machine Learning, and Big Data?](https://www.quora.com/What-is-the-difference-between-Data-Analytics-Data-Analysis-Data-Mining-Data-Science-Machine-Learning-and-Big-Data-1)
- [ ] [Learning How to Learn](https://www.coursera.org/learn/learning-how-to-learn)
- [ ] [Don’t Break The Chain](http://lifehacker.com/281626/jerry-seinfelds-productivity-secret)
- [ ] [How to learn on your own](https://metacademy.org/roadmaps/rgrosse/learn_on_your_own)

## The Daily Plan

Each subject does not require a whole day to be able to understand it fully, and you can do multiple of these in a day.

Each day I take one subject from the list below, read it cover to cover, take notes, do the exercises and write an implementation in Python or R.

# Motivation
- [ ] [Dream](https://www.youtube.com/watch?v=g-jwWYX7Jlo)

## Machine learning overview
- [ ] [A Visual Introduction to Machine Learning](http://www.r2d3.us/visual-intro-to-machine-learning-part-1/)
- [ ] [A Gentle Guide to Machine Learning](https://blog.monkeylearn.com/a-gentle-guide-to-machine-learning/)
- [ ] [Introduction to Machine Learning for Developers](http://blog.algorithmia.com/introduction-machine-learning-developers/)
- [ ] [Machine Learning basics for a newbie](https://www.analyticsvidhya.com/blog/2015/06/machine-learning-basics/)
- [ ] [How do you explain Machine Learning and Data Mining to non Computer Science people?](https://www.quora.com/How-do-you-explain-Machine-Learning-and-Data-Mining-to-non-Computer-Science-people)
- [ ] [Machine Learning: Under the hood. Blog post explains the principles of machine learning in layman terms. Simple and clear](https://georgemdallas.wordpress.com/2013/06/11/big-data-data-mining-and-machine-learning-under-the-hood/)
- [ ] [What is machine learning, and how does it work?](https://www.youtube.com/watch?v=elojMnjn4kk&list=PL5-da3qGB5ICeMbQuqbbCOQWcS6OYBr5A&index=1)
- [ ] [Deep Learning - A Non-Technical Introduction](http://www.slideshare.net/AlfredPong1/deep-learning-a-nontechnical-introduction-69385936)

## Machine learning mastery
- [ ] [The Machine Learning Mastery Method](http://machinelearningmastery.com/machine-learning-mastery-method/)
- [ ] [Machine Learning for Programmers](http://machinelearningmastery.com/machine-learning-for-programmers/)
- [ ] [Applied Machine Learning with Machine Learning Mastery](http://machinelearningmastery.com/start-here/)
- [ ] [Python Machine Learning Mini-Course](http://machinelearningmastery.com/python-machine-learning-mini-course/)
- [ ] [Machine Learning Algorithms Mini-Course](http://machinelearningmastery.com/machine-learning-algorithms-mini-course/)

## Machine learning is fun
- [ ] [Machine Learning is Fun!](https://medium.com/@ageitgey/machine-learning-is-fun-80ea3ec3c471#.37ue6caww)
- [ ] [Part 2: Using Machine Learning to generate Super Mario Maker levels](https://medium.com/@ageitgey/machine-learning-is-fun-part-2-a26a10b68df3#.kh7qgvp1b)
- [ ] [Part 3: Deep Learning and Convolutional Neural Networks](https://medium.com/@ageitgey/machine-learning-is-fun-part-3-deep-learning-and-convolutional-neural-networks-f40359318721#.44rhxy637)
- [ ] [Part 4: Modern Face Recognition with Deep Learning](https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78#.3rwmq0ddc)
- [ ] [Part 5: Language Translation with Deep Learning and the Magic of Sequences](https://medium.com/@ageitgey/machine-learning-is-fun-part-5-language-translation-with-deep-learning-and-the-magic-of-sequences-2ace0acca0aa#.wyfthap4c)

## [Inky Machine Learning](https://triskell.github.io/2016/11/15/Inky-Machine-Learning.html)
- [ ] [Part 1: What is Machine Learning ?](https://triskell.github.io/2016/10/23/What-is-Machine-Learning.html)
- [ ] [Part 2: Supervised Learning and Unsupervised Learning](https://triskell.github.io/2016/11/13/Supervised-Learning-and-Unsupervised-Learning.html)

## Machine learning: an in-depth, non-technical guide
- [ ] [Overview, goals, learning types, and algorithms](http://www.innoarchitech.com/machine-learning-an-in-depth-non-technical-guide/)
- [ ] [Data selection, preparation, and modeling](http://www.innoarchitech.com/machine-learning-an-in-depth-non-technical-guide-part-2/)
- [ ] [Model evaluation, validation, complexity, and improvement](http://www.innoarchitech.com/machine-learning-an-in-depth-non-technical-guide-part-3/)
- [ ] [Model performance and error analysis](http://www.innoarchitech.com/machine-learning-an-in-depth-non-technical-guide-part-4/)
- [ ] [Unsupervised learning, related fields, and machine learning in practice](http://www.innoarchitech.com/machine-learning-an-in-depth-non-technical-guide-part-5/)

## Stories and experiences
- [ ] [Machine Learning in a Week](https://medium.com/learning-new-stuff/machine-learning-in-a-week-a0da25d59850#.tk6ft2kcg)
- [ ] [Machine Learning in a Year](https://medium.com/learning-new-stuff/machine-learning-in-a-year-cdb0b0ebd29c#.hhcb9fxk1)
- [ ] [How I wrote my first Machine Learning program in 3 days](http://blog.adnansiddiqi.me/how-i-wrote-my-first-machine-learning-program-in-3-days/)
- [ ] [Learning Path : Your mentor to become a machine learning expert](https://www.analyticsvidhya.com/learning-path-learn-machine-learning/)
- [ ] [You Too Can Become a Machine Learning Rock Star! No PhD](https://backchannel.com/you-too-can-become-a-machine-learning-rock-star-no-phd-necessary-107a1624d96b#.g9p16ldp7)
- [ ] How to become a Data Scientist in 6 months: A hacker’s approach to career planning
    - [Video](https://www.youtube.com/watch?v=rIofV14c0tc)
    - [Slide](http://www.slideshare.net/TetianaIvanova2/how-to-become-a-data-scientist-in-6-months)
- [ ] [5 Skills You Need to Become a Machine Learning Engineer](http://blog.udacity.com/2016/04/5-skills-you-need-to-become-a-machine-learning-engineer.html)
- [ ] [Are you a self-taught machine learning engineer? If yes, how did you do it & how long did it take you?](https://www.quora.com/Are-you-a-self-taught-machine-learning-engineer-If-yes-how-did-you-do-it-how-long-did-it-take-you)
- [ ] [How can one become a good machine learning engineer?](https://www.quora.com/How-can-one-become-a-good-machine-learning-engineer)
- [ ] [A Learning Sabbatical focused on Machine Learning](http://karlrosaen.com/ml/)

## Machine Learning Algorithms
- [ ] [10 Machine Learning Algorithms Explained to an ‘Army Soldier’](https://www.analyticsvidhya.com/blog/2015/12/10-machine-learning-algorithms-explained-army-soldier/)
- [ ] [Top 10 data mining algorithms in plain English](https://rayli.net/blog/data/top-10-data-mining-algorithms-in-plain-english/)
- [ ] [10 Machine Learning Terms Explained in Simple English](http://blog.aylien.com/10-machine-learning-terms-explained-in-simple/)
- [ ] [A Tour of Machine Learning Algorithms](http://machinelearningmastery.com/a-tour-of-machine-learning-algorithms/)
- [ ] [The 10 Algorithms Machine Learning Engineers Need to Know](https://gab41.lab41.org/the-10-algorithms-machine-learning-engineers-need-to-know-f4bb63f5b2fa#.ofc7t2965)
- [ ] [Comparing supervised learning algorithms](http://www.dataschool.io/comparing-supervised-learning-algorithms/)
- [Machine Learning Algorithms: A collection of minimal and clean implementations of machine learning algorithms](https://github.com/rushter/MLAlgorithms)

## Beginner Books
- [ ] [Data Smart: Using Data Science to Transform Information into Insight 1st Edition](https://www.amazon.com/Data-Smart-Science-Transform-Information/dp/111866146X)
- [ ] [Data Science for Business: What you need to know about data mining and data­ analytic-thinking](https://www.amazon.com/Data-Science-Business-Data-Analytic-Thinking/dp/1449361323/)
- [ ] [Predictive Analytics: The Power to Predict Who Will Click, Buy, Lie, or Die](https://www.amazon.com/Predictive-Analytics-Power-Predict-Click/dp/1118356853)

## Practical Books
- [ ] [Machine Learning for Hackers](https://www.amazon.com/Machine-Learning-Hackers-Drew-Conway/dp/1449303714)
    - [GitHub repository(R)](https://github.com/johnmyleswhite/ML_for_Hackers)
    - [GitHub repository(Python)](https://github.com/carljv/Will_it_Python)
- [ ] [Python Machine Learning](https://www.amazon.com/Python-Machine-Learning-Sebastian-Raschka-ebook/dp/B00YSILNL0)
    - [GitHub repository](https://github.com/rasbt/python-machine-learning-book)
- [ ] [Programming Collective Intelligence: Building Smart Web 2.0 Applications](https://www.amazon.com/Programming-Collective-Intelligence-Building-Applications-ebook/dp/B00F8QDZWG)
- [ ] [Machine Learning: An Algorithmic Perspective, Second Edition](https://www.amazon.com/Machine-Learning-Algorithmic-Perspective-Recognition/dp/1466583282)
    - [GitHub repository](https://github.com/alexsosn/MarslandMLAlgo)
    - [Resource repository](http://seat.massey.ac.nz/personal/s.r.marsland/MLbook.html)
- [ ] [Introduction to Machine Learning with Python: A Guide for Data Scientists](http://shop.oreilly.com/product/0636920030515.do)
    - [GitHub repository](https://github.com/amueller/introduction_to_ml_with_python)
- [ ] [Data Mining: Practical Machine Learning Tools and Techniques, Third Edition](https://www.amazon.com/Data-Mining-Practical-Techniques-Management/dp/0123748569)
    - Teaching material
        - [Slides for Chapters 1-5 (zip)](http://www.cs.waikato.ac.nz/ml/weka/Slides3rdEd_Ch1-5.zip)
        - [Slides for Chapters 6-8 (zip)](http://www.cs.waikato.ac.nz/ml/weka/Slides3rdEd_Ch6-8.zip)
- [ ] [Machine Learning in Action](https://www.amazon.com/Machine-Learning-Action-Peter-Harrington/dp/1617290181/)
    - [GitHub repository](https://github.com/pbharrin/machinelearninginaction)
- [ ] [Reactive Machine Learning Systems(MEAP)](https://www.manning.com/books/reactive-machine-learning-systems)
    - [GitHub repository](https://github.com/jeffreyksmithjr/reactive-machine-learning-systems)
- [ ] [An Introduction to Statistical Learning](http://www-bcf.usc.edu/~gareth/ISL/)
    - [GitHub repository(R)](http://www-bcf.usc.edu/~gareth/ISL/code.html)
    - [GitHub repository(Python)](https://github.com/JWarmenhoven/ISLR-python)
    - [Videos](http://www.dataschool.io/15-hours-of-expert-machine-learning-videos/)
- [ ] [Building Machine Learning Systems with Python](https://www.packtpub.com/big-data-and-business-intelligence/building-machine-learning-systems-python)
    - [GitHub repository](https://github.com/luispedro/BuildingMachineLearningSystemsWithPython)
- [ ] [Learning scikit-learn: Machine Learning in Python](https://www.packtpub.com/big-data-and-business-intelligence/learning-scikit-learn-machine-learning-python)
    - [GitHub repository](https://github.com/gmonce/scikit-learn-book)
- [ ] [Probabilistic Programming & Bayesian Methods for Hackers](https://camdavidsonpilon.github.io/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/)
- [ ] [Probabilistic Graphical Models: Principles and Techniques](https://www.amazon.com/Probabilistic-Graphical-Models-Principles-Computation/dp/0262013193)
- [ ] [Machine Learning: Hands-On for Developers and Technical Professionals](https://www.amazon.com/Machine-Learning-Hands-Developers-Professionals/dp/1118889061)
    - [Machine Learning Hands-On for Developers and Technical Professionals review](https://blogs.msdn.microsoft.com/querysimon/2015/01/01/book-review-machine-learning-hands-on-for-developers-and-technical-professionals/)
    - [GitHub repository](https://github.com/jasebell/mlbook)
- [ ] [Learning from Data](https://www.amazon.com/Learning-Data-Yaser-S-Abu-Mostafa/dp/1600490069)
    - [Online tutorials](https://work.caltech.edu/telecourse.html)
- [ ] [Reinforcement Learning: An Introduction (2nd Edition)](https://webdocs.cs.ualberta.ca/~sutton/book/the-book-2nd.html)
    - [GitHub repository](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction)
- [ ] [Machine Learning with TensorFlow(MEAP)](https://www.manning.com/books/machine-learning-with-tensorflow)

## Kaggle knowledge competitions
- [ ] [Kaggle Competitions: How and where to begin?](https://www.analyticsvidhya.com/blog/2015/06/start-journey-kaggle/)
- [ ] [How a Beginner Used Small Projects To Get Started in Machine Learning and Compete on Kaggle](http://machinelearningmastery.com/how-a-beginner-used-small-projects-to-get-started-in-machine-learning-and-compete-on-kaggle)
- [ ] [Master Kaggle By Competing Consistently](http://machinelearningmastery.com/master-kaggle-by-competing-consistently/)

## Video Series
- [ ] [Machine Learning for Hackers](https://www.youtube.com/playlist?list=PL2-dafEMk2A4ut2pyv0fSIXqOzXtBGkLj)
- [ ] [Fresh Machine Learning](https://www.youtube.com/playlist?list=PL2-dafEMk2A6Kc7pV6gHH-apBFxwFjKeY)
- [ ] [Machine Learning Recipes with Josh Gordon](https://www.youtube.com/playlist?list=PLOU2XLYxmsIIuiBfYad6rFYQU_jL2ryal)
- [ ] [Everything You Need to know about Machine Learning in 30 Minutes or Less](https://vimeo.com/43547079)
- [ ] [A Friendly Introduction to Machine Learning](https://www.youtube.com/watch?v=IpGxLWOIZy4)
- [ ] [Nuts and Bolts of Applying Deep Learning - Andrew Ng](https://www.youtube.com/watch?v=F1ka6a13S9I)
- [ ] BigML Webinar
    - [Video](https://www.youtube.com/watch?list=PL1bKyu9GtNYHcjGa6ulrvRVcm1lAB8he3&v=W62ehrnOVqo)
    - [Resources](https://bigml.com/releases)
- [ ] [mathematicalmonk's Machine Learning tutorials](https://www.youtube.com/playlist?list=PLD0F06AA0D2E8FFBA)
- [ ] [Machine learning in Python with scikit-learn](https://www.youtube.com/playlist?list=PL5-da3qGB5ICeMbQuqbbCOQWcS6OYBr5A)
    - [GitHub repository](https://github.com/justmarkham/scikit-learn-videos)
    - [Blog](http://blog.kaggle.com/author/kevin-markham/)
- [ ] [My playlist – Top YouTube Videos on Machine Learning, Neural Network & Deep Learning](https://www.analyticsvidhya.com/blog/2015/07/top-youtube-videos-machine-learning-neural-network-deep-learning/)
- [ ] [16 New Must Watch Tutorials, Courses on Machine Learning](https://www.analyticsvidhya.com/blog/2016/10/16-new-must-watch-tutorials-courses-on-machine-learning/)
- [ ] [DeepLearning.TV](https://www.youtube.com/channel/UC9OeZkIwhzfv-_Cb7fCikLQ)
- [ ] [Learning To See](https://www.youtube.com/playlist?list=PLiaHhY2iBX9ihLasvE8BKnS2Xg8AhY6iV)

## MOOC
- [ ] [Udacity’s Intro to Machine Learning](https://www.udacity.com/course/intro-to-machine-learning--ud120)
    - [Udacity Intro to Machine Learning Review](http://hamelg.blogspot.com/2014/12/udacity-intro-to-machine-learning-review.html)
- [ ] [Udacity’s Supervised, Unsupervised & Reinforcement](https://www.udacity.com/course/machine-learning--ud262)
- [ ] [Machine Learning Foundations: A Case Study Approach](https://www.coursera.org/learn/ml-foundations)
- [ ] [Coursera’s Machine Learning](https://www.coursera.org/learn/machine-learning)
    - [Video only](https://www.youtube.com/playlist?list=PLZ9qNFMHZ-A4rycgrgOYma6zxF4BZGGPW)
    - [Coursera Machine Learning review](https://rayli.net/blog/data/coursera-machine-learning-review/)
    - [Coursera: Machine Learning Roadmap](https://metacademy.org/roadmaps/cjrd/coursera_ml_supplement)
- [ ] [Machine Learning Distilled](https://code.tutsplus.com/courses/machine-learning-distilled)
- [ ] [BigML training](https://bigml.com/training)
- [ ] [Coursera’s Neural Networks for Machine Learning](https://www.coursera.org/learn/neural-networks)
    - Taught by Geoffrey Hinton, a pioneer in the field of neural networks
- [ ] [Machine Learning - CS - Oxford University](https://www.cs.ox.ac.uk/people/nando.defreitas/machinelearning/)
- [ ] [Creative Applications of Deep Learning with TensorFlow](https://www.kadenze.com/courses/creative-applications-of-deep-learning-with-tensorflow/info)
- [ ] [Intro to Descriptive Statistics](https://www.udacity.com/course/intro-to-descriptive-statistics--ud827)
- [ ] [Intro to Inferential Statistics](https://www.udacity.com/course/intro-to-inferential-statistics--ud201)

## Resources
- [ ] [Learn Machine Learning in a Single Month](https://elitedatascience.com/machine-learning-masterclass)
- [ ] [The Non-Technical Guide to Machine Learning & Artificial Intelligence](https://medium.com/@samdebrule/a-humans-guide-to-machine-learning-e179f43b67a0#.cpzf3a5c0)
- [ ] [Machine Learning for Software Engineers on Hacker News](https://news.ycombinator.com/item?id=12898718)
- [ ] [Machine Learning for Developers](https://xyclade.github.io/MachineLearning/) 
- [ ] [Machine Learning Advice for Developers](https://dev.to/thealexlavin/machine-learning-advice-for-developers)
- [ ] [Machine Learning For Complete Beginners](http://pythonforengineers.com/machine-learning-for-complete-beginners/)
- [ ] [Getting Started with Machine Learning: For absolute beginners and fifth graders](https://medium.com/@suffiyanz/getting-started-with-machine-learning-f15df1c283ea#.yjtiy7ei9)
- [ ] [How to Learn Machine Learning: The Self-Starter Way](https://elitedatascience.com/learn-machine-learning)
- [ ] [Machine Learning Self-study Resources](https://ragle.sanukcode.net/articles/machine-learning-self-study-resources/)
- [ ] [Level-Up Your Machine Learning](https://metacademy.org/roadmaps/cjrd/level-up-your-ml)
- [ ] Enough Machine Learning to Make Hacker News Readable Again
    - [Video](https://www.youtube.com/watch?v=O7IezJT9uSI)
    - [Slide](https://speakerdeck.com/pycon2014/enough-machine-learning-to-make-hacker-news-readable-again-by-ned-jackson-lovely)
- [ ] [Dive into Machine Learning](https://github.com/hangtwenty/dive-into-machine-learning)
- Machine Learning courses in Universities
    - [ ] [Stanford](http://ai.stanford.edu/courses/)
    - [ ] [Machine Learning Summer Schools](http://mlss.cc/)
    - [ ] [Oxford](https://www.cs.ox.ac.uk/people/nando.defreitas/machinelearning/)
    - [ ] [Cambridge](http://mlg.eng.cam.ac.uk/)
- Flipboard Topics
    - [Machine learning](https://flipboard.com/topic/machinelearning)
    - [Deep learning](https://flipboard.com/topic/deeplearning)
    - [Artificial Intelligence](https://flipboard.com/topic/artificialintelligence)
- Medium Topics
    - [Machine learning](https://medium.com/tag/machine-learning/latest)
    - [Deep learning](https://medium.com/tag/deep-learning)
    - [Artificial Intelligence](https://medium.com/tag/artificial-intelligence)
- Monthly top 10 articles
    - Machine Learning
        - [July 2016](https://medium.mybridge.co/top-ten-machine-learning-articles-for-the-past-month-9c1202351144#.lyycen64y)
        - [August 2016](https://medium.mybridge.co/machine-learning-top-10-articles-for-the-past-month-2f3cb815ffed#.i9ee7qkjz)
        - [September 2016](https://medium.mybridge.co/machine-learning-top-10-in-september-6838169e9ee7#.4jbjcibft)
        - [October 2016](https://medium.mybridge.co/machine-learning-top-10-articles-for-the-past-month-35c37825a943#.td5im1p5z)
        - [November 2016](https://medium.mybridge.co/machine-learning-top-10-articles-for-the-past-month-b499e4213a34#.7k39i08tv)
    - Algorithms
        - [September 2016](https://medium.mybridge.co/algorithm-top-10-articles-in-september-8a0e6afb0807#.hgjzuyxdb)
        - [October-November 2016](https://medium.mybridge.co/algorithm-top-10-articles-v-november-e73cba2fa87e#.kothimkhb)
- [Comprehensive list of data science resources](http://www.datasciencecentral.com/group/resources/forum/topics/comprehensive-list-of-data-science-resources)
- [DigitalMind's Artificial Intelligence resources](http://blog.digitalmind.io/post/artificial-intelligence-resources)
- [Awesome Machine Learning](https://github.com/josephmisiti/awesome-machine-learning)
- [CreativeAi's Machine Learning](http://www.creativeai.net/?cat%5B0%5D=machine-learning)

## Games
- [Halite: A.I. Coding Game](https://halite.io/)
- [Vindinium: A.I. Programming Challenge](http://vindinium.org/)
- [General Video Game AI Competition](http://www.gvgai.net/)
- [Angry Birds AI Competition](https://aibirds.org/)
- [The AI Games](http://theaigames.com/)
- [Fighting Game AI Competition](http://www.ice.ci.ritsumei.ac.jp/~ftgaic/)
- [CodeCup](http://www.codecup.nl/intro.php)
- [Student StarCraft AI Tournament](http://sscaitournament.com/)
- [AIIDE StarCraft AI Competition](http://www.cs.mun.ca/~dchurchill/starcraftaicomp/)
- [CIG StarCraft AI Competition](https://sites.google.com/site/starcraftaic/)
- [CodinGame - AI Bot Games](https://www.codingame.com/training/machine-learning)

## Becoming an Open Source Contributor
- [ ] [tensorflow/magenta: Magenta: Music and Art Generation with Machine Intelligence](https://github.com/tensorflow/magenta)
- [ ] [tensorflow/tensorflow: Computation using data flow graphs for scalable machine learning](https://github.com/tensorflow/tensorflow)
- [ ] [cmusatyalab/openface: Face recognition with deep neural networks.](https://github.com/cmusatyalab/openface)
- [ ] [tensorflow/models/syntaxnet: Neural Models of Syntax.](https://github.com/tensorflow/models/tree/master/syntaxnet)

## Podcasts
- ### Podcasts for Beginners:
    - [Talking Machines](http://www.thetalkingmachines.com/)
    - [Linear Digressions](http://lineardigressions.com/)
    - [Data Skeptic](http://dataskeptic.com/)
    - [This Week in Machine Learning & AI](https://twimlai.com/)

- ### "More" advanced podcasts
    - [Partially Derivative](http://partiallyderivative.com/)
    - [O’Reilly Data Show](http://radar.oreilly.com/tag/oreilly-data-show-podcast)
    - [Not So Standard Deviation](https://soundcloud.com/nssd-podcast)

- ### Podcasts to think outside the box:
    - [Data Stories](http://datastori.es/)

## Communities
- Quora
    - [Machine Learning](https://www.quora.com/topic/Machine-Learning)
    - [Statistics](https://www.quora.com/topic/Statistics-academic-discipline)
    - [Data Mining](https://www.quora.com/topic/Data-Mining)

- Reddit
    - [Machine Learning](https://www.reddit.com/r/machinelearning)
    - [Computer Vision](https://www.reddit.com/r/computervision)
    - [Natural Language](https://www.reddit.com/r/languagetechnology)
    - [Data Science](https://www.reddit.com/r/datascience)
    - [Big Data](https://www.reddit.com/r/bigdata)
    - [Statistics](https://www.reddit.com/r/statistics)

- [Data Tau](http://www.datatau.com/)

- [Deep Learning News](http://news.startup.ml/)

- [KDnuggets](http://www.kdnuggets.com/)

## Interview Questions
- [ ] [How To Prepare For A Machine Learning Interview](http://blog.udacity.com/2016/05/prepare-machine-learning-interview.html)
- [ ] [40 Interview Questions asked at Startups in Machine Learning / Data Science](https://www.analyticsvidhya.com/blog/2016/09/40-interview-questions-asked-at-startups-in-machine-learning-data-science)
- [ ] [21 Must-Know Data Science Interview Questions and Answers](http://www.kdnuggets.com/2016/02/21-data-science-interview-questions-answers.html)
- [ ] [Top 50 Machine learning Interview questions & Answers](http://career.guru99.com/top-50-interview-questions-on-machine-learning/)
- [ ] [Machine Learning Engineer interview questions](https://resources.workable.com/machine-learning-engineer-interview-questions)
- [ ] [Popular Machine Learning Interview Questions](http://www.learn4master.com/machine-learning/popular-machine-learning-interview-questions)
- [ ] [What are some common Machine Learning interview questions?](https://www.quora.com/What-are-some-common-Machine-Learning-interview-questions)
- [ ] [What are the best interview questions to evaluate a machine learning researcher?](https://www.quora.com/What-are-the-best-interview-questions-to-evaluate-a-machine-learning-researcher)
- [ ] [Collection of Machine Learning Interview Questions](http://analyticscosm.com/machine-learning-interview-questions-for-data-scientist-interview/)
- [ ] [121 Essential Machine Learning Questions & Answers](https://learn.elitedatascience.com/mlqa-welcome)


## My admired companies
- [ ] [ELSA - Your virtual pronunciation coach](https://www.elsanow.io/home)

============================================================================================

#### 2016-12

- Knowing When to Look: Adaptive Attention via A Visual Sentinel for Image Captioning [[arXiv](https://arxiv.org/abs/1612.01887)]
- Overcoming catastrophic forgetting in neural networks [[arXiv](https://arxiv.org/abs/1612.00796)]

#### 2016-11 (ICLR Edition)

- Image-to-Image Translation with Conditional Adversarial Networks [[arXiv](https://arxiv.org/abs/1611.07004)]
- [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](notes/mixture-experts.md) [[OpenReview](https://openreview.net/forum?id=B1ckMDqlg)]
- Learning to reinforcement learn [[arXiv](https://arxiv.org/abs/1611.05763)]
- A Way out of the Odyssey: Analyzing and Combining Recent Insights for LSTMs [[arXiv](https://arxiv.org/abs/1611.05104)]
- [Adversarial Training Methods for Semi-Supervised Text Classification](notes/adversarial-text-classification.md) [[arXiv](https://arxiv.org/abs/1605.07725)]
- Importance Sampling with Unequal Support [[arXiv](https://arxiv.org/abs/1611.03451)]
- Quasi-Recurrent Neural Networks [[arXiv](https://arxiv.org/abs/1611.01576)]
- Capacity and Learnability in Recurrent Neural Networks [[OpenReview](http://openreview.net/forum?id=BydARw9ex)]
- Unrolled Generative Adversarial Networks [[OpenReview](http://openreview.net/forum?id=BydrOIcle)]
- Deep Information Propagation [[OpenReview](http://openreview.net/forum?id=H1W1UN9gg)]
- Structured Attention Networks [[OpenReview](http://openreview.net/forum?id=HkE0Nvqlg)]
- Incremental Sequence Learning [[arXiv](https://arxiv.org/abs/1611.03068)]
- b-GAN: Unified Framework of Generative Adversarial Networks [[OpenReview](http://openreview.net/forum?id=S1JG13oee)]
- A Joint Many-Task Model: Growing a Neural Network for Multiple NLP Tasks [[OpenReview](http://openreview.net/forum?id=SJZAb5cel)]
- Categorical Reparameterization with Gumbel-Softmax [[arXiv](https://arxiv.org/abs/1611.01144)]
- Lip Reading Sentences in the Wild [[arXiv](https://arxiv.org/abs/1611.05358)]

Reinforcement Learning:

-Learning to reinforcement learn [[arXiv](https://arxiv.org/abs/1611.05763)]
- A Connection between Generative Adversarial Networks, Inverse Reinforcement Learning, and Energy-Based Models [[arXiv](https://arxiv.org/abs/1611.03852)]
- The Predictron: End-To-End Learning and Planning [[OpenReview](http://openreview.net/forum?id=BkJsCIcgl)]
- [Third-Person Imitation Learning](notes/third-person-imitation-learning.md) [[OpenReview](http://openreview.net/forum?id=B16dGcqlx)]
- Generalizing Skills with Semi-Supervised Reinforcement Learning [[OpenReview](http://openreview.net/forum?id=ryHlUtqge)]
- Sample Efficient Actor-Critic with Experience Replay [[OpenReview](http://openreview.net/forum?id=HyM25Mqel)]
- [Reinforcement Learning with Unsupervised Auxiliary Tasks](notes/rl-auxiliary-tasks.md) [[arXiv](https://arxiv.org/abs/1611.05397)]
- Neural Architecture Search with Reinforcement Learning [[OpenReview](http://openreview.net/forum?id=r1Ue8Hcxg)]
- Towards Information-Seeking Agents [[OpenReview](http://openreview.net/forum?id=SyW2QSige)]
- Multi-Agent Cooperation and the Emergence of (Natural) Language [[OpenReview](http://openreview.net/forum?id=Hk8N3Sclg)]
- Improving Policy Gradient by Exploring Under-appreciated Rewards [[OpenReview](http://openreview.net/forum?id=ryT4pvqll)]
- Stochastic Neural Networks for Hierarchical Reinforcement Learning [[OpenReview](http://openreview.net/forum?id=B1oK8aoxe)]
- Tuning Recurrent Neural Networks with Reinforcement Learning [[OpenReview](https://arxiv.org/abs/1611.02796)]
- RL^2: Fast Reinforcement Learning via Slow Reinforcement Learning [[arXiv](https://arxiv.org/abs/1611.02779)]
- Learning Invariant Feature Spaces to Transfer Skills with Reinforcement Learning [[OpenReview](http://openreview.net/forum?id=Hyq4yhile)]
- Learning to Perform Physics Experiments via Deep Reinforcement Learning [[OpenReview](http://openreview.net/forum?id=r1nTpv9eg)]
- Reinforcement Learning through Asynchronous Advantage Actor-Critic on a GPU [[OpenReview](http://openreview.net/forum?id=r1VGvBcxl)]
- Learning to Compose Words into Sentences with Reinforcement Learning[[OpenReview](http://openreview.net/forum?id=Skvgqgqxe)]
- Deep Reinforcement Learning for Accelerating the Convergence Rate [[OpenReview](http://openreview.net/forum?id=Syg_lYixe)]
- [#Exploration: A Study of Count-Based Exploration for Deep Reinforcement Learning](notes/count-based-exploration.md) [[arXiv](https://arxiv.org/abs/1611.04717)]
- Learning to Compose Words into Sentences with Reinforcement Learning [[OpenReview](http://openreview.net/forum?id=Skvgqgqxe)]
- Learning to Navigate in Complex Environments [[arXiv](https://arxiv.org/abs/1611.03673)]
- Unsupervised Perceptual Rewards for Imitation Learning [[OpenReview](http://openreview.net/forum?id=Bkul3t9ee)]
- Q-Prop: Sample-Efficient Policy Gradient with An Off-Policy Critic [[OpenReview](http://openreview.net/forum?id=SJ3rcZcxl)]


Machine Translation & Dialog

- [Google's Multilingual Neural Machine Translation System: Enabling Zero-Shot Translation](notes/gnmt-multilingual.md) [[arXiv](https://arxiv.org/abs/1611.04558)]
- [Neural Machine Translation with Reconstruction](notes/nmt-with-reconstruction.md) [[arXiv](https://arxiv.org/abs/1611.01874v1)]
- Iterative Refinement for Machine Translation [[OpenReview](http://openreview.net/forum?id=r1y1aawlg)]
- A Convolutional Encoder Model for Neural Machine Translation [[arXiv](https://arxiv.org/abs/1611.02344)]
- Improving Neural Language Models with a Continuous Cache [[OpenReview](http://openreview.net/forum?id=B184E5qee)]
- Vocabulary Selection Strategies for Neural Machine Translation [[OpenReview](http://openreview.net/forum?id=Bk8N0RLxx)]
- Towards an automatic Turing test: Learning to evaluate dialogue responses [[OpenReview](http://openreview.net/forum?id=HJ5PIaseg)]
- Dialogue Learning With Human-in-the-Loop [[OpenReview](http://openreview.net/forum?id=HJgXCV9xx)]
- Batch Policy Gradient Methods for Improving Neural Conversation Models [[OpenReview](http://openreview.net/forum?id=rJfMusFll)]
- Learning through Dialogue Interactions [[OpenReview](http://openreview.net/forum?id=rkE8pVcle)]
- [Dual Learning for Machine Translation](notes/dual-learning-mt.md) [[arXiv](https://arxiv.org/abs/1611.00179)]
- Unsupervised Pretraining for Sequence to Sequence Learning [[arXiv](https://arxiv.org/abs/1611.02683)]



#### 2016-10

- [Neural Machine Translation in Linear Time](notes/nmt-linear-time.md) [[arXiv](https://arxiv.org/abs/1610.10099)]
- [Professor Forcing: A New Algorithm for Training Recurrent Networks](notes/professor-forcing.md) [[arXiv](https://arxiv.org/abs/1610.09038)]
- Learning to Protect Communications with Adversarial Neural Cryptography [[arXiv](https://arxiv.org/abs/1610.06918v1)]
- Can Active Memory Replace Attention? [[arXiv](https://arxiv.org/abs/1610.08613)]
- [Using Fast Weights to Attend to the Recent Past](notes/fast-weight-to-attend.md) [[arXiv](https://arxiv.org/abs/1610.06258)]
- [Fully Character-Level Neural Machine Translation without Explicit Segmentation](notes/conv-char-level-nmt.md) [[arXiv](https://arxiv.org/abs/1610.03017)]
- [Diverse Beam Search: Decoding Diverse Solutions from Neural Sequence Models](notes/diverse-beam-search.md) [[arXiv](https://arxiv.org/abs/1610.02424)]
- Video Pixel Networks [[arXiv](https://arxiv.org/abs/1610.00527)]
- Connecting Generative Adversarial Networks and Actor-Critic Methods [[arXiv](https://arxiv.org/abs/1610.01945)]
- [Learning to Translate in Real-time with Neural Machine Translation](notes/learning-to-translate-real-time.md) [[arXiv](https://arxiv.org/abs/1610.00388)]
- Collective Robot Reinforcement Learning with Distributed Asynchronous Guided Policy Search [arXiv](https://arxiv.org/abs/1610.00673)
- [Pointer Sentinel Mixture Models](notes/pointer-sentinel-mixture.md) [arXiv](https://arxiv.org/abs/1609.07843)

#### 2016-09

- Towards Deep Symbolic Reinforcement Learning [[arXiv](https://arxiv.org/abs/1609.05518)]
- HyperNetworks [[arXiv](https://arxiv.org/abs/1609.09106)]
- Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation [[arXiv](http://arxiv.org/abs/1609.08144)]
- Safe and Efficient Off-Policy Reinforcement Learning [[arXiv](http://arxiv.org/abs/1606.02647)]
- Playing FPS Games with Deep Reinforcement Learning [[arXiv](http://arxiv.org/abs/1609.05521)]
- [SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient](notes/seq-gan.md) [[arXiv](https://arxiv.org/abs/1609.05473)]
- Episodic Exploration for Deep Deterministic Policies: An Application to StarCraft Micromanagement Tasks [[arXiv](http://arxiv.org/abs/1609.02993)]
- Energy-based Generative Adversarial Network [[arXiv](https://arxiv.org/abs/1609.03126)]
- Stealing Machine Learning Models via Prediction APIs [[arXiv](http://arxiv.org/abs/1609.02943)]
- Semi-Supervised Classification with Graph Convolutional Networks [[arXiv](http://arxiv.org/abs/1609.02907)]
- WaveNet: A Generative Model For Raw Audio [[arXiv](https://arxiv.org/abs/1609.03499)]
- [Hierarchical Multiscale Recurrent Neural Networks](notes/hm-rnn.md) [[arXiv](https://arxiv.org/abs/1609.01704)]
- End-to-End Reinforcement Learning of Dialogue Agents for Information Access [[arXiv](https://arxiv.org/abs/1609.00777)]
- Deep Neural Networks for YouTube Recommendations [[paper](https://research.google.com/pubs/pub45530.html)]

#### 2016-08

- Machine Comprehension Using Match-LSTM and Answer Pointer [[arXiv](https://arxiv.org/abs/1608.07905)]
- Stacked Approximated Regression Machine: A Simple Deep Learning Approach [[arXiv](http://arxiv.org/abs/1608.04062)]
- Decoupled Neural Interfaces using Synthetic Gradients [[arXiv](http://arxiv.org/abs/1608.05343)]
- WikiReading: A Novel Large-scale Language Understanding Task over Wikipedia [[arXiv](https://arxiv.org/abs/1608.03542)]
- Temporal Attention Model for Neural Machine Translation [[arXiv](http://arxiv.org/abs/1608.02927)]
- Residual Networks of Residual Networks: Multilevel Residual Networks [[arXiv](http://arxiv.org/abs/1608.02908)]
- [Learning Online Alignments with Continuous Rewards Policy Gradient](notes/online-alignments-pg.md) [[arXiv](https://arxiv.org/abs/1608.01281)]

#### 2016-07

- [An Actor-Critic Algorithm for Sequence Prediction](notes/actor-critic-sequence.md) [[arXiv](http://arxiv.org/abs/1607.07086)]
- Cognitive Science in the era of Artificial Intelligence: A roadmap for reverse-engineering the infant language-learner [[arXiv](http://arxiv.org/abs/1607.08723v1)]
- [Recurrent Neural Machine Translation](notes/recurrent-nmt.md) [[arXiv](http://arxiv.org/abs/1607.08725)]
- MS-Celeb-1M: A Dataset and Benchmark for Large-Scale Face Recognition [[arXiv](http://arxiv.org/abs/1607.08221)]
- [Layer Normalization](notes/layer-norm.md) [[arXiv](https://arxiv.org/abs/1607.06450)]
- [Neural Machine Translation with Recurrent Attention Modeling](notes/nmt-rec-attention.md) [[arXiv](https://arxiv.org/abs/1607.05108)]
- Neural Semantic Encoders [[arXiv](https://arxiv.org/abs/1607.04315)]
- [Attention-over-Attention Neural Networks for Reading Comprehension](notes/att-over-att.md) [[arXiv](https://arxiv.org/abs/1607.04423)]
- sk_p: a neural program corrector for MOOCs [[arXiv](http://arxiv.org/abs/1607.02902)]
- Recurrent Highway Networks [[arXiv](https://arxiv.org/abs/1607.03474)]
- Bag of Tricks for Efficient Text Classification [[arXiv](http://arxiv.org/abs/1607.01759)]
- Context-Dependent Word Representation for Neural Machine Translation [[arXiv](https://arxiv.org/abs/1607.00578)]
- Dynamic Neural Turing Machine with Soft and Hard Addressing Schemes [[arXiv](http://arxiv.org/abs/1607.00036)]

#### 2016-06

- Sequence-to-Sequence Learning as Beam-Search Optimization [[arXiv](https://arxiv.org/abs/1606.02960)]
- [Sequence-Level Knowledge Distillation](notes/seq-knowledge-distillation.md) [[arXiv](https://arxiv.org/abs/1606.07947)]
- Policy Networks with Two-Stage Training for Dialogue Systems [[arXiv](http://arxiv.org/abs/1606.03152)]
- Towards an integration of deep learning and neuroscience [[arXiv](https://arxiv.org/abs/1606.03813)]
- On Multiplicative Integration with Recurrent Neural Networks [[arxiv](https://arxiv.org/abs/1606.06630)]
- [Wide & Deep Learning for Recommender Systems](wide-and-deep.md) [[arXiv](https://arxiv.org/abs/1606.07792)]
- Online and Offline Handwritten Chinese Character Recognition [[arXiv](https://arxiv.org/abs/1606.05763)]
- Tutorial on Variational Autoencoders [[arXiv](http://arxiv.org/abs/1606.05908)]
- Concrete Problems in AI Safety [[arXiv](https://arxiv.org/abs/1606.06565)]
- Deep Reinforcement Learning Discovers Internal Models [[arXiv](http://arxiv.org/abs/1606.05174v1)]
- [SQuAD: 100,000+ Questions for Machine Comprehension of Text](notes/squad.md) [[arXiv](http://arxiv.org/abs/1606.05250)]
- Conditional Image Generation with PixelCNN Decoders [[arXiv](http://arxiv.org/abs/1606.05328)]
- Model-Free Episodic Control [[arXiv](http://arxiv.org/abs/1606.04460)]
- [Progressive Neural Networks](notes/progressive-nn.md) [[arXiv](http://arxiv.org/abs/1606.04671)]
- Improved Techniques for Training GANs [[arXiv](http://arxiv.org/abs/1606.03498)])
- Memory-Efficient Backpropagation Through Time [[arXiv](http://arxiv.org/abs/1606.03401)]
- InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets [[arXiv](http://arxiv.org/abs/1606.03657)]
- Zero-Resource Translation with Multi-Lingual Neural Machine Translation [[arXiv](http://arxiv.org/abs/1606.04164)]
- Key-Value Memory Networks for Directly Reading Documents [[arXiv](http://arxiv.org/abs/1606.03126)]
- Deep Recurrent Models with Fast-Forward Connections for Neural Machine Translatin [[arXiv](http://arxiv.org/abs/1606.04199)]
- Learning to learn by gradient descent by gradient descent [[arXiv](http://arxiv.org/abs/1606.04474)]
- Learning Language Games through Interaction [[arXiv](http://arxiv.org/abs/1606.02447)]
- Zoneout: Regularizing RNNs by Randomly Preserving Hidden Activations [[arXiv](https://arxiv.org/abs/1606.01305)]
- Smart Reply: Automated Response Suggestion for Email [[arXiv](http://arxiv.org/abs/1606.04870)]
- Virtual Adversarial Training for Semi-Supervised Text Classification [[arXiv](https://arxiv.org/abs/1605.07725)]
- Deep Reinforcement Learning for Dialogue Generation [[arXiv](http://arxiv.org/abs/1606.01541)]
- Very Deep Convolutional Networks for Natural Language Processing [[arXiv](https://arxiv.org/abs/1606.01781)]
- Neural Net Models for Open-Domain Discourse Coherence [[arXiv](https://arxiv.org/abs/1606.01545)]
- Neural Architectures for Fine-grained Entity Type Classification [[arXiv](https://arxiv.org/abs/1606.01341)]
- Gated-Attention Readers for Text Comprehension [[arXiv](http://arxiv.org/abs/1606.01549)]
- [End-to-end LSTM-based dialog control optimized with supervised and reinforcement learning](notes/e2e-dialog-control-sl-rl.md) [[arXiv](https://arxiv.org/abs/1606.01269)]
- Iterative Alternating Neural Attention for Machine Reading [[arXiv](https://arxiv.org/abs/1606.02245)]
- Memory-enhanced Decoder for Neural Machine Translation [[arXiv](http://arxiv.org/abs/1606.02003)]
- Multiresolution Recurrent Neural Networks: An Application to Dialogue Response Generation [[arXiv](https://arxiv.org/abs/1606.00776)]
- [Natural Language Comprehension with the EpiReader](notes/epireader.md) [[arXiv](https://arxiv.org/abs/1606.02270)]
- Conversational Contextual Cues: The Case of Personalization and History for Response Ranking [[arXiv](https://arxiv.org/abs/1606.00372)]

- Adversarially Learned Inference [[arXiv](https://arxiv.org/abs/1606.00704)]
- Neural Network Translation Models for Grammatical Error Correction [[arXiv](https://arxiv.org/abs/1606.00189)]

#### 2016-05

- Hierarchical Memory Networks [[arXiv](https://arxiv.org/abs/1605.07427)]
- Deep API Learning [[arXiv](http://arxiv.org/abs/1605.08535)]
- Wide Residual Networks [[arXiv](http://arxiv.org/abs/1605.07146)]
- TensorFlow: A system for large-scale machine learning [[arXiv](http://arxiv.org/abs/1605.08695)]
- Learning Natural Language Inference using Bidirectional LSTM model and Inner-Attention [[arXiv](http://arxiv.org/abs/1605.09090)]
- Aspect Level Sentiment Classification with Deep Memory Network [[arXiv](http://arxiv.org/abs/1605.08900)]
- FractalNet: Ultra-Deep Neural Networks without Residuals [[arXiv](https://arxiv.org/abs/1605.07648)]
- Learning End-to-End Goal-Oriented Dialog [[arXiv](http://arxiv.org/abs/1605.07683)]
- One-shot Learning with Memory-Augmented Neural Networks [[arXiv](http://arxiv.org/abs/1605.06065)]
- Deep Learning without Poor Local Minima [[arXiv](http://arxiv.org/abs/1605.07110)]
- AVEC 2016 - Depression, Mood, and Emotion Recognition Workshop and Challenge [[arXiv](https://arxiv.org/abs/1605.01600)]
- Data Programming: Creating Large Training Sets, Quickly [[arXiv](http://arxiv.org/abs/1605.07723)]
- Deeply-Fused Nets [[arXiv](http://arxiv.org/abs/1605.07716)]
- Deep Portfolio Theory [[arXiv](http://arxiv.org/abs/1605.07230)]
- Unsupervised Learning for Physical Interaction through Video Prediction [[arXiv](http://arxiv.org/abs/1605.07157)]
- Movie Description [[arXiv](http://arxiv.org/abs/1605.03705)]


#### 2016-04

- Higher Order Recurrent Neural Networks [[arXiv](https://arxiv.org/abs/1605.00064)]
- Joint Line Segmentation and Transcription for End-to-End Handwritten Paragraph Recognition [[arXiv](https://arxiv.org/abs/1604.08352)]
- Hierarchical Deep Reinforcement Learning: Integrating Temporal Abstraction and Intrinsic Motivation [[arXiv](https://arxiv.org/abs/1604.06057)]
- The IBM 2016 English Conversational Telephone Speech Recognition System [[arXiv](https://arxiv.org/abs/1604.08242)]
- Dialog-based Language Learning [[arXiv](https://arxiv.org/abs/1604.06045)]
- Multilingual Part-of-Speech Tagging with Bidirectional Long Short-Term Memory Models and Auxiliary Loss [[arXiv](https://arxiv.org/abs/1604.05529)]
- Sentence-Level Grammatical Error Identification as Sequence-to-Sequence Correction [[arXiv](https://arxiv.org/abs/1604.04677)]
- A Network-based End-to-End Trainable Task-oriented Dialogue System [[arXiv](http://arxiv.org/abs/1604.04562)]
- Visual Storytelling [[arXiv](https://arxiv.org/abs/1604.03968)]
- Improving the Robustness of Deep Neural Networks via Stability Training [[arXiv](http://arxiv.org/abs/1604.04326)]
- [Bridging the Gaps Between Residual Learning, Recurrent Neural Networks and Visual Cortex](notes/bridging-gap-resnet-rnn.md) [[arXiv](https://arxiv.org/abs/1604.03640)]
- Scan, Attend and Read: End-to-End Handwritten Paragraph Recognition with MDLSTM Attention [[arXiv](https://arxiv.org/abs/1604.03286)]
- [Sentence Level Recurrent Topic Model: Letting Topics Speak for Themselves](notes/slrtm.md) [[arXiv](https://arxiv.org/abs/1604.02038)]
- [Achieving Open Vocabulary Neural Machine Translation with Hybrid Word-Character Models](notes/open-vocab-nmt-hybrid-word-character.md) [[arXiv](http://arxiv.org/abs/1604.00788)]
- [Building Machines That Learn and Think Like People](notes/building-machines-that-learn-and-think-like-people.md) [[arXiv](http://arxiv.org/abs/1604.00289)]
- A Semisupervised Approach for Language Identification based on Ladder Networks [[arXiv](http://arxiv.org/abs/1604.00317)]
- [Deep Networks with Stochastic Depth](notes/stochastic-depth.md) [[arXiv](http://arxiv.org/abs/1603.09382)]
- PHOCNet: A Deep Convolutional Neural Network for Word Spotting in Handwritten Documents [[arXiv](http://arxiv.org/abs/1604.00187)]


#### 2016-03

- Improving Information Extraction by Acquiring External Evidence with Reinforcement Learning [[arXiv](https://arxiv.org/abs/1603.07954)]
- A Fast Unified Model for Parsing and Sentence Understanding [[arXiv](http://arxiv.org/abs/1603.06021)]
- [Latent Predictor Networks for Code Generation](notes/latent-predictor-networks.md) [[arXiv](http://arxiv.org/abs/1603.06744)]
- Attend, Infer, Repeat: Fast Scene Understanding with Generative Models [[arXiv](http://arxiv.org/abs/1603.08575)]
- Recurrent Batch Normalization [[arXiv](http://arxiv.org/abs/1603.09025)]
- Neural Language Correction with Character-Based Attention [[arXiv](http://arxiv.org/abs/1603.09727)]
- [Incorporating Copying Mechanism in Sequence-to-Sequence Learning](notes/copynet.md) [[arXiv](http://arxiv.org/abs/1603.06393)]
- How NOT To Evaluate Your Dialogue System [[arXiv](http://arxiv.org/abs/1603.08023)]
- [Adaptive Computation Time for Recurrent Neural Networks](notes/act-rnn.md) [[arXiv](http://arxiv.org/abs/1603.08983)]
- A guide to convolution arithmetic for deep learning [[arXiv](http://arxiv.org/abs/1603.07285)]
- Colorful Image Colorization [[arXiv](http://arxiv.org/abs/1603.08983)]
- Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles [[arXiv](http://arxiv.org/abs/1603.09246)]
- Generating Factoid Questions With Recurrent Neural Networks: The 30M Factoid Question-Answer Corpus [[arXiv](http://arxiv.org/abs/1603.06807)]
- A Persona-Based Neural Conversation Model [[arXiv](http://arxiv.org/abs/1603.06155)]
- A Character-level Decoder without Explicit Segmentation for Neural Machine Translation [[arXiv](http://arxiv.org/abs/1603.06147)]
- Multi-Task Cross-Lingual Sequence Tagging from Scratch [[arXiv](http://arxiv.org/abs/1603.06270)]
- Neural Variational Inference for Text Processing [[arXiv](http://arxiv.org/abs/1511.06038)]
- Recurrent Dropout without Memory Loss [[arXiv](http://arxiv.org/abs/1603.05118)]
- One-Shot Generalization in Deep Generative Models [[arXiv](http://arxiv.org/abs/1603.05106)]
- Recursive Recurrent Nets with Attention Modeling for OCR in the Wild [[arXiv](Recursive Recurrent Nets with Attention Modeling for OCR in the Wild)]
- A New Method to Visualize Deep Neural Networks [[arXiv](A New Method to Visualize Deep Neural Networks)]
- Neural Architectures for Named Entity Recognition [[arXiv](http://arxiv.org/abs/1603.01360)]
- End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF [[arXiv](http://arxiv.org/abs/1603.01354)]
- Character-based Neural Machine Translation [[arXiv](http://arxiv.org/abs/1603.00810)]
- Learning Word Segmentation Representations to Improve Named Entity Recognition for Chinese Social Media [[arXiv](http://arxiv.org/abs/1603.00786)]

#### 2016-02

- Architectural Complexity Measures of Recurrent Neural Networks [[arXiv](http://arxiv.org/abs/1602.08210)]
- Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks [[arXiv](http://arxiv.org/abs/1602.07868)]
- Recurrent Neural Network Grammars [[arXiv](http://arxiv.org/abs/1602.07776)]
- Visual Genome: Connecting Language and Vision Using Crowdsourced Dense Image Annotations [[arXiv](http://arxiv.org/abs/1602.07332)]
- [Contextual LSTM (CLSTM) models for Large scale NLP tasks](notes/clstm-large-scale.md) [[arXiv](http://arxiv.org/abs/1602.06291)]
- Sequence-to-Sequence RNNs for Text Summarization [[arXiv](http://arxiv.org/abs/1602.06023)]
- Extraction of Salient Sentences from Labelled Documents [[arXiv](http://arxiv.org/abs/1412.6815)]
- Learning Distributed Representations of Sentences from Unlabelled Data [[arXiv](http://arxiv.org/abs/1602.03483)]
- Benefits of depth in neural networks [[arXiv](http://arxiv.org/abs/1602.04485)]
- [Associative Long Short-Term Memory](notes/associative-lstm.md) [[arXiv](http://arxiv.org/abs/1602.03032)]
- Generating images with recurrent adversarial networks [[arXiv](http://arxiv.org/abs/1602.05110)]
- [Exploring the Limits of Language Modeling](notes/exploring-the-limits-of-lm.md) [[arXiv](http://arxiv.org/abs/1602.02410)]
- Swivel: Improving Embeddings by Noticing What’s Missing [[arXiv](http://arxiv.org/abs/1602.02215)]
- [WebNav: A New Large-Scale Task for Natural Language based Sequential Decision Making](notes/webnav.md) [[arXiv](http://arxiv.org/abs/1602.02261)]
- [Efficient Character-level Document Classification by Combining Convolution and Recurrent Layers](notes/efficient-char-level-document-classification-cnn-rnn.md) [[arXiv](http://arxiv.org/abs/1602.00367)]
- BinaryNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1 [[arXiv](http://arxiv.org/abs/1602.02830)]
- Learning Discriminative Features via Label Consistent Neural Network [[arXiv](http://arxiv.org/abs/1602.01168)]

#### 2016-01

- Pixel Recurrent Neural Networks [[arXiv](http://arxiv.org/abs/1601.06759)]
- Bitwise Neural Networks [[arXiv](http://arxiv.org/abs/1601.06071)]
- Long Short-Term Memory-Networks for Machine Reading [[arXiv](http://arxiv.org/abs/1601.06733)]
- Coverage-based Neural Machine Translation [[arXiv](http://arxiv.org/abs/1601.04811)]
- Understanding Deep Convolutional Networks [[arXiv](http://arxiv.org/abs/1601.04920)]
- Training Recurrent Neural Networks by Diffusion [[arXiv](http://arxiv.org/abs/1601.04114)]
- Automatic Description Generation from Images: A Survey of Models, Datasets, and Evaluation Measures [[arXiv](http://arxiv.org/abs/1601.03896)]
- [Multi-Way, Multilingual Neural Machine Translation with a Shared Attention Mechanism](notes/multi-way-nmt-shared-attention.md) [[arXiv](http://arxiv.org/abs/1601.01073)]
- [Recurrent Memory Network for Language Modeling](notes/rmn-language-modeling.md) [[arXiv](http://arxiv.org/abs/1601.01272)]
- Language to Logical Form with Neural Attention [[arXiv](http://arxiv.org/abs/1601.01280)]
- Learning to Compose Neural Networks for Question Answering [[arXiv](http://arxiv.org/abs/1601.01705)]
- The Inevitability of Probability: Probabilistic Inference in Generic Neural Networks Trained with Non-Probabilistic Feedback [[arXiv](http://arxiv.org/abs/1601.03060)]
- COCO-Text: Dataset and Benchmark for Text Detection and Recognition in Natural Images [[arXiv](http://arxiv.org/abs/1601.07140)]
- Survey on the attention based RNN model and its applications in computer vision [[arXiv](http://arxiv.org/abs/1601.06823)]

#### 2015-12

NLP

- [Strategies for Training Large Vocabulary Neural Language Models](notes/strategies-for-training-large-vocab-lm.md) [[arXiv](http://arxiv.org/abs/1512.04906)]
- [Multilingual Language Processing From Bytes](notes/multilingual-language-processing-from-bytes.md) [[arXiv](http://arxiv.org/abs/1512.00103)]
- [Learning Document Embeddings by Predicting N-grams for Sentiment Classification of Long Movie Reviews](notes/learning-document-embeddings-ngrams.md) [[arXiv](http://arxiv.org/abs/1512.08183)]
- [Target-Dependent Sentiment Classification with Long Short Term Memory](notes/target-dependent-sentiment-lstm.md) [[arXiv](http://arxiv.org/abs/1512.01100)]
- Reading Text in the Wild with Convolutional Neural Networks [[arXiv](http://arxiv.org/abs/1412.1842)]

Vision

- [Deep Residual Learning for Image Recognition](notes/deep-residual-learning.md) [[arXiv](http://arxiv.org/abs/1512.03385)]
- Rethinking the Inception Architecture for Computer Vision [[arXiv](http://arxiv.org/abs/1512.00567)]
- Inside-Outside Net: Detecting Objects in Context with Skip Pooling and Recurrent Neural Networks [[arXiv](http://arxiv.org/abs/1512.04143)]
- Deep Speech 2: End-to-End Speech Recognition in English and Mandarin [[arXiv](http://arxiv.org/abs/1512.02595)]


#### 2015-11

NLP

- [Deep Reinforcement Learning with a Natural Language Action Space](notes/drl-nlp-action.md) [[arXiv](https://arxiv.org/abs/1511.04636)]
- Sequence Level Training with Recurrent Neural Networks [[arXiv](http://arxiv.org/abs/1511.06732)]
- [Teaching Machines to Read and Comprehend](notes/teaching-machines-to-read-and-comprehend.md) [[arxiv](http://arxiv.org/abs/1506.03340)]
- [Semi-supervised Sequence Learning](notes/semi-supervised-sequence-learning.md) [[arXiv](http://arxiv.org/abs/1511.01432)]
- [Multi-task Sequence to Sequence Learning](notes/multitask-seq2seq.md) [[arXiv](http://arxiv.org/abs/1511.06114)]
- [Alternative structures for character-level RNNs](notes/alternative-structure-char-rnn.md) [[arXiv](http://arxiv.org/abs/1511.06303)]
- [Larger-Context Language Modeling](notes/larger-context-lm.md) [[arXiv](http://arxiv.org/abs/1511.03729)]
- [A Unified Tagging Solution: Bidirectional LSTM Recurrent Neural Network with Word Embedding](notes/unified-tagging-blstm.md) [[arXiv](http://arxiv.org/abs/1511.00215)]
- Towards Universal Paraphrastic Sentence Embeddings [[arXiv](http://arxiv.org/abs/1511.08198)]
- BlackOut: Speeding up Recurrent Neural Network Language Models With Very Large Vocabularies [[arXiv](http://arxiv.org/abs/1511.06909)]
- Sequence Level Training with Recurrent Neural Networks [[arXiv](http://arxiv.org/abs/1511.06732)]
- Natural Language Understanding with Distributed Representation [[arXiv](http://arxiv.org/abs/1511.07916)]
- sense2vec - A Fast and Accurate Method for Word Sense Disambiguation In Neural Word Embeddings [[arXiv](http://arxiv.org/abs/1511.06388)]
- LSTM-based Deep Learning Models for non-factoid answer selection [[arXiv](http://arxiv.org/abs/1511.04108)]

Programs

- Neural Random-Access Machines [[arxiv](http://arxiv.org/abs/1511.06392)]
- Neural Programmer: Inducing Latent Programs with Gradient Descent [[arXiv](http://arxiv.org/abs/1511.04834)]
- Neural Programmer-Interpreters [[arXiv](http://arxiv.org/abs/1511.06279)]
- Learning Simple Algorithms from Examples [[arXiv](http://arxiv.org/abs/1511.07275)]
- Neural GPUs Learn Algorithms [[arXiv](http://arxiv.org/abs/1511.08228)]
- On Learning to Think: Algorithmic Information Theory for Novel Combinations of Reinforcement Learning Controllers and Recurrent Neural World Models [[arXiv](http://arxiv.org/abs/1511.09249)]

Vision

- ReSeg: A Recurrent Neural Network for Object Segmentation [[arXiv](http://arxiv.org/abs/1511.07053)]
- Deconstructing the Ladder Network Architecture [[arXiv](http://arxiv.org/abs/1511.06430)]
- Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks [[arXiv](http://arxiv.org/abs/1511.06434)]

General

- Towards Principled Unsupervised Learning [[arXiv](http://arxiv.org/abs/1511.06440)]
- Dynamic Capacity Networks [[arXiv](http://arxiv.org/abs/1511.07838)]
- [Generating Sentences from a `ous Space](notes/generating-sentences-cont-space.md) [[arXiv](http://arxiv.org/abs/1511.06349)]
- Net2Net: Accelerating Learning via Knowledge Transfer [[arXiv](http://arxiv.org/abs/1511.05641)]
- A Roadmap towards Machine Intelligence [[arXiv](http://arxiv.org/abs/1511.08130)]
- Session-based Recommendations with Recurrent Neural Networks [[arXiv](http://arxiv.org/abs/1511.06939)]
- Regularizing RNNs by Stabilizing Activations [[arXiv](http://arxiv.org/abs/1511.08400)]




#### 2015-10

- [A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification](notes/sensitivity-analysis-cnn-sentence-classification.md) [[arXiv](http://arxiv.org/abs/1510.03820)]
- [Attention with Intention for a Neural Network Conversation Model](notes/attention-with-intention.md) [[arXiv](http://arxiv.org/abs/1510.08565)]
- Part-of-Speech Tagging with Bidirectional Long Short-Term Memory Recurrent Neural Network [[arXiv](http://arxiv.org/abs/1510.06168)]
- A Survey: Time Travel in Deep Learning Space: An Introduction to Deep Learning Models and How Deep Learning Models Evolved from the Initial Ideas [[arXiv](http://arxiv.org/abs/1510.04781)]
- A Primer on Neural Network Models for Natural Language Processing [[arXiv](http://arxiv.org/abs/1510.00726)]
- [A Diversity-Promoting Objective Function for Neural Conversation Models](notes/diversity-promoting-objective-ncm.md) [[arXiv](http://arxiv.org/abs/1510.03055)]


#### 2015-09

- [Character-level Convolutional Networks for Text Classification](notes/character-level-cnn-for-text-classification.md) [[arXiv](http://arxiv.org/abs/1509.01626)]
- [A Neural Attention Model for Abstractive Sentence Summarization](notes/neural-attention-model-for-abstractive-sentence-summarization.md) [[arXiv](http://arxiv.org/abs/1509.00685)]
- Poker-CNN: A Pattern Learning Strategy for Making Draws and Bets in Poker Games [[arXiv](http://arxiv.org/abs/1509.06731)]

#### 2015-08

- Listen, Attend and Spell [[arxiv](http://arxiv.org/abs/1508.01211)]
- [Character-Aware Neural Language Models](notes/character-aware-nlm.md) [[arXiv](http://arxiv.org/abs/1508.06615)]
- Improved Transition-Based Parsing by Modeling Characters instead of Words with LSTMs [[arXiv](http://arxiv.org/abs/1508.00657)]
- Finding Function in Form: Compositional Character Models for Open Vocabulary Word Representation [[arXiv](http://arxiv.org/abs/1508.02096)]

#### 2015-07

- [Building End-To-End Dialogue Systems Using Generative Hierarchical Neural Network Models](e2e-dialog-ghnnm.md) [[arXiv](http://arxiv.org/abs/1507.04808)]
- Semi-Supervised Learning with Ladder Networks [[arXiv](http://arxiv.org/abs/1507.02672)]
- [Document Embedding with Paragraph Vectors](notes/document-embedding-with-pv.md) [[arXiv](http://arxiv.org/abs/1507.07998)]
- [Training Very Deep Networks](notes/training-very-deep-networks.md) [[arXiv](http://arxiv.org/abs/1507.06228)]

#### 2015-06

- Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning [[arXiv](https://arxiv.org/abs/1506.02142)]
- [A Neural Network Approach to Context-Sensitive Generation of Conversational Responses](notes/nn-context-sentitive-responses.md) [[arXiv](http://arxiv.org/abs/1506.06714)]
- [Document Embedding with Paragraph Vectors](notes/document-embedding-with-pv.md) [[arXiv](http://arxiv.org/abs/1507.07998)]
- [A Neural Conversational Model](notes/neural-conversational-model.md) [[arXiv](http://arxiv.org/abs/1506.05869)]
- [Skip-Thought Vectors](notes/skip-thought-vectors.md) [[arXiv](http://arxiv.org/abs/1506.06726)]
- [Pointer Networks](notes/pointer-networks.md) [[arXiv](http://arxiv.org/abs/1506.03134)]
- [Spatial Transformer Networks](notes/spatial-transformer-networks.md) [[arXiv](http://arxiv.org/abs/1506.02025)]
- Tree-structured composition in neural networks without tree-structured architectures [[arXiv](http://arxiv.org/abs/1506.04834)]
- Visualizing and Understanding Neural Models in NLP [[arXiv](http://arxiv.org/abs/1506.01066)]
- Learning to Transduce with Unbounded Memory [[arXiv](http://arxiv.org/abs/1506.02516)]
- Ask Me Anything: Dynamic Memory Networks for Natural Language Processing [[arXiv](http://arxiv.org/abs/1506.07285)]
- [Deep Knowledge Tracing](notes/deep-knowledge-tracing.md) [[arXiv](http://arxiv.org/abs/1506.05908)]

#### 2015-05

- [ReNet: A Recurrent Neural Network Based Alternative to Convolutional Networks](notes/renet-rnn-alternative-to-convnet.md) [[arXiv](http://arxiv.org/abs/1505.00393)]
- Reinforcement Learning Neural Turing Machines [[arXiv](http://arxiv.org/abs/1505.00521)]

#### 2015-04

- Correlational Neural Networks [[arXiv](http://arxiv.org/abs/1504.07225)]

#### 2015-03


- [Distilling the Knowledge in a Neural Network](notes/distilling-the-knowledge-in-a-nn.md) [[arXiv](http://arxiv.org/abs/1503.02531)]
- [End-To-End Memory Networks](notes/end-to-end-memory-networks.md) [[arXiv](http://arxiv.org/abs/1503.08895)]
- [Neural Responding Machine for Short-Text Conversation](notes/neural-responding-machine.md) [[arXiv](http://arxiv.org/abs/1503.02364)]
- [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](notes/batch-normalization.md) [[arXiv](http://arxiv.org/abs/1502.03167)]


#### 2015-02

- [Text Understanding from Scratch](notes/text-understanding-from-scratch.md) [[arXiv](http://arxiv.org/abs/1502.01710)]
- [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](notes/show-attend-tell.md) [[arXiv](http://arxiv.org/abs/1502.03044)]

#### 2015-01

#### 2014-12

- Learning Longer Memory in Recurrent Neural Networks [[arXiv](http://arxiv.org/abs/1412.7753)]
- [Neural Turing Machines](notes/neural-turing-machines.md) [[arxiv](http://arxiv.org/abs/1410.5401)]
- [Grammar as a Foreign Langauage](notes/grammar-as-a-foreign-language.md) [[arXiv](http://arxiv.org/abs/1412.7449)]
- [On Using Very Large Target Vocabulary for Neural Machine Translation](notes/on-using-very-large-target-vocabulary-for-nmt.md) [[arXiv](http://arxiv.org/abs/1412.2007)]
- Effective Use of Word Order for Text Categorization with Convolutional Neural Networks [[arXiv](http://arxiv.org/abs/1412.1058v1)]
- Multiple Object Recognition with Visual Attention [[arXiv](http://arxiv.org/abs/1412.7755)]

#### 2014-11

#### 2014-10

- [Learning to Execute](notes/learning-to-execute.md) [[arXiv](http://arxiv.org/abs/1410.4615)]

#### 2014-09

- [Sequence to Sequence Learning with Neural Networks](notes/seq2seq-with-neural-networks.md) [[arXiv](http://arxiv.org/abs/1409.3215)]
- [Neural Machine Translation by Jointly Learning to Align and Translate](notes/nmt-jointly-learning-to-align-and-translate.md) [[arxiv](http://arxiv.org/abs/1409.0473)]
- [On the Properties of Neural Machine Translation: Encoder-Decoder Approaches](notes/properties-of-neural-mt.md) [[arXiv](http://arxiv.org/abs/1409.1259)]
- [Recurrent Neural Network Regularization](notes/rnn-regularization.md) [[arXiv](http://arxiv.org/abs/1409.2329)]
- Very Deep Convolutional Networks for Large-Scale Image Recognition [[arXiv](http://arxiv.org/abs/1409.1556)]
- Going Deeper with Convolutions [[arXiv](http://arxiv.org/abs/1409.4842)]

#### 2014-08

- Convolutional Neural Networks for Sentence Classification [[arxiv](http://arxiv.org/abs/1408.5882)]

#### 2014-07

#### 2014-06

- [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](notes/learning-phrase-representations.md) [[arXiv](http://arxiv.org/abs/1406.1078)]
- [Recurrent Models of Visual Attention](notes/recurrent-models-of-visual-attention.md) [[arXiv](http://arxiv.org/abs/1406.6247)]
- Generative Adversarial Networks [[arXiv](http://arxiv.org/abs/1406.2661)]

#### 2014-05

- [Distributed Representations of Sentences and Documents](notes/distributed-representations-of-sentences-and-documents.md) [[arXiv](http://arxiv.org/abs/1405.4053)]

#### 2014-04

- A Convolutional Neural Network for Modelling Sentences [[arXiv](http://arxiv.org/abs/1404.2188)]

#### 2014-03

#### 2014-02

#### 2014-01

#### 2013

- Visualizing and Understanding Convolutional Networks [[arXiv](http://arxiv.org/abs/1311.2901)]
- DeViSE: A Deep Visual-Semantic Embedding Model [[pub](http://research.google.com/pubs/pub41473.html)]
- Maxout Networks [[arXiv](http://arxiv.org/abs/1302.4389)]
- Exploiting Similarities among Languages for Machine Translation [[arXiv](http://arxiv.org/abs/1309.4168)]
- Efficient Estimation of Word Representations in Vector Space [[arXiv](http://arxiv.org/abs/1301.3781)]


#### 2011

- Natural Language Processing (almost) from Scratch [[arXiv](http://arxiv.org/abs/1103.0398)]

====================================================================================

# Deep Learning Papers Reading Roadmap

>If you are a newcomer to the Deep Learning area, the first question you may have is "Which paper should I start reading from?"

>Here is a reading roadmap of Deep Learning papers!

The roadmap is constructed in accordance with the following four guidelines:

- From outline to detail
- From old to state-of-the-art
- from generic to specific areas
- focus on state-of-the-art

You will find many papers that are quite new but really worth reading.

I would continue adding papers to this roadmap.


---------------------------------------

# 1 Deep Learning History and Basics

## 1.0 Book

**[0]** Bengio, Yoshua, Ian J. Goodfellow, and Aaron Courville. "**Deep learning**." An MIT Press book. (2015). [[pdf]](https://github.com/HFTrader/DeepLearningBook/raw/master/DeepLearningBook.pdf) **(Deep Learning Bible, you can read this book while reading following papers.)** :star::star::star::star::star:

## 1.1 Survey

**[1]** LeCun, Yann, Yoshua Bengio, and Geoffrey Hinton. "**Deep learning**." Nature 521.7553 (2015): 436-444. [[pdf]](http://www.cs.toronto.edu/~hinton/absps/NatureDeepReview.pdf) **(Three Giants' Survey)** :star::star::star::star::star:

## 1.2 Deep Belief Network(DBN)(Milestone of Deep Learning Eve)

**[2]** Hinton, Geoffrey E., Simon Osindero, and Yee-Whye Teh. "**A fast learning algorithm for deep belief nets**." Neural computation 18.7 (2006): 1527-1554. [[pdf]](http://www.cs.toronto.edu/~hinton/absps/ncfast.pdf)**(Deep Learning Eve)** :star::star::star:

**[3]** Hinton, Geoffrey E., and Ruslan R. Salakhutdinov. "**Reducing the dimensionality of data with neural networks**." Science 313.5786 (2006): 504-507. [[pdf]](http://www.cs.toronto.edu/~hinton/science.pdf) **(Milestone, Show the promise of deep learning)** :star::star::star:

## 1.3 ImageNet Evolution（Deep Learning broke out from here）

**[4]** Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "**Imagenet classification with deep convolutional neural networks**." Advances in neural information processing systems. 2012. [[pdf]](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) **(AlexNet, Deep Learning Breakthrough)** :star::star::star::star::star:

**[5]** Simonyan, Karen, and Andrew Zisserman. "**Very deep convolutional networks for large-scale image recognition**." arXiv preprint arXiv:1409.1556 (2014). [[pdf]](https://arxiv.org/pdf/1409.1556.pdf) **(VGGNet,Neural Networks become very deep!)** :star::star::star:

**[6]** Szegedy, Christian, et al. "**Going deeper with convolutions**." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2015. [[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Szegedy_Going_Deeper_With_2015_CVPR_paper.pdf) **(GoogLeNet)** :star::star::star:

**[7]** He, Kaiming, et al. "**Deep residual learning for image recognition**." arXiv preprint arXiv:1512.03385 (2015). [[pdf]](https://arxiv.org/pdf/1512.03385.pdf) **(ResNet,Very very deep networks, CVPR best paper)** :star::star::star::star::star:

## 1.4 Speech Recognition Evolution

**[8]** Hinton, Geoffrey, et al. "**Deep neural networks for acoustic modeling in speech recognition: The shared views of four research groups**." IEEE Signal Processing Magazine 29.6 (2012): 82-97. [[pdf]](http://cs224d.stanford.edu/papers/maas_paper.pdf) **(Breakthrough in speech recognition)**:star::star::star::star:

**[9]** Graves, Alex, Abdel-rahman Mohamed, and Geoffrey Hinton. "**Speech recognition with deep recurrent neural networks**." 2013 IEEE international conference on acoustics, speech and signal processing. IEEE, 2013. [[pdf]](http://arxiv.org/pdf/1303.5778.pdf) **(RNN)**:star::star::star:

**[10]** Graves, Alex, and Navdeep Jaitly. "**Towards End-To-End Speech Recognition with Recurrent Neural Networks**." ICML. Vol. 14. 2014. [[pdf]](http://www.jmlr.org/proceedings/papers/v32/graves14.pdf):star::star::star:

**[11]** Sak, Haşim, et al. "**Fast and accurate recurrent neural network acoustic models for speech recognition**." arXiv preprint arXiv:1507.06947 (2015). [[pdf]](http://arxiv.org/pdf/1507.06947) **(Google Speech Recognition System)** :star::star::star:

**[12]** Amodei, Dario, et al. "**Deep speech 2: End-to-end speech recognition in english and mandarin**." arXiv preprint arXiv:1512.02595 (2015). [[pdf]](https://arxiv.org/pdf/1512.02595.pdf) **(Baidu Speech Recognition System)** :star::star::star::star:

**[13]** W. Xiong, J. Droppo, X. Huang, F. Seide, M. Seltzer, A. Stolcke, D. Yu, G. Zweig "**Achieving Human Parity in Conversational Speech Recognition**." arXiv preprint arXiv:1610.05256 (2016). [[pdf]](https://arxiv.org/pdf/1610.05256v1) **(State-of-the-art in speech recognition, Microsoft)** :star::star::star::star:

>After reading above papers, you will have a basic understanding of the Deep Learning history, the basic architectures of Deep Learning model(including CNN, RNN, LSTM) and how deep learning can be applied to image and speech recognition issues. The following papers will take you in-depth understanding of the Deep Learning method, Deep Learning in different areas of application and the frontiers. I suggest that you can choose the following papers based on your interests and research direction.

#2 Deep Learning Method

## 2.1 Model

**[14]** Hinton, Geoffrey E., et al. "**Improving neural networks by preventing co-adaptation of feature detectors**." arXiv preprint arXiv:1207.0580 (2012). [[pdf]](https://arxiv.org/pdf/1207.0580.pdf) **(Dropout)** :star::star::star:

**[15]** Srivastava, Nitish, et al. "**Dropout: a simple way to prevent neural networks from overfitting**." Journal of Machine Learning Research 15.1 (2014): 1929-1958. [[pdf]](http://www.jmlr.org/papers/volume15/srivastava14a.old/source/srivastava14a.pdf) :star::star::star:

**[16]** Ioffe, Sergey, and Christian Szegedy. "**Batch normalization: Accelerating deep network training by reducing internal covariate shift**." arXiv preprint arXiv:1502.03167 (2015). [[pdf]](http://arxiv.org/pdf/1502.03167) **(An outstanding Work in 2015)** :star::star::star::star:

**[17]** Ba, Jimmy Lei, Jamie Ryan Kiros, and Geoffrey E. Hinton. "**Layer normalization**." arXiv preprint arXiv:1607.06450 (2016). [[pdf]](https://arxiv.org/pdf/1607.06450.pdf?utm_source=sciontist.com&utm_medium=refer&utm_campaign=promote) **(Update of Batch Normalization)** :star::star::star::star:

**[18]** Courbariaux, Matthieu, et al. "**Binarized Neural Networks: Training Neural Networks with Weights and Activations Constrained to+ 1 or−1**." [[pdf]](https://pdfs.semanticscholar.org/f832/b16cb367802609d91d400085eb87d630212a.pdf) **(New Model,Fast)**  :star::star::star:

**[19]** Jaderberg, Max, et al. "**Decoupled neural interfaces using synthetic gradients**." arXiv preprint arXiv:1608.05343 (2016). [[pdf]](https://arxiv.org/pdf/1608.05343) **(Innovation of Training Method,Amazing Work)** :star::star::star::star::star:

**[20]** Chen, Tianqi, Ian Goodfellow, and Jonathon Shlens. "Net2net: Accelerating learning via knowledge transfer." arXiv preprint arXiv:1511.05641 (2015). [[pdf]](https://arxiv.org/abs/1511.05641) **(Modify previously trained network to reduce training epochs)** :star::star::star:

**[21]** Wei, Tao, et al. "Network Morphism." arXiv preprint arXiv:1603.01670 (2016). [[pdf]](https://arxiv.org/abs/1603.01670) **(Modify previously trained network to reduce training epochs)** :star::star::star:

## 2.2 Optimization

**[22]** Sutskever, Ilya, et al. "**On the importance of initialization and momentum in deep learning**." ICML (3) 28 (2013): 1139-1147. [[pdf]](http://www.jmlr.org/proceedings/papers/v28/sutskever13.pdf) **(Momentum optimizer)** :star::star:

**[23]** Kingma, Diederik, and Jimmy Ba. "**Adam: A method for stochastic optimization**." arXiv preprint arXiv:1412.6980 (2014). [[pdf]](http://arxiv.org/pdf/1412.6980) **(Maybe used most often currently)** :star::star::star:

**[24]** Andrychowicz, Marcin, et al. "**Learning to learn by gradient descent by gradient descent**." arXiv preprint arXiv:1606.04474 (2016). [[pdf]](https://arxiv.org/pdf/1606.04474) **(Neural Optimizer,Amazing Work)** :star::star::star::star::star:

**[25]** Han, Song, Huizi Mao, and William J. Dally. "**Deep compression: Compressing deep neural network with pruning, trained quantization and huffman coding**." CoRR, abs/1510.00149 2 (2015). [[pdf]](https://pdfs.semanticscholar.org/5b6c/9dda1d88095fa4aac1507348e498a1f2e863.pdf) **(ICLR best paper, new direction to make NN running fast,DeePhi Tech Startup)** :star::star::star::star::star:

**[26]** Iandola, Forrest N., et al. "**SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and< 1MB model size**." arXiv preprint arXiv:1602.07360 (2016). [[pdf]](http://arxiv.org/pdf/1602.07360) **(Also a new direction to optimize NN,DeePhi Tech Startup)** :star::star::star::star:

## 2.3 Unsupervised Learning / Deep Generative Model

**[27]** Le, Quoc V. "**Building high-level features using large scale unsupervised learning**." 2013 IEEE international conference on acoustics, speech and signal processing. IEEE, 2013. [[pdf]](http://arxiv.org/pdf/1112.6209.pdf&embed) **(Milestone, Andrew Ng, Google Brain Project, Cat)** :star::star::star::star:


**[28]** Kingma, Diederik P., and Max Welling. "**Auto-encoding variational bayes**." arXiv preprint arXiv:1312.6114 (2013). [[pdf]](http://arxiv.org/pdf/1312.6114) **(VAE)** :star::star::star::star:

**[29]** Goodfellow, Ian, et al. "**Generative adversarial nets**." Advances in Neural Information Processing Systems. 2014. [[pdf]](http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf) **(GAN,super cool idea)** :star::star::star::star::star:

**[30]** Radford, Alec, Luke Metz, and Soumith Chintala. "**Unsupervised representation learning with deep convolutional generative adversarial networks**." arXiv preprint arXiv:1511.06434 (2015). [[pdf]](http://arxiv.org/pdf/1511.06434) **(DCGAN)** :star::star::star::star:

**[31]** Gregor, Karol, et al. "**DRAW: A recurrent neural network for image generation**." arXiv preprint arXiv:1502.04623 (2015). [[pdf]](http://jmlr.org/proceedings/papers/v37/gregor15.pdf) **(VAE with attention, outstanding work)** :star::star::star::star::star:

**[32]** Oord, Aaron van den, Nal Kalchbrenner, and Koray Kavukcuoglu. "**Pixel recurrent neural networks**." arXiv preprint arXiv:1601.06759 (2016). [[pdf]](http://arxiv.org/pdf/1601.06759) **(PixelRNN)** :star::star::star::star:

**[33]** Oord, Aaron van den, et al. "Conditional image generation with PixelCNN decoders." arXiv preprint arXiv:1606.05328 (2016). [[pdf]](https://arxiv.org/pdf/1606.05328) **(PixelCNN)** :star::star::star::star:

## 2.4 RNN / Sequence-to-Sequence Model

**[34]** Graves, Alex. "**Generating sequences with recurrent neural networks**." arXiv preprint arXiv:1308.0850 (2013). [[pdf]](http://arxiv.org/pdf/1308.0850) **(LSTM, very nice generating result, show the power of RNN)** :star::star::star::star:

**[35]** Cho, Kyunghyun, et al. "**Learning phrase representations using RNN encoder-decoder for statistical machine translation**." arXiv preprint arXiv:1406.1078 (2014). [[pdf]](http://arxiv.org/pdf/1406.1078) **(First Seq-to-Seq Paper)** :star::star::star::star:

**[36]** Sutskever, Ilya, Oriol Vinyals, and Quoc V. Le. "**Sequence to sequence learning with neural networks**." Advances in neural information processing systems. 2014. [[pdf]](http://papers.nips.cc/paper/5346-information-based-learning-by-agents-in-unbounded-state-spaces.pdf) **(Outstanding Work)** :star::star::star::star::star:

**[37]** Bahdanau, Dzmitry, KyungHyun Cho, and Yoshua Bengio. "**Neural Machine Translation by Jointly Learning to Align and Translate**." arXiv preprint arXiv:1409.0473 (2014). [[pdf]](https://arxiv.org/pdf/1409.0473v7.pdf) :star::star::star::star:

**[38]** Vinyals, Oriol, and Quoc Le. "**A neural conversational model**." arXiv preprint arXiv:1506.05869 (2015). [[pdf]](http://arxiv.org/pdf/1506.05869.pdf%20(http://arxiv.org/pdf/1506.05869.pdf)) **(Seq-to-Seq on Chatbot)** :star::star::star:

## 2.5 Neural Turing Machine

**[39]** Graves, Alex, Greg Wayne, and Ivo Danihelka. "**Neural turing machines**." arXiv preprint arXiv:1410.5401 (2014). [[pdf]](http://arxiv.org/pdf/1410.5401.pdf) **(Basic Prototype of Future Computer)** :star::star::star::star::star:

**[40]** Zaremba, Wojciech, and Ilya Sutskever. "**Reinforcement learning neural Turing machines**." arXiv preprint arXiv:1505.00521 362 (2015). [[pdf]](https://pdfs.semanticscholar.org/f10e/071292d593fef939e6ef4a59baf0bb3a6c2b.pdf) :star::star::star:

**[41]** Weston, Jason, Sumit Chopra, and Antoine Bordes. "**Memory networks**." arXiv preprint arXiv:1410.3916 (2014). [[pdf]](http://arxiv.org/pdf/1410.3916) :star::star::star:


**[42]** Sukhbaatar, Sainbayar, Jason Weston, and Rob Fergus. "**End-to-end memory networks**." Advances in neural information processing systems. 2015. [[pdf]](http://papers.nips.cc/paper/5846-end-to-end-memory-networks.pdf) :star::star::star::star:

**[43]** Vinyals, Oriol, Meire Fortunato, and Navdeep Jaitly. "**Pointer networks**." Advances in Neural Information Processing Systems. 2015. [[pdf]](http://papers.nips.cc/paper/5866-pointer-networks.pdf) :star::star::star::star:

**[44]** Graves, Alex, et al. "**Hybrid computing using a neural network with dynamic external memory**." Nature (2016). [[pdf]](https://www.dropbox.com/s/0a40xi702grx3dq/2016-graves.pdf) **(Milestone,combine above papers' ideas)** :star::star::star::star::star:

## 2.6 Deep Reinforcement Learning

**[45]** Mnih, Volodymyr, et al. "**Playing atari with deep reinforcement learning**." arXiv preprint arXiv:1312.5602 (2013). [[pdf]](http://arxiv.org/pdf/1312.5602.pdf)) **(First Paper named deep reinforcement learning)** :star::star::star::star:

**[46]** Mnih, Volodymyr, et al. "**Human-level control through deep reinforcement learning**." Nature 518.7540 (2015): 529-533. [[pdf]](http://www.davidqiu.com:8888/research/nature14236.pdf) **(Milestone)** :star::star::star::star::star:

**[47]** Wang, Ziyu, Nando de Freitas, and Marc Lanctot. "**Dueling network architectures for deep reinforcement learning**." arXiv preprint arXiv:1511.06581 (2015). [[pdf]](http://arxiv.org/pdf/1511.06581) **(ICLR best paper,great idea)**  :star::star::star::star:

**[48]** Mnih, Volodymyr, et al. "**Asynchronous methods for deep reinforcement learning**." arXiv preprint arXiv:1602.01783 (2016). [[pdf]](http://arxiv.org/pdf/1602.01783) **(State-of-the-art method)** :star::star::star::star::star:

**[49]** Lillicrap, Timothy P., et al. "**Continuous control with deep reinforcement learning**." arXiv preprint arXiv:1509.02971 (2015). [[pdf]](http://arxiv.org/pdf/1509.02971) **(DDPG)** :star::star::star::star:

**[50]** Gu, Shixiang, et al. "**Continuous Deep Q-Learning with Model-based Acceleration**." arXiv preprint arXiv:1603.00748 (2016). [[pdf]](http://arxiv.org/pdf/1603.00748) **(NAF)** :star::star::star::star:

**[51]** Schulman, John, et al. "**Trust region policy optimization**." CoRR, abs/1502.05477 (2015). [[pdf]](http://www.jmlr.org/proceedings/papers/v37/schulman15.pdf) **(TRPO)** :star::star::star::star:

**[52]** Silver, David, et al. "**Mastering the game of Go with deep neural networks and tree search**." Nature 529.7587 (2016): 484-489. [[pdf]](http://willamette.edu/~levenick/cs448/goNature.pdf) **(AlphaGo)** :star::star::star::star::star:

## 2.7 Deep Transfer Learning / Lifelong Learning / especially for RL

**[53]** Bengio, Yoshua. "**Deep Learning of Representations for Unsupervised and Transfer Learning**." ICML Unsupervised and Transfer Learning 27 (2012): 17-36. [[pdf]](http://www.jmlr.org/proceedings/papers/v27/bengio12a/bengio12a.pdf) **(A Tutorial)** :star::star::star:

**[54]** Silver, Daniel L., Qiang Yang, and Lianghao Li. "**Lifelong Machine Learning Systems: Beyond Learning Algorithms**." AAAI Spring Symposium: Lifelong Machine Learning. 2013. [[pdf]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.696.7800&rep=rep1&type=pdf) **(A brief discussion about lifelong learning)**  :star::star::star:

**[55]** Hinton, Geoffrey, Oriol Vinyals, and Jeff Dean. "**Distilling the knowledge in a neural network**." arXiv preprint arXiv:1503.02531 (2015). [[pdf]](http://arxiv.org/pdf/1503.02531) **(Godfather's Work)** :star::star::star::star:

**[56]** Rusu, Andrei A., et al. "**Policy distillation**." arXiv preprint arXiv:1511.06295 (2015). [[pdf]](http://arxiv.org/pdf/1511.06295) **(RL domain)** :star::star::star:

**[57]** Parisotto, Emilio, Jimmy Lei Ba, and Ruslan Salakhutdinov. "**Actor-mimic: Deep multitask and transfer reinforcement learning**." arXiv preprint arXiv:1511.06342 (2015). [[pdf]](http://arxiv.org/pdf/1511.06342) **(RL domain)** :star::star::star:

**[58]** Rusu, Andrei A., et al. "**Progressive neural networks**." arXiv preprint arXiv:1606.04671 (2016). [[pdf]](https://arxiv.org/pdf/1606.04671) **(Outstanding Work, A novel idea)** :star::star::star::star::star:


## 2.8 One Shot Deep Learning

**[59]** Lake, Brenden M., Ruslan Salakhutdinov, and Joshua B. Tenenbaum. "**Human-level concept learning through probabilistic program induction**." Science 350.6266 (2015): 1332-1338. [[pdf]](http://clm.utexas.edu/compjclub/wp-content/uploads/2016/02/lake2015.pdf) **(No Deep Learning,but worth reading)** :star::star::star::star::star:

**[60]** Koch, Gregory, Richard Zemel, and Ruslan Salakhutdinov. "**Siamese Neural Networks for One-shot Image Recognition**."(2015) [[pdf]](http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf) :star::star::star:

**[61]** Santoro, Adam, et al. "**One-shot Learning with Memory-Augmented Neural Networks**." arXiv preprint arXiv:1605.06065 (2016). [[pdf]](http://arxiv.org/pdf/1605.06065) **(A basic step to one shot learning)** :star::star::star::star:

**[62]** Vinyals, Oriol, et al. "**Matching Networks for One Shot Learning**." arXiv preprint arXiv:1606.04080 (2016). [[pdf]](https://arxiv.org/pdf/1606.04080) :star::star::star:

**[63]** Hariharan, Bharath, and Ross Girshick. "**Low-shot visual object recognition**." arXiv preprint arXiv:1606.02819 (2016). [[pdf]](http://arxiv.org/pdf/1606.02819) **(A step to large data)** :star::star::star::star:


# 3 Applications

## 3.1 NLP(Natural Language Processing)

**[1]** Antoine Bordes, et al. "**Joint Learning of Words and Meaning Representations for Open-Text Semantic Parsing**." AISTATS(2012) [[pdf]](https://www.hds.utc.fr/~bordesan/dokuwiki/lib/exe/fetch.php?id=en%3Apubli&cache=cache&media=en:bordes12aistats.pdf) :star::star::star::star:

**[2]** Mikolov, et al. "**Distributed representations of words and phrases and their compositionality**." ANIPS(2013): 3111-3119 [[pdf]](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) **(word2vec)** :star::star::star:

**[3]** Sutskever, et al. "**“Sequence to sequence learning with neural networks**." ANIPS(2014) [[pdf]](http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf) :star::star::star:

**[4]** Ankit Kumar, et al. "**“Ask Me Anything: Dynamic Memory Networks for Natural Language Processing**." arXiv preprint arXiv:1506.07285(2015) [[pdf]](https://arxiv.org/abs/1506.07285) :star::star::star::star:

**[5]** Yoon Kim, et al. "**Character-Aware Neural Language Models**." NIPS(2015) arXiv preprint arXiv:1508.06615(2015) [[pdf]](https://arxiv.org/abs/1508.06615) :star::star::star::star:

**[6]** Jason Weston, et al. "**Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks**." arXiv preprint arXiv:1502.05698(2015) [[pdf]](https://arxiv.org/abs/1502.05698) **(bAbI tasks)** :star::star::star:

**[7]** Karl Moritz Hermann, et al. "**Teaching Machines to Read and Comprehend**." arXiv preprint arXiv:1506.03340(2015) [[pdf]](https://arxiv.org/abs/1506.03340) **(CNN/DailyMail cloze style questions)** :star::star:

**[8]** Alexis Conneau, et al. "**Very Deep Convolutional Networks for Natural Language Processing**." arXiv preprint arXiv:1606.01781(2016) [[pdf]](https://arxiv.org/abs/1606.01781) **(state-of-the-art in text classification)** :star::star::star:

**[9]** Armand Joulin, et al. "**Bag of Tricks for Efficient Text Classification**." arXiv preprint arXiv:1607.01759(2016) [[pdf]](https://arxiv.org/abs/1607.01759) **(slightly worse than state-of-the-art, but a lot faster)** :star::star::star:

## 3.2 Object Detection

**[1]** Szegedy, Christian, Alexander Toshev, and Dumitru Erhan. "**Deep neural networks for object detection**." Advances in Neural Information Processing Systems. 2013. [[pdf]](http://papers.nips.cc/paper/5207-deep-neural-networks-for-object-detection.pdf) :star::star::star:

**[2]** Girshick, Ross, et al. "**Rich feature hierarchies for accurate object detection and semantic segmentation**." Proceedings of the IEEE conference on computer vision and pattern recognition. 2014. [[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Girshick_Rich_Feature_Hierarchies_2014_CVPR_paper.pdf) **(RCNN)** :star::star::star::star::star:

**[3]** He, Kaiming, et al. "**Spatial pyramid pooling in deep convolutional networks for visual recognition**." European Conference on Computer Vision. Springer International Publishing, 2014. [[pdf]](http://arxiv.org/pdf/1406.4729) **(SPPNet)** :star::star::star::star:

**[4]** Girshick, Ross. "**Fast r-cnn**." Proceedings of the IEEE International Conference on Computer Vision. 2015. [[pdf]](https://pdfs.semanticscholar.org/8f67/64a59f0d17081f2a2a9d06f4ed1cdea1a0ad.pdf) :star::star::star::star:

**[5]** Ren, Shaoqing, et al. "**Faster R-CNN: Towards real-time object detection with region proposal networks**." Advances in neural information processing systems. 2015. [[pdf]](http://papers.nips.cc/paper/5638-analysis-of-variational-bayesian-latent-dirichlet-allocation-weaker-sparsity-than-map.pdf) :star::star::star::star:

**[6]** Redmon, Joseph, et al. "**You only look once: Unified, real-time object detection**." arXiv preprint arXiv:1506.02640 (2015). [[pdf]](http://homes.cs.washington.edu/~ali/papers/YOLO.pdf) **(YOLO,Oustanding Work, really practical)** :star::star::star::star::star:

**[7]** Liu, Wei, et al. "**SSD: Single Shot MultiBox Detector**." arXiv preprint arXiv:1512.02325 (2015). [[pdf]](http://arxiv.org/pdf/1512.02325) :star::star::star:

## 3.3 Visual Tracking

**[1]** Wang, Naiyan, and Dit-Yan Yeung. "**Learning a deep compact image representation for visual tracking**." Advances in neural information processing systems. 2013. [[pdf]](http://papers.nips.cc/paper/5192-learning-a-deep-compact-image-representation-for-visual-tracking.pdf) **(First Paper to do visual tracking using Deep Learning,DLT Tracker)** :star::star::star:

**[2]** Wang, Naiyan, et al. "**Transferring rich feature hierarchies for robust visual tracking**." arXiv preprint arXiv:1501.04587 (2015). [[pdf]](http://arxiv.org/pdf/1501.04587) **(SO-DLT)** :star::star::star::star:

**[3]** Wang, Lijun, et al. "**Visual tracking with fully convolutional networks**." Proceedings of the IEEE International Conference on Computer Vision. 2015. [[pdf]](http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Wang_Visual_Tracking_With_ICCV_2015_paper.pdf) **(FCNT)** :star::star::star::star:

**[4]** Held, David, Sebastian Thrun, and Silvio Savarese. "**Learning to Track at 100 FPS with Deep Regression Networks**." arXiv preprint arXiv:1604.01802 (2016). [[pdf]](http://arxiv.org/pdf/1604.01802) **(GOTURN,Really fast as a deep learning method,but still far behind un-deep-learning methods)** :star::star::star::star:

**[5]** Bertinetto, Luca, et al. "**Fully-Convolutional Siamese Networks for Object Tracking**." arXiv preprint arXiv:1606.09549 (2016). [[pdf]](https://arxiv.org/pdf/1606.09549) **(SiameseFC,New state-of-the-art for real-time object tracking)** :star::star::star::star:

**[6]** Martin Danelljan, Andreas Robinson, Fahad Khan, Michael Felsberg. "**Beyond Correlation Filters: Learning Continuous Convolution Operators for Visual Tracking**." ECCV (2016) [[pdf]](http://www.cvl.isy.liu.se/research/objrec/visualtracking/conttrack/C-COT_ECCV16.pdf) **(C-COT)** :star::star::star::star:

**[7]** Nam, Hyeonseob, Mooyeol Baek, and Bohyung Han. "**Modeling and Propagating CNNs in a Tree Structure for Visual Tracking**." arXiv preprint arXiv:1608.07242 (2016). [[pdf]](https://arxiv.org/pdf/1608.07242) **(VOT2016 Winner,TCNN)** :star::star::star::star:

## 3.4 Image Caption
**[1]** Farhadi,Ali,etal. "**Every picture tells a story: Generating sentences from images**". In Computer VisionECCV 2010. Springer Berlin Heidelberg:15-29, 2010. [[pdf]](https://www.cs.cmu.edu/~afarhadi/papers/sentence.pdf) :star::star::star:

**[2]** Kulkarni, Girish, et al. "**Baby talk: Understanding and generating image descriptions**". In Proceedings of the 24th CVPR, 2011. [[pdf]](http://tamaraberg.com/papers/generation_cvpr11.pdf):star::star::star::star:

**[3]** Vinyals, Oriol, et al. "**Show and tell: A neural image caption generator**". In arXiv preprint arXiv:1411.4555, 2014. [[pdf]](https://arxiv.org/pdf/1411.4555.pdf):star::star::star:

**[4]** Donahue, Jeff, et al. "**Long-term recurrent convolutional networks for visual recognition and description**". In arXiv preprint arXiv:1411.4389 ,2014. [[pdf]](https://arxiv.org/pdf/1411.4389.pdf)

**[5]** Karpathy, Andrej, and Li Fei-Fei. "**Deep visual-semantic alignments for generating image descriptions**". In arXiv preprint arXiv:1412.2306, 2014. [[pdf]](https://cs.stanford.edu/people/karpathy/cvpr2015.pdf):star::star::star::star::star:

**[6]** Karpathy, Andrej, Armand Joulin, and Fei Fei F. Li. "**Deep fragment embeddings for bidirectional image sentence mapping**". In Advances in neural information processing systems, 2014. [[pdf]](https://arxiv.org/pdf/1406.5679v1.pdf):star::star::star::star:

**[7]** Fang, Hao, et al. "**From captions to visual concepts and back**". In arXiv preprint arXiv:1411.4952, 2014. [[pdf]](https://arxiv.org/pdf/1411.4952v3.pdf):star::star::star::star::star:

**[8]** Chen, Xinlei, and C. Lawrence Zitnick. "**Learning a recurrent visual representation for image caption generation**". In arXiv preprint arXiv:1411.5654, 2014. [[pdf]](https://arxiv.org/pdf/1411.5654v1.pdf):star::star::star::star:

**[9]** Mao, Junhua, et al. "**Deep captioning with multimodal recurrent neural networks (m-rnn)**". In arXiv preprint arXiv:1412.6632, 2014. [[pdf]](https://arxiv.org/pdf/1412.6632v5.pdf):star::star::star:

**[10]** Xu, Kelvin, et al. "**Show, attend and tell: Neural image caption generation with visual attention**". In arXiv preprint arXiv:1502.03044, 2015. [[pdf]](https://arxiv.org/pdf/1502.03044v3.pdf):star::star::star::star::star:

## 3.5 Machine Translation

> Some milestone papers are listed in RNN / Seq-to-Seq topic.

**[1]** Luong, Minh-Thang, et al. "**Addressing the rare word problem in neural machine translation**." arXiv preprint arXiv:1410.8206 (2014). [[pdf]](http://arxiv.org/pdf/1410.8206) :star::star::star::star:


**[2]** Sennrich, et al. "**Neural Machine Translation of Rare Words with Subword Units**". In arXiv preprint arXiv:1508.07909, 2015. [[pdf]](https://arxiv.org/pdf/1508.07909.pdf):star::star::star:

**[3]** Luong, Minh-Thang, Hieu Pham, and Christopher D. Manning. "**Effective approaches to attention-based neural machine translation**." arXiv preprint arXiv:1508.04025 (2015). [[pdf]](http://arxiv.org/pdf/1508.04025) :star::star::star::star:

**[4]** Chung, et al. "**A Character-Level Decoder without Explicit Segmentation for Neural Machine Translation**". In arXiv preprint arXiv:1603.06147, 2016. [[pdf]](https://arxiv.org/pdf/1603.06147.pdf):star::star:

**[5]** Lee, et al. "**Fully Character-Level Neural Machine Translation without Explicit Segmentation**". In arXiv preprint arXiv:1610.03017, 2016. [[pdf]](https://arxiv.org/pdf/1610.03017.pdf):star::star::star::star::star:

**[6]** Wu, Schuster, Chen, Le, et al. "**Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation**". In arXiv preprint arXiv:1609.08144v2, 2016. [[pdf]](https://arxiv.org/pdf/1609.08144v2.pdf) **(Milestone)** :star::star::star::star:

## 3.6 Robotics

**[1]** Koutník, Jan, et al. "**Evolving large-scale neural networks for vision-based reinforcement learning**." Proceedings of the 15th annual conference on Genetic and evolutionary computation. ACM, 2013. [[pdf]](http://repository.supsi.ch/4550/1/koutnik2013gecco.pdf) :star::star::star:

**[2]** Levine, Sergey, et al. "**End-to-end training of deep visuomotor policies**." Journal of Machine Learning Research 17.39 (2016): 1-40. [[pdf]](http://www.jmlr.org/papers/volume17/15-522/15-522.pdf) :star::star::star::star::star:

**[3]** Pinto, Lerrel, and Abhinav Gupta. "**Supersizing self-supervision: Learning to grasp from 50k tries and 700 robot hours**." arXiv preprint arXiv:1509.06825 (2015). [[pdf]](http://arxiv.org/pdf/1509.06825) :star::star::star:

**[4]** Levine, Sergey, et al. "**Learning Hand-Eye Coordination for Robotic Grasping with Deep Learning and Large-Scale Data Collection**." arXiv preprint arXiv:1603.02199 (2016). [[pdf]](http://arxiv.org/pdf/1603.02199) :star::star::star::star:

**[5]** Zhu, Yuke, et al. "**Target-driven Visual Navigation in Indoor Scenes using Deep Reinforcement Learning**." arXiv preprint arXiv:1609.05143 (2016). [[pdf]](https://arxiv.org/pdf/1609.05143) :star::star::star::star:

**[6]** Yahya, Ali, et al. "**Collective Robot Reinforcement Learning with Distributed Asynchronous Guided Policy Search**." arXiv preprint arXiv:1610.00673 (2016). [[pdf]](https://arxiv.org/pdf/1610.00673) :star::star::star::star:

**[7]** Gu, Shixiang, et al. "**Deep Reinforcement Learning for Robotic Manipulation**." arXiv preprint arXiv:1610.00633 (2016). [[pdf]](https://arxiv.org/pdf/1610.00633) :star::star::star::star:

**[8]** A Rusu, M Vecerik, Thomas Rothörl, N Heess, R Pascanu, R Hadsell."**Sim-to-Real Robot Learning from Pixels with Progressive Nets**." arXiv preprint arXiv:1610.04286 (2016). [[pdf]](https://arxiv.org/pdf/1610.04286.pdf) :star::star::star::star:

## 3.7 Art

**[1]** Mordvintsev, Alexander; Olah, Christopher; Tyka, Mike (2015). "**Inceptionism: Going Deeper into Neural Networks**". Google Research. [[html]](https://research.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html) **(Deep Dream)**
:star::star::star::star:

**[2]** Gatys, Leon A., Alexander S. Ecker, and Matthias Bethge. "**A neural algorithm of artistic style**." arXiv preprint arXiv:1508.06576 (2015). [[pdf]](http://arxiv.org/pdf/1508.06576) **(Outstanding Work, most successful method currently)** :star::star::star::star::star:

**[3]** Zhu, Jun-Yan, et al. "**Generative Visual Manipulation on the Natural Image Manifold**." European Conference on Computer Vision. Springer International Publishing, 2016. [[pdf]](https://arxiv.org/pdf/1609.03552) **(iGAN)** :star::star::star::star:

**[4]** Champandard, Alex J. "**Semantic Style Transfer and Turning Two-Bit Doodles into Fine Artworks**." arXiv preprint arXiv:1603.01768 (2016). [[pdf]](http://arxiv.org/pdf/1603.01768) **(Neural Doodle)** :star::star::star::star:

**[5]** Zhang, Richard, Phillip Isola, and Alexei A. Efros. "**Colorful Image Colorization**." arXiv preprint arXiv:1603.08511 (2016). [[pdf]](http://arxiv.org/pdf/1603.08511) :star::star::star::star:

**[6]** Johnson, Justin, Alexandre Alahi, and Li Fei-Fei. "**Perceptual losses for real-time style transfer and super-resolution**." arXiv preprint arXiv:1603.08155 (2016). [[pdf]](https://arxiv.org/pdf/1603.08155.pdf) :star::star::star::star:

**[7]** Vincent Dumoulin, Jonathon Shlens and Manjunath Kudlur. "**A learned representation for artistic style**." arXiv preprint arXiv:1610.07629 (2016). [[pdf]](https://arxiv.org/pdf/1610.07629v1.pdf) :star::star::star::star:

## 3.8 Audio

## 3.9 Game

## 3.10 Knowledge Graph

## 3.11 Recommender Systems

## 3.12 Bioinformatics / Computational Biology

## 3.13 Neural Network Chip

## 3.14 Other Frontiers

