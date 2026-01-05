**Multilingual Video Captioning Using Deep Learning**
**Overview**

This project presents an end-to-end multilingual video captioning system built using deep learning techniques. The goal is to automatically generate meaningful textual descriptions for video content by integrating audio, visual, and language information, with a focus on accessibility and human–machine interaction.

The work was developed as an undergraduate capstone project and explores multimodal learning by combining computer vision, speech recognition, and natural language processing within a unified AI pipeline.

**System Pipeline**

The system follows a multimodal architecture consisting of the following stages:

1. **Video Processing**

- Extraction of visual frames from video input
- Feature extraction using Convolutional Neural Networks (CNNs)

**Audio Processing**

- Audio stream extraction from video

- Speech-to-text transcription using a pretrained ASR model

**Temporal & Language Modeling**

- Temporal modeling using recurrent architectures (LSTM)

- Multilingual text representation using pretrained language embeddings

**Caption Generation**

- Mapping multimodal features to natural language captions

- Support for multilingual caption outputs

**Technologies Used**

- Programming Language: Python

- Deep Learning: TensorFlow / Keras

- Computer Vision: OpenCV

- Speech Recognition: OpenAI Whisper (pretrained)

- Natural Language Processing: Multilingual BERT (mBERT)

- Evaluation Metrics: BLEU, METEOR, ROUGE

**Evaluation**

The system was evaluated using standard captioning metrics such as BLEU, METEOR, and ROUGE. Due to the multilingual and generative nature of the task, lexical overlap metrics exhibited variability. Metrics such as METEOR and ROUGE were more informative in reflecting semantic alignment between generated captions and reference descriptions.

In addition to quantitative evaluation, qualitative analysis confirmed that the system was capable of producing coherent and contextually relevant captions across different video samples.

**Project Status**

This repository contains:

- Core model architecture

- Training and evaluation logic

- Experimental pipeline used during development

The code is shared for academic and educational purposes and reflects an experimental research-oriented implementation rather than a production-ready system.

**Author**

**Eliud Mbouala Akiana**
Bachelor’s in Computer Science
**Interest Areas:** Artificial Intelligence, Machine Learning, Computer Vision, NLP

**Notes**

- Datasets are not included due to size and licensing constraints.

- The notebook is provided in view-only form to demonstrate methodology and system design.

- Results are discussed qualitatively to reflect the limitations of standard metrics for multilingual captioning tasks.

**License**

This project is shared for educational and academic review purposes only.
