# A Preventative Techniques to Disrupt the Deep Fake Audio Synthesis

This repository contains the Python implementation of the project. 

## Abstract

This project titled “A Preventative Techniques to Disrupt DeepFake Audio Synthesis” is a step toward addressing the growing concern of DeepFake audio, where synthesized speech is used for malicious purposes such as fraud
and spreading misinformation. The project proposes a defense mechanism that employs adversarial examples to disrupt the synthesis process, making it difficult for attackers to
synthesize convincing DeepFake audio. Using an ensemble
learning approach will ensure the transferability of adversarial examples to unknown synthesis models. The system
is able to test against various state-of-the-art synthesizers
and speaker verification systems.


## Required Libraries

The code is implemented and tested with **Python 3.9** on Ubuntu 20.04. 

We recommend running the code in a conda environment with at least 32GB of RAM and a GPU with at least 16 GB of VRAM. 

Install all dependencies by running the following:
```
conda create --name pytorchaudio python=3.7
conda activate pytorchaudio
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0
pip install -r ./TTS/requirements.txt
pip install -r requirements.txt
sudo apt install ffmpeg
```

# Inference

To convert the audio file to the compatible format, please use the following:
```
ffmpeg -i <source audio path> -acodec pcm_s16le -ac 1 -ar 16000 -ab 256k output_audio.wav
```

The code takes in two arguments of a source audio file to be protected (.wav), and a protected output audio file (e.g. "./output/completed.wav")
```
python run.py <source wav path> <output wav path> 
```

Example: 
```
python run.py "./samples/libri_2/source/source.wav" "./samples/libri_2/protected/protected.wav"
```

Some sample results of the source and the protected wav files are located in ./samples/[speaker]/source, and ./samples/[speaker]/protected. The synthesized results from the voice cloning engine are also included:

*_rtvc.wav: synthesized using Real Time Voice Cloning (also known as SV2TTS)

*_avc.wav: synthesized using Adaptive Voice Conversion 

*_coqui.wav: synthesized using COQUI TTS

*_tortoise.wav: synthesized using Tortoise TTS

