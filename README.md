# lipread-aware-asr
This is a lipreading project in Junior year.

## motivation and purpose
With the continuous advancement of deep learning, artificial intelligence has achieved remarkable breakthroughs in various fields, including image processing and language understanding. In recent years, these technologies have gradually been introduced into cross-model research, among which Visual Speech Recognition (VSR) has attracted widespread attention. The core objective of VSR is to reconstruct speech content from silent videos by leveraging visual signals, particularly the motion features of the lips and face. The value of this technology lies in its ability to compensate for the limitations of traditional speech recognition in noisy environments or speech-missing scenarios, while also demonstrating practical potential in applications such as assistive technologies for the hearing impaired, surveillance systems, and intelligent interactive systems. However, lip-reading itself remains a highly challenging task: similar lip shapes may correspond to different speech units, and variations in oral structure and facial expressions across individuals further complicate the mapping between visual movements and speech content. Therefore, this project aims to enhance the model’s understanding of lip dynamics through more refined feature perception and optimization strategies, thereby improving recognition accuracy and cross-speaker generalization, and ultimately promoting the practicality and reliability of VSR in real-world scenarios.

## Implementation
We first pretrain the model on the [LRS2 dataset](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html) using the implementation available at [deep_avsr](https://github.com/smeetrs/deep_avsr?tab=readme-ov-file#). 

The pretrained model is then fine-tuned on our self-recorded Chinese corpus in 20 people with 10mins for each person. Recorded following the dataset collection methodology proposed in [VoiceBank-2023: A Multi-Speaker Mandarin Speech Corpus for Constructing Personalized TTS Systems for the Speech Impaired](https://arxiv.org/abs/2308.14763) . To ensure compatibility with the original framework and preserve the consistency of the training setting, the Chinese dataset is recorded and organized according to the specifications of the original dataset. The detailed recording conditions and dataset format are presented below.
* Videofile - .mp4 file
* Video: 25 fps, 160x160 RGB frames, Mouth approx. in center (face size should be comparable to frame size)
* Audio: Mono audio, 16000 Hz sample rate

## contributor
- Huang Yu Min