# lipread-aware-asr
This is a lipreading project in Junior year.

## motivation and purpose
With the continuous advancement of deep learning, artificial intelligence has achieved remarkable breakthroughs in various fields, including image processing and language understanding. In recent years, these technologies have gradually been introduced into cross-model research, among which Visual Speech Recognition (VSR) has attracted widespread attention. The core objective of VSR is to reconstruct speech content from silent videos by leveraging visual signals, particularly the motion features of the lips and face. The value of this technology lies in its ability to compensate for the limitations of traditional speech recognition in noisy environments or speech-missing scenarios, while also demonstrating practical potential in applications such as assistive technologies for the hearing impaired, surveillance systems, and intelligent interactive systems. However, lip-reading itself remains a highly challenging task: similar lip shapes may correspond to different speech units, and variations in oral structure and facial expressions across individuals further complicate the mapping between visual movements and speech content. Therefore, this project aims to enhance the model’s understanding of lip dynamics through more refined feature perception and optimization strategies, thereby improving recognition accuracy and cross-speaker generalization, and ultimately promoting the practicality and reliability of VSR in real-world scenarios.

## Implementation
We first pretrain the model on the [LRS2 dataset](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html) using the implementation available at [deep_avsr](https://github.com/smeetrs/deep_avsr?tab=readme-ov-file#). 

The pretrained model is then fine-tuned on our self-recorded Chinese corpus in 20 people with 10mins for each person. Recorded following the dataset collection methodology proposed in [VoiceBank-2023: A Multi-Speaker Mandarin Speech Corpus for Constructing Personalized TTS Systems for the Speech Impaired](https://arxiv.org/abs/2308.14763) . To ensure compatibility with the original framework and preserve the consistency of the training setting, the Chinese dataset is recorded and organized according to the specifications of the original dataset. The detailed recording conditions and dataset format are presented below.
* Videofile: .mp4 file
* Video: 25 fps, 160x160 RGB frames, Mouth approx. in center (face size should be comparable to frame size)
* Audio: Mono audio, 16000 Hz sample rate

## Record step
The recording program was written in C and executed in Visual Studio Code Insiders. GStreamer was used to construct the audio-visual recording pipeline. When the program is launched, the computer’s camera and microphone are activated. After the speaker finishes recording the assigned text, the system sequentially generates raw audio-visual files named 00001.mp4, 00002.mp4, 00003.mp4, and so on. Each file contains both the recorded video and its corresponding audio.

After all texts have been recorded, the MP4 files are batch-processed using a separate preprocessing program written in Python. First, FFmpeg is used to standardize the frame rate of the original videos to 25 fps and resample the audio to 16 kHz. The program then reads each video frame by frame and passes every frame to the YOLOv8 Face model to obtain the bounding-box coordinates of the speaker’s face. ByteTrack is also used to continuously track the same face throughout the video.

To prevent the cropped video from shaking because of speaker movement or minor detection errors, the facial coordinates of consecutive frames are smoothed. The cropping region is then moderately expanded around the detected face to preserve the complete facial and mouth regions. Each frame is subsequently cropped and resized to 160 × 160 pixels.

All processed frames are first combined into a temporary video without audio. FFmpeg is then used to merge the cropped video with the 16 kHz audio extracted from the original video, producing the final processed MP4 file. For example, after the original 00001.mp4 has been processed, another file with the same name, 00001.mp4, is generated. However, its video content has been converted into 160 × 160 cropped facial images while retaining the synchronized audio.

Finally, all processed videos are organized according to speaker ID and paired with their corresponding Chinese transcripts and video identifiers. This process produces a standardized Chinese audio-visual speech dataset with speaker labels and text annotations.

## contributor
- Huang Yu Min