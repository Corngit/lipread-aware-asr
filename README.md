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
How To Use

1. Install the required software and dependencies, including GStreamer, FFmpeg, Python, and the Python packages required by the FaceCrop program.

2. Open the FaceRecord directory in Visual Studio Code Insiders. Make sure that the camera and microphone are working properly, then compile and run the C recording program.

3. After the program starts, the speaker records the assigned text. Each completed recording is saved as a raw audio-visual file named sequentially as 00001.mp4, 00002.mp4, 00003.mp4, and so on.

4. The recorded MP4 files are automatically saved to the input directory used by FaceCrop, so no manual file transfer is required. Before preprocessing, confirm that the videos have been saved successfully.

5. Open the FaceCrop directory in Visual Studio Code. Confirm that test_face_crop.py, yolov8n-face-lindevs.pt, and bytetrack.yaml are present, and verify that the input and output paths in the program are configured correctly. Then, run test_face_crop.py.

6. The program automatically batch-processes the videos in the input directory. It converts the videos to 25 fps, resamples the audio to 16 kHz, uses YOLOv8 Face and ByteTrack to detect and track the speaker’s face, and crops and resizes each frame to 160 × 160 pixels.

7. After preprocessing, the program merges the cropped video with the original audio and saves the processed MP4 files in the specified output directory. The output files retain their original filenames and can then be organized by speaker ID and corresponding transcript for model training and testing.

Why Use Visual Studio Insiders?

This project uses Visual Studio Insiders because the C compiler, related extensions, and the GStreamer header and library paths were originally configured in the Insiders environment. This does not mean that the program can only run in Insiders. When using the stable version of Visual Studio Code, the same C/C++ extension, compiler, and GStreamer Development package must be installed. The correct include paths, library paths, and environment variables must also be added to the .vscode configuration before the program can be compiled and executed properly.

## contributor
- Huang Yu Min