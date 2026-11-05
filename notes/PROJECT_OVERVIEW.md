# Project Overview & Interview Pitch

## 1. What is Sign Language Recognition (SLR)?
Sign Language Recognition is a computer vision and sequence modeling task that translates visual sign language gestures into text. It has two main sub-fields:
- **Isolated SLR (ISLR):** Classifying individual word-level signs ("glosses") from video segments.
- **Continuous SLR (CSLR):** Recognizing full, unsegmented sentences with grammar and co-articulation.

This project builds a real-time **Isolated American Sign Language Recognition** system trained on the **WLASL-100** dataset (100 common ASL words).

## 2. Why Use Skeletal Landmarks Instead of Raw Video Frames?
 processing raw RGB video frames ($640 \times 480 \times 3 = 921,600$ numbers per frame) with 3D CNNs (like I3D) has three major drawbacks:
1. **Compute Bottleneck:** Requires expensive GPUs for training and struggles to achieve 30 FPS on a laptop CPU.
2. **Overfitting to Background & Appearance:** With only ~7–10 training videos per word in WLASL-100, pixel-based models easily memorize the signer's shirt color, room background, or lighting instead of hand motion.
3. **Dimensionality Reduction:** MediaPipe Holistic extracts the exact 3D geometric coordinates of the hands and upper body ($144$ numbers per frame)—a **6,400x compression** that is invariant to background and skin tone.

## 3. Elevator Pitch for Technical Interviews
> *"I built an end-to-end real-time American Sign Language recognition system on the 100-class WLASL-100 dataset. To make the system run at real-time FPS on a standard CPU and prevent overfitting on small per-class video counts, I used MediaPipe Holistic to extract 48 3D upper-body and hand landmarks per frame, normalized them relative to the signer's shoulder center and width, and engineered first-order velocity features. I trained a 2-layer Bidirectional LSTM with Temporal Attention in PyTorch, achieving 41.5% validation Top-1 (72.7% Top-5) and 39.1% test Top-1 (67.2% Top-5) across 100 classes—nearly 40x above the 1% random baseline. Finally, I wrapped the model in a stateful OpenCV inference engine featuring a 30-frame sliding buffer, hand-presence gating, and moving-average probability smoothing."*