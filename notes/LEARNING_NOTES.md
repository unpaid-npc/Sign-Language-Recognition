# Key Engineering Lessons Learned

## 1. Python Version & C++ Binary Wheel Compatibility
- **Lesson:** Bleeding-edge Python versions (like Python 3.14 / 3.15) often lack pre-compiled C++ wheels for computer vision libraries, and newer MediaPipe releases (`>=0.10.30`) deprecated `mp.solutions.holistic`.
- **Solution:** Pinning Python `3.12` alongside `mediapipe==0.10.21` and `numpy<2.0.0` guarantees stable C-API interoperability between OpenCV, MediaPipe, and PyTorch.

## 2. The Selfie Mirror Flip Trap in Hand Tracking
- **Lesson:** Flipping a webcam image horizontally (`cv2.flip(frame, 1)`) *before* passing it to MediaPipe causes MediaPipe's handedness classifier to label the user's physical right hand as their left hand. Since WLASL training videos are unmirrored, this places features into the wrong indices (`[18:81]` instead of `[81:144]`).
- **Solution:** Always run MediaPipe on the raw, unmirrored camera frame first, draw skeletal landmarks, and flip the display image *afterward* right before drawing text overlays.

## 3. Camera Framing & Coordinate Normalization
- **Lesson:** Body-centric normalization relies on the Left and Right Shoulders (`landmarks 11 & 12`) to determine scale. When a user sits very close to a laptop webcam so only their face is visible, shoulder distance estimation degrades.
- **Solution:** Adding a real-time shoulder visibility check (`check_shoulders_in_frame`) on the HUD guides the user to step back slightly so their upper torso matches the framing of WLASL signers.

## 4. Low-Sample Regime in 100-Class Classification
- **Lesson:** With 709 training clips across 100 classes (~7 clips/class), a recurrent neural network will rapidly memorize training coordinates without domain-specific regularization.
- **Solution:** Combining 3D geometric data augmentation, first-order velocity features ($\Delta \mathbf{x}_t$), label smoothing (`0.05`), and dual temporal pooling pushed unseen Test Top-1 accuracy to `39.06%` and Test Top-5 accuracy to `67.19%`.