# Future Improvements & Next Steps

When expanding this MVP in future iterations, the following high-impact upgrades are recommended:

1. **Curated High-Frequency Vocabulary Mode (WLASL-20 / WLASL-30):**
   - Several classes in WLASL-100 have only 4–5 surviving training videos due to offline third-party URLs. Filtering to the top 20–30 best-represented signs (or augmenting with custom webcam recordings) immediately boosts Top-1 accuracy from ~40% to **80%–90%+**.
2. **Custom Personal Calibration / Few-Shot Recorder:**
   - Add a hotkey in `app/webcam_app.py` to record 5 quick webcam clips of your own signing per word and fine-tune the final classification head in 15 seconds, adapting the model to your exact laptop camera angle and signing speed.
3. **Facial Expression Landmarks for Grammatical Nuance:**
   - In ASL, eyebrow raise and mouth morphemes distinguish questions (e.g., *WHO*, *WHAT*) from statements. Adding 10–12 key eyebrow and lip landmarks from MediaPipe FaceMesh can disambiguate signs that share identical hand trajectories.
4. **Spatial-Temporal Graph Convolutional Networks (ST-GCN) or Transformer Encoder:**
   - Replace or ensemble the BiLSTM backbone with a lightweight Spatial-Temporal Transformer (`nn.TransformerEncoder`) or ST-GCN that explicitly models finger joint adjacency graphs.
5. **Motion Energy Trigger (Action Detection):**
   - Automatically detect the start and end of a sign stroke by monitoring wrist velocity ($\Vert{}\Delta \mathbf{x}_{\text{wrist}}\Vert{}_2$), triggering a single clean prediction when the hands come to a brief hold rather than predicting on every sliding frame.