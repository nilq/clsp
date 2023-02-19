# CLSP

Attempt at training a proper speech-CLIP.

## Roadmap

### Model things 🥞

Models resemble those of OpenCLIP and TorToiSe's CLVP.

- [x] Transformer architecture re-implementation.
- [x] CLSP implementation.
- [x] Simple distributed training loop.
- [x] Tiktoken text encoder.
- [x] Add Whisper encoder for speech.
- [x] Text-speech alignment using Whisper attention heads.

### Data things 📠

Here, main functionality consists of aligned token-speech alignment.

- [x] Split transcribed audio into aligned token-chunk audio pairs.
- [x] Process to encode this and translate tokens.


## Training 🤹

- [ ] Need data.
- [ ] Confirm proper contrastive training.

---
