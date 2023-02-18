# CLSP

Adaptation and attempt at retraining TorToiSe's CLVP.

## Roadmap

- [x] Transformer architecture re-implementation.
- [x] CLSP implementation.
- [x] Simple distributed training loop.
- [x] Tiktoken text encoder.
- [x] Add Whisper encoder for speech.
- [x] Text-speech alignment using Whisper attention heads.
- [ ] Confirm proper contrastive training.

### Data things

- [x] Split transcribed audio into aligned token-chunk audio pairs.
- [ ] Process to encode this and translate tokens.

---

`torchaudio` is currently holding this project back from using Python 3.11.
