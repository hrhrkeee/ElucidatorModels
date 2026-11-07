# Elucidator Models

Versioned optional on-device knowledge image encoders for Elucidator. This repository contains only model distribution material and reproducible conversion tools.

## Knowledge image encoder v1

Source: [Google SigLIP 2 Base](https://huggingface.co/google/siglip2-base-patch16-224), revision `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2`, Apache-2.0. No fine tuning. The image tower is converted to Core ML FP16; the text tower is excluded.

Input `pixels`: float32 NCHW `[1,3,224,224]`, RGB, upright full image resized to 224x224, values `pixel / 127.5 - 1`. Output `embedding`: L2-normalized 768-dimensional float32 vector. Core ML target iOS 18+. Preprocessing is part of the versioned contract; do not mix indexes from different revisions or preprocessing versions.

Release `knowledge-image-v1` distributes three package files separately. Reconstruct `KnowledgeImageEncoder.mlpackage` using the paths in `KnowledgeModel.json`, verify each size and SHA-256, then compile with Core ML. The manifest fixes the version; applications must not follow `latest` or load a remote model-selection policy. The source `.mlpackage` and compiled result are regenerable download data.

## Reproduce conversion and parity verification

```sh
uv run --python 3.12 --with torch==2.7.1 --with transformers==4.57.1 --with pillow --with numpy==2.2.6 --with torchvision==0.22.1 --with coremltools==9.0 python convert.py --output KnowledgeImageEncoder.mlpackage
uv run --python 3.12 --with torch==2.7.1 --with transformers==4.57.1 --with pillow --with numpy==2.2.6 --with torchvision==0.22.1 --with coremltools==9.0 python verify.py --model KnowledgeImageEncoder.mlpackage --images image1.jpg image2.jpg --output conversion-parity.json
```

Verification requires macOS/Core ML. The included report compares six development product photographs; it verifies conversion, not independent retrieval quality, unknown-object rejection, or iPhone performance. The calibration report is exploratory and reuses existing development features. The selected intermediate standard trades known-correct acceptance against unknown false acceptance. On previously observed development check partitions it yields 67.18% known-correct single acceptance and 8.33% unknown single false acceptance, averaged equally across datasets and inventory sizes. These are not product guarantees, and the harder previously observed CORe50 partition still has 15.56% unknown single false acceptance. See the complete report; do not interpret it as a quality certification.

Swift production preprocessing was also checked against the official reference on six real images (minimum embedding cosine 0.99858). See `swift-preprocessing-parity.json`.

`extract.py` re-extracts local manifest images using the released model. `calibrate.py` selects native embedding distance gates and the model-relative competition margin using only development calibration partitions, with one-class Prototype and multi-class Ridge. Raw images and individual embeddings are not published. The current calibration uses 3,605 re-extracted original images across development and previously observed check cohorts.

See `NOTICE`, `LICENSE`, `conversion-parity.json`, and `calibration.json`. Model downloads run locally; no image upload or hosted inference is required.
