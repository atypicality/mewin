# Mahjong tile detector

The command-line detector reports only dot (circle) tiles, numbered 1–9, plus red five dots
when supported by the selected checkpoint. Other classes are filtered from the
terminal output, JSON, and annotated image. This restricts the existing model's
output; it does not retrain the model or guarantee that other suits cannot be
misclassified as dots.

A local tile detector using pretrained Mahjong-specific weights. The default is
AR-Mahjong's YOLOv8n ONNX detector on all platforms. No photos are uploaded and
no API key is required.

## Web app

Start the local upload interface:

```sh
.venv/bin/python web_app.py
```

Open http://127.0.0.1:8000 in your browser. Drop a JPEG, PNG, WebP, HEIF, or HEIC image,
choose a confidence threshold, and click **Identify tiles**. Switch between
the original and annotated image, review tile names and confidence scores, and
download the annotated JPEG or JSON report. Images are processed in memory on
your computer and are not saved by the web app. Maximum upload: 50 MB / 80 million
pixels. Photos are resized to at most 2400 pixels on the longest edge and rotated
using their orientation metadata. HEIF/HEIC previews are converted locally to JPEG.
JSON boxes use the processed image dimensions (original dimensions are also included). The web app now reports all 42 classes in the existing model: 27 suited tiles,
four winds, three dragons, four flowers, and four seasons. Jokers and red fives
are not separate classes in this checkpoint. The model's accuracy limitations
still apply; removing the circle-only filter does not retrain it. The CLI remains
circle-only.

Use `--port 8001` if port 8000 is occupied. Press Ctrl+C in the terminal to stop.
The server binds only to your computer's loopback address.

## Run

Your existing virtual environment already has the dependencies:

```sh
.venv/bin/python test.py
.venv/bin/python test.py /path/to/photo.jpg --show
.venv/bin/python test.py /path/to/photo.jpg --conf 0.4
```

Both `test.py` and the existing `.venv/test.py` call `mahjong.py`. The default input
is `examples/images.jpg`. To test the one-dot illustration, use
`.venv/bin/python test.py examples/one.jpg`. Outputs are an annotated
JPEG and JSON with tile codes, readable names, confidence scores, and pixel boxes
in `outputs/`. `--show` opens the image viewer. CPU is the default; use `--device mps`
or `--device 0` only with PyTorch weights when that accelerator is available.
The default ONNX model uses `--imgsz 640`. The optional Core ML model also requires
macOS and `--imgsz 640`. Install dependencies using
`.venv/bin/python -m pip install -r requirements.txt`.
The default runs directly through ONNX Runtime on CPU with two inference threads.
It loads one session and uses Pillow for image preparation and annotations, so
ONNX inference does not import PyTorch, Ultralytics, or OpenCV at startup.

The ONNX path adds a white border equal to 40% of the longest image dimension
before inference, then maps detections back onto the original image. The optional
Core ML path uses 20%. The confidence threshold remains 0.4. Scores are model
scores, not measured accuracy.

The default AR-Mahjong model correctly identified the one-dot illustration in
testing, but also mislabeled the five-bamboo photo as one dot. It was selected
for circle-only experimentation at the user's request. These examples are
development checks, not an independent accuracy benchmark. See MODEL_COMPARISON.md.

The original YOLO11 model includes 34 standard tiles, three red fives, and UNKNOWN. It does not
include flowers, seasons, or jokers. Codes: m = characters, p = circles, s = bamboo;
z = honors. Class IDs come from the checkpoint, not an assumed ordering. UNKNOWN
is a trained class, not a guarantee that unfamiliar tiles will be rejected.
Confidence is a model score, not a measured probability of correctness.
The default ONNX and optional Core ML models instead have 34 standard tiles and eight flowers/seasons;
it has no dedicated red-five or joker class. Its labels are normalized to the same
standard tile codes. JSON `class_id` values remain specific to the selected model.

The supplied stylized bird illustration was not detected by either model at
0.15 confidence, or by the default small model at 0.4 confidence. Real-photo training does not establish illustration accuracy.
An empty detection list is not a classification of the tile. Avoid lowering the
threshold until an arbitrary prediction appears and treating that as success.

## Weights and provenance

Default: https://github.com/LYiHub/AR-Mahjong-Assistant-preview
Pinned revision: `e6bc06cbdbc22ab53f40b1ef5ebeca6dc49f8299`.
`models/mahjong-ar-yolov8n.onnx` comes from `server/models/yolo/weights.onnx`.
Its numeric class-name metadata was replaced with the published labels from
`server/models/yolo/class_names.txt`; network weights were not changed.
Source and local SHA256 hashes are in `models/mahjong-ar-yolov8n.provenance.json`.
The source repository does not provide an explicit repository license; its ONNX
metadata identifies Ultralytics AGPL-3.0. This download is for local experimentation.

Optional Core ML model: https://github.com/Andy8647/MahjongVis
Pinned revision: `983491f30e88be28863c3df7edcd2e7a55fab105`.
Downloaded `App/MahjongVis/mahjong_final.mlpackage` to
`models/mahjong-vis.mlpackage`; license: `models/LICENSE.mahjong-vis`.
SHA256 of `Data/com.apple.CoreML/weights/weight.bin`:
`e86190e2ad282b07bfecad12d6ad2a7860968f415cce03080a722a70870d0ee9`.
SHA256 of `Data/com.apple.CoreML/model.mlmodel`:
`8328ae73e3eef5fb1d5580dadf90da5e8c0af3bb30d9e5cfae971bb488a1f27b`.

The following describes the original PyTorch models, which remain available for
inference and fine-tuning. `train.py` still defaults to the small PyTorch model.

Source: https://github.com/nikmomo/Mahjong-YOLO
Pinned revision: `28ffceed232ad95fd019c47a6c51ae7c78791a0e`.

- `models/mahjong-yolo11s.pt`: upstream `trained_models_v2/yolo11s_best.pt`.
  SHA256 `1b755721c3b581fe550d9568c85a6c5cf12558c6232453943b73dc669eafd6fd`.
- `models/mahjong-yolo11m.pt`: upstream `trained_models_v2/yolo11m_best.pt`.
  SHA256 `a00a03dd9229f5809d4de3c9474b9bcdd74b7b42a5a507e92a803755e268d372`.

The small PyTorch model has the best reported mAP50-95 among upstream benchmarked
models: 0.695, versus nano 0.676 and medium 0.652. This is the selection criterion;
it is not a claim that this is the most accurate Mahjong model available anywhere.

The repository reports small-model mAP50 of 0.881 and recall of 0.765 on its own
evaluation. Those numbers do not measure performance on your tiles. Upstream MIT
notice is preserved in `models/LICENSE.upstream`; dependencies have their own licenses.
The original `yolov8n.pt` remains as a general-purpose model and is not used.

Try the medium alternative:

```sh
.venv/bin/python test.py photo.jpg --weights models/mahjong-yolo11m.pt
```

## Improve and measure accuracy

Use representative photos of your actual tiles: different manufacturers, camera
angles, lighting, rotations, crowded hands, and backgrounds. Label every visible
tile with a bounding box and class. Explicitly review all auto-generated labels.
Split by capture session or tile set so near-duplicate photos do not cross splits.
Keep a separate test set untouched during training and threshold selection.

A public labeled dataset linked by the model authors is available at:
https://www.kaggle.com/datasets/shinz114514/mahjong-hand-photos-taken-with-mobile-camera
It has not been downloaded or independently evaluated here. Verify its class-ID
mapping against `dataset.example.yaml` before using it; remap labels if needed.

For custom training, copy `dataset.example.yaml` to `dataset.yaml` and set its
absolute `path` to your dataset root. Create `images/train`, `images/val`,
`images/test`, and matching `labels/train`, `labels/val`, `labels/test` directories.
Each image needs a same-stem `.txt` annotation file. Each row is
`class_id x_center y_center width height`, with coordinates normalized to [0, 1].
Use empty annotation files only for verified negative images.

```sh
.venv/bin/python train.py --data dataset.yaml --epochs 100 --device cpu
.venv/bin/python train.py --data dataset.yaml --evaluate --split test --weights runs/mahjong/weights/best.pt
.venv/bin/python test.py photo.jpg --weights runs/mahjong/weights/best.pt
```

Training prints the actual best-checkpoint path (repeat runs may get a suffix).
Evaluation saves plots and prints metrics. Check per-class errors, precision,
recall, mAP50-95, and the fraction of entire hands recognized correctly. Training
on CPU can be slow. No training is run without labeled data.

Dataset format: https://docs.ultralytics.com/datasets/detect/

The web app reports the model’s predictions directly, with no user-reference
corrections or label overrides.

## Compare with CLIP

Use the **Model** selector to choose **YOLO · tile detector** or **CLIP ·
single-tile classifier**. Both run locally and support the same uploads. CLIP
uses OpenAI ViT-B/32, independent of YOLO, and ranks the same 42 tile labels.
Use one tile per image: CLIP does not locate or count tiles. It always ranks
candidates, even for an image without a tile. The detector threshold is hidden
for CLIP and does not filter its rankings.

Install the dependencies and download the pinned checkpoint once:

```sh
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python clip_classifier.py --download
```

The checkpoint is `openai/clip-vit-base-patch32` at revision
`3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268` (about 600 MB). After download,
CLIP loads with local-only access; photos are never sent to Hugging Face.
Model loading is lazy and cached for the server lifetime.

CLIP sees the full oriented image, fit onto a white 224×224 square, followed by
the model's normalization. Each candidate uses the prompt
`A photo of a Mahjong tile: {tile name}.`, with `(bird)` included for 1 bamboo.
These prompts are fixed, not tuned per upload. All candidates and exact prompts
are in `clip_classifier.py` and in the JSON results.

The UI shows the top five labels. JSON exports all 42 rankings, cosine
similarities, relative softmax scores, model revision, and preprocessing.
Relative CLIP scores sum to one across this fixed candidate set; they are not
calibrated probabilities or comparable to YOLO confidence. No reference-based
corrections are applied to either model.

Model and implementation references:
https://huggingface.co/openai/clip-vit-base-patch32
https://huggingface.co/docs/transformers/model_doc/clip

## Gemini Mahjong assistant

The web app includes an optional Gemini assistant for photo identification and
follow-up questions. YOLO and CLIP still run locally. Clicking **Send to Gemini**
sends the chat transcript to Google; selecting **Include the current photo**, or
clicking **Analyze photo with Gemini**, also sends the prepared photo. API usage
may incur charges under your Google account.

1. Create an API key at https://aistudio.google.com/apikey.
2. Open `.env` in this project and replace `PASTE_YOUR_GEMINI_API_KEY_HERE`:

   ```dotenv
   GEMINI_API_KEY=PASTE_YOUR_GEMINI_API_KEY_HERE
   GEMINI_MODEL=gemini-4-hydrogen
   ```

3. Run `.venv/bin/python web_app.py` and open http://127.0.0.1:8000.
4. Upload a photo and choose **Analyze photo with Gemini**, or type a question in
   the assistant panel. Follow up with corrections and your Mahjong ruleset.

No additional dependencies are required. The backend uses Google's REST
Interactions API. Configuration is read on each request, so key changes do not
require restarting the server. Environment variables override `.env`. The model
can be changed to an image-capable Gemini model available to your account.
The key is never served to the browser; `.env` is excluded from Git. Do not share
that file. `.env.example` provides a reusable configuration template.

Chat stays in browser memory, with a limit of ten exchanges per conversation.
**New chat**, refreshing, or selecting a new photo clears the conversation.
The transcript is resent for follow-ups; include the photo again for visual
follow-ups. Requests use `store: false` for the Interactions resource; Google's
applicable API data policies still apply. There is no automatic retry on errors.

Scoring replies are unverified AI estimates. This integration does not implement
a rules-based scoring calculator. It asks for tile confirmation, a ruleset and
relevant game details before discussing scores.

API references: https://aistudio.google.com/docs/get-started and
https://aistudio.google.com/docs/image-understanding.

## Formal evaluation

[Evaluation plan](EVALUATION_PLAN.md): 10 selected HEIC photos (four singles,
four groups, two rotated tiles) plus six separate development examples. Labels
are in `evaluation/cases.json`; teammate review is pending. No negative scenes
were supplied.

```sh
python evaluate.py --validate-only
python evaluate.py
```

Results: `outputs/evaluation.json` includes exact-hand accuracy, failures by
condition, and paired brightness checks for the web app's YOLO detector.
