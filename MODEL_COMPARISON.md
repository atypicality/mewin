# Local dot tile model comparison

Checked October 2, 2026. No default model was changed during this comparison.

After the comparison, the user selected AR-Mahjong YOLOv8n for testing on circle
tiles. It is now the default at `models/mahjong-ar-yolov8n.onnx`, with its
published class labels added to the ONNX metadata and 40% white padding.

## Conclusion

No tested candidate demonstrated reliable dot recognition on the supplied
examples. A correct one-dot prediction alone is insufficient: two alternatives
also confidently classified the five-bamboo image as one dot. There is no
verified best model for this project's tile designs from these checks.

## Images and scope

- `examples/one.jpg`: a stylized one-dot illustration, expected `1p`.
- `examples/images.jpg`: five bamboo, expected no output in dot-only mode.
- `examples/dragon.jpg`: white dragon, expected no output in dot-only mode.
- A generated plain white image: expected no detections.

These are development checks, not a held-out benchmark covering dot tiles 1–9,
different manufacturers, lighting, or multi-tile scenes. Model confidence is not
accuracy. Padding variants were explored on the development images.

## Results

| Candidate | One-dot illustration | Relevant limitation |
| --- | --- | --- |
| Existing Mahjong-YOLO YOLO11s | No detections at 0.4 | Tested borders of 0, 100, and 200 pixels |
| Existing Mahjong-YOLO YOLO11m | No detections at 0.4 | Same padding checks |
| Existing MahjongVis Core ML | Predicted Winter at 0.514 without a border; no detections with 100/200 pixel borders | Does not identify this dot example |
| AR-Mahjong YOLOv8n ONNX | Correct `1D` at 0.901 with 200 pixel padding | With a border of 40% of the longest image dimension, predicted `1D` at 0.885 for one-dot and **incorrectly `1D` at 0.853 for five-bamboo**; white dragon classified as `WD`, blank rejected |
| Embedded-Mahjong-Bot YOLOv9 ONNX | No detections at 0.4 | Tested borders of 0, 100, and 200 pixels at its fixed 800 pixel input size |
| HaseLab ResNet-50 classifier | Correct `1p` at 0.959 on the full image | **Incorrectly `1p` at 0.9981 for five-bamboo**; hand-cropped faces did not solve the confusion |

The HaseLab classifier check used its checkpoint class list and a torchvision
ResNet-50 whose state dictionary loaded successfully, with 224×224 RGB resize
and ImageNet normalization. The model card does not fully specify preprocessing;
this was an exploratory classifier check, not an evaluation of the complete
recommended segmentation/perspective-correction pipeline. Its checkpoint records
validation accuracy 0.9808743 and macro accuracy 0.9852797 on the author's data.

The s-seow ViT single-tile classifier was found but not downloaded/tested after
the model-access request was declined. It is not ranked here.

## Sources and revisions

- [Mahjong-YOLO](https://github.com/nikmomo/Mahjong-YOLO): existing local YOLO11s/m weights.
- [MahjongVis](https://github.com/Andy8647/MahjongVis): existing local Core ML weights.
- [AR-Mahjong-Assistant-preview](https://github.com/LYiHub/AR-Mahjong-Assistant-preview),
  revision `e6bc06cbdbc22ab53f40b1ef5ebeca6dc49f8299`:
  `server/models/yolo/weights.onnx` and the associated `class_names.txt`.
  ONNX metadata labels are numeric; the published label file was used for interpretation.
- [Embedded-Mahjong-Bot](https://github.com/colonel-aureliano/Embedded-Mahjong-Bot),
  revision `47683cf45bc69ba540bb6161c3828a72ea1a592e`:
  `tile_classifier/onnx/weights/best-v4-800-half-fp32.onnx`.
- [HaseLab/mahjong-models](https://huggingface.co/HaseLab/mahjong-models),
  revision `db01725eaa2b40c160282b0f582d22f4c0333150`:
  `classifier-resnet50.pt`, loaded using `weights_only=True`.
- [s-seow/mahjong-tile-classifier-vit](https://huggingface.co/s-seow/mahjong-tile-classifier-vit): untested.

## Next useful comparison

Use labeled real photos containing each dot value 1–9 plus non-dot tiles and
empty backgrounds. Compare correct dot count, missed dots, wrong dot values, and
false detections on the same held-out set. If no pretrained model meets those
checks, fine-tune on the target tile designs with non-dot negative examples.
Restricting outputs to dot classes does not teach a model to reject other suits.

## Local changes from this investigation

ONNX Runtime 1.19.2 and ONNX 1.17.0 were installed in `.venv` for candidate
testing. Candidate downloads and experimental scripts were kept in `/tmp` during
the comparison. The selected AR-Mahjong model was subsequently installed in
`models` at the user's request. No photos were uploaded.
