"""Detect dot (circle) Mahjong tiles using Mahjong-trained weights."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Best reported mAP50-95 among the upstream benchmarked checkpoints:
# small=0.695, nano=0.676, medium=0.652. Larger is not necessarily more accurate.
# https://github.com/nikmomo/Mahjong-YOLO#currently-available-trained-models
TRAINING_WEIGHTS = ROOT / 'models' / 'mahjong-yolo11s.pt'
DEFAULT_WEIGHTS = ROOT / 'models' / 'mahjong-ar-yolov8n.onnx'
DOT_CODES = {f'{number}p' for number in range(10)}


def tile_code(label):
    """Normalize the two checkpoints' different class naming conventions."""
    honors = {'EW': '1z', 'SW': '2z', 'WW': '3z', 'NW': '4z',
              'WD': '5z', 'GD': '6z', 'RD': '7z'}
    if label in honors:
        return honors[label]
    if len(label) == 2 and label[0] in '123456789' and label[1] in 'BCD':
        return label[0] + {'B': 's', 'C': 'm', 'D': 'p'}[label[1]]
    return label


def tile_name(code):
    bonus = {'1F': 'Plum flower', '2F': 'Orchid flower',
             '3F': 'Chrysanthemum flower', '4F': 'Bamboo flower',
             '1S': 'Spring', '2S': 'Summer', '3S': 'Autumn', '4S': 'Winter'}
    if code in bonus:
        return bonus[code]
    honors = {'1z': 'East wind', '2z': 'South wind', '3z': 'West wind',
              '4z': 'North wind', '5z': 'White dragon', '6z': 'Green dragon',
              '7z': 'Red dragon'}
    if code in honors:
        return honors[code]
    if len(code) == 2 and code[0].isdigit() and code[1] in 'mps':
        suit = {'m': 'characters', 'p': 'circles', 's': 'bamboo'}[code[1]]
        return f"{'Red 5' if code[0] == '0' else code[0]} {suit}"
    return code


def dot_class_ids(names):
    """Find dot classes using labels, since checkpoint class IDs differ."""
    return [index for index, label in names.items() if tile_code(label) in DOT_CODES]


def keep_dot_boxes(result, class_ids):
    """Also filter exported models whose built-in NMS chooses classes itself."""
    allowed = set(class_ids)
    keep = [index for index, cls in enumerate(result.boxes.cls.tolist())
            if int(cls) in allowed]
    result.update(boxes=result.boxes.data[keep])
    return result


def load_model(weights):
    weights = Path(weights).expanduser().resolve()
    if weights.suffix == '.onnx':
        if not weights.is_file():
            raise FileNotFoundError(f'Mahjong weights not found: {weights}')
        try:
            from onnx_detector import OnnxDetector
            model = OnnxDetector(weights)
        except ModuleNotFoundError as exc:
            raise ValueError('Run with .venv/bin/python mahjong.py; install dependencies '
                             'with .venv/bin/python -m pip install -r requirements.txt.') from exc
        if not dot_class_ids(model.names):
            raise ValueError('The ONNX model has no Mahjong dot tile labels.')
        return model
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        if exc.name != 'ultralytics':
            raise
        raise ValueError(
            'This Python interpreter does not have ultralytics installed. '
            'Run with .venv/bin/python mahjong.py or select '
            '.venv/bin/python as your IDE Python interpreter.'
        ) from exc
    weights = Path(weights).expanduser().resolve()
    coreml = weights.suffix == '.mlpackage'
    if not (weights.is_dir() if coreml else weights.is_file()):
        raise FileNotFoundError(f'Mahjong weights not found: {weights}')
    if coreml and sys.platform != 'darwin':
        raise ValueError('Core ML weights require macOS. Use models/mahjong-yolo11s.pt on other systems.')
    model = YOLO(str(weights), task='detect')
    if model.task != 'detect' or not dot_class_ids(model.names):
        raise ValueError('Use Mahjong detection weights with dot tile classes (1p–9p or 1D–9D).')
    return model


def detect(model, source, weights, conf, imgsz, device):
    """Add context around close-ups for exported models, then undo padding."""
    class_ids = dot_class_ids(model.names)
    if Path(weights).suffix == '.onnx':
        return model.detect(source, class_ids, conf, imgsz, device)
    exported = Path(weights).suffix == '.mlpackage'
    if not exported:
        result = model.predict(str(source), conf=conf, imgsz=imgsz, device=device,
                               classes=class_ids, agnostic_nms=True, verbose=False)[0]
        return keep_dot_boxes(result, class_ids)
    import cv2
    import numpy as np
    from PIL import Image, ImageOps
    from ultralytics.engine.results import Results

    if imgsz != 640:
        raise ValueError('The exported model has a fixed image size; use --imgsz 640.')
    with Image.open(source) as image:
        original = np.array(ImageOps.exif_transpose(image).convert('RGB'))[:, :, ::-1].copy()
    height, width = original.shape[:2]
    padding = round(max(height, width) * 0.2)
    padded = cv2.copyMakeBorder(original, padding, padding, padding, padding,
                                cv2.BORDER_CONSTANT, value=(255, 255, 255))
    result = model.predict(padded, conf=conf, imgsz=imgsz, device=device,
                           classes=class_ids, agnostic_nms=True, verbose=False)[0]
    result = keep_dot_boxes(result, class_ids)
    boxes = result.boxes.data.clone()
    boxes[:, :4] -= padding
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, height)
    boxes = boxes[(boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])]
    names = {index: tile_name(tile_code(label)) for index, label in result.names.items()}
    # Keep checkpoint labels for JSON; draw readable names on the original photo.
    readable = Results(original, path=str(source), names=names, boxes=boxes)
    readable.tile_labels = result.names
    return readable


def predict(source, weights=DEFAULT_WEIGHTS, output=ROOT / 'outputs', conf=0.4,
            imgsz=640, device='cpu', show=False):
    source = Path(source).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f'Image not found: {source}')
    if not 0 < conf <= 1:
        raise ValueError('Confidence must be greater than 0 and at most 1.')
    print('Loading Mahjong model...', flush=True)
    model = load_model(weights)
    print(f'Detecting dot tiles in {source.name}...', flush=True)
    result = detect(model, source, weights, conf, imgsz, device)
    detections = getattr(result, 'detections', None)
    if detections is not None:
        for item in detections:
            code = tile_code(result.names[item['class_id']])
            item.update(tile=code, name=tile_name(code))
    else:
        detections = yolo_detections(result)
    detections.sort(key=lambda item: (item['xyxy'][1], item['xyxy'][0]))
    print(f'\nModel predictions for {source.name}:', flush=True)
    for index, item in enumerate(detections, start=1):
        print(f"  Tile {index}: {item['name']} ({item['tile']}) "
              f"— confidence {item['confidence']:.1%}", flush=True)
    if not detections:
        print('  No dot tiles detected above the confidence threshold.', flush=True)
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    annotated = output / f'{source.stem}_detected.jpg'
    report = output / f'{source.stem}_detections.json'
    result.save(filename=str(annotated))
    report.write_text(json.dumps({'image': str(source), 'weights': str(Path(weights).resolve()),
                                  'confidence_threshold': conf, 'detections': detections}, indent=2))
    print(f'Annotated image: {annotated}\nJSON detections: {report}')
    if show:
        result.show()
    return detections


def yolo_detections(result):
    detections = []
    for box in result.boxes:
        class_id = int(box.cls.item())
        labels = getattr(result, 'tile_labels', result.names)
        code = tile_code(labels[class_id])
        detections.append({'class_id': class_id, 'tile': code, 'name': tile_name(code),
                           'confidence': round(float(box.conf.item()), 4),
                           'xyxy': [round(float(v), 2) for v in box.xyxy[0].tolist()]})
    return detections


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', nargs='?', type=Path, default=ROOT / 'examples' / 'five.jpg')
    parser.add_argument('--weights', type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs')
    parser.add_argument('--conf', type=float, default=0.4)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--device', default='cpu', help='cpu, mps, or CUDA device number')
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()
    try:
        predict(args.image, args.weights, args.output, args.conf, args.imgsz, args.device, args.show)
    except (FileNotFoundError, ValueError) as exc:
        parser.exit(2, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
