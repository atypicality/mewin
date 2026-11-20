"""Run the AR-Mahjong detector without importing PyTorch or OpenCV."""
import ast
from dataclasses import dataclass


# The pinned AR-Mahjong checkpoint has 42 classes. Some copies retain
# Ultralytics' numeric metadata, so keep the published class order as a
# compatibility fallback for that exact checkpoint shape.
AR_MAHJONG_NAMES = (
    '1B', '1C', '1D', '1F', '1S', '2B', '2C', '2D', '2F', '2S',
    '3B', '3C', '3D', '3F', '3S', '4B', '4C', '4D', '4F', '4S',
    '5B', '5C', '5D', '6B', '6C', '6D', '7B', '7C', '7D', '8B',
    '8C', '8D', '9B', '9C', '9D', 'EW', 'GD', 'NW', 'RD', 'SW',
    'WD', 'WW'
)


@dataclass
class DetectionResult:
    image: object
    names: dict
    detections: list

    def annotated(self):
        from PIL import ImageDraw, ImageFont
        image = self.image.copy()
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default(size=16)
        for item in self.detections:
            box = item['xyxy']
            score = f"{item['confidence']:.1%}"
            label = f"{item.get('name', self.names[item['class_id']])} {score}"
            draw.rectangle(box, outline='blue', width=3)
            x, y = box[:2]
            y = max(0, y - 22)
            draw.rectangle(draw.textbbox((x, y), label, font=font), fill='blue')
            draw.text((x, y), label, fill='white', font=font)
        return image

    def save(self, filename):
        self.annotated().save(filename)

    def show(self):
        self.annotated().show()


def nms(boxes, scores, threshold=0.7):
    """Keep the highest-scoring boxes and suppress overlapping duplicates."""
    import numpy as np
    order = np.argsort(-scores)
    keep = []
    areas = np.prod(np.maximum(0, boxes[:, 2:] - boxes[:, :2]), axis=1)
    while len(order):
        current = order[0]
        keep.append(int(current))
        rest = order[1:]
        intersection = np.prod(np.maximum(
            0, np.minimum(boxes[current, 2:], boxes[rest, 2:])
            - np.maximum(boxes[current, :2], boxes[rest, :2])), axis=1)
        union = areas[current] + areas[rest] - intersection
        overlap = intersection / np.maximum(union, 1e-9)
        order = rest[overlap <= threshold]
    return keep


class OnnxDetector:
    task = 'detect'

    def __init__(self, weights):
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(weights), sess_options=options,
                                            providers=['CPUExecutionProvider'])
        self.input = self.session.get_inputs()[0]
        if self.input.shape != [1, 3, 640, 640]:
            raise ValueError('This ONNX detector requires a fixed 1×3×640×640 RGB input.')
        labels = ast.literal_eval(self.session.get_modelmeta().custom_metadata_map['names'])
        if (len(labels) == len(AR_MAHJONG_NAMES)
                and all(str(index) == label for index, label in labels.items())):
            labels = dict(enumerate(AR_MAHJONG_NAMES))
        self.names = {int(index): label for index, label in labels.items()}

    def detect(self, source, class_ids, conf, imgsz, device):
        import numpy as np
        from PIL import Image, ImageOps
        if imgsz != 640:
            raise ValueError('The AR-Mahjong ONNX model requires --imgsz 640.')
        if device != 'cpu':
            raise ValueError('The direct ONNX detector uses CPU; use --device cpu.')
        with Image.open(source) as opened:
            original = ImageOps.exif_transpose(opened).convert('RGB')
        width, height = original.size
        border = round(max(width, height) * 0.4)
        padded = ImageOps.expand(original, border=border, fill='white')
        scale = min(640 / padded.width, 640 / padded.height)
        resized = padded.resize((round(padded.width * scale), round(padded.height * scale)),
                                Image.Resampling.BILINEAR)
        left = round((640 - resized.width) / 2 - 0.1)
        top = round((640 - resized.height) / 2 - 0.1)
        canvas = Image.new('RGB', (640, 640), (114, 114, 114))
        canvas.paste(resized, (left, top))
        tensor = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
        output = self.session.run(None, {self.input.name: np.ascontiguousarray(tensor)})[0]
        if output.ndim != 3 or output.shape[1] != 4 + len(self.names):
            raise ValueError('Unexpected ONNX output; expected YOLOv8 box and class scores.')
        predictions = output[0].T
        classes = predictions[:, 4:].argmax(axis=1)
        scores = predictions[:, 4:].max(axis=1)
        mask = (scores >= conf) & np.isin(classes, class_ids)
        predictions, classes, scores = predictions[mask], classes[mask], scores[mask]
        boxes = np.concatenate((predictions[:, :2] - predictions[:, 2:4] / 2,
                                predictions[:, :2] + predictions[:, 2:4] / 2), axis=1)
        keep = nms(boxes, scores)
        detections = []
        for index in keep:
            box = (boxes[index] - [left, top, left, top]) / scale - border
            box[[0, 2]] = np.clip(box[[0, 2]], 0, width)
            box[[1, 3]] = np.clip(box[[1, 3]], 0, height)
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            detections.append({'class_id': int(classes[index]),
                               'confidence': round(float(scores[index]), 4),
                               'xyxy': [round(float(value), 2) for value in box]})
        return DetectionResult(original, self.names, detections)
