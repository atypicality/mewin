import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from mahjong import DEFAULT_WEIGHTS, detect, load_model, tile_code
from onnx_detector import OnnxDetector, nms

REAL_WEIGHTS = Path(os.environ.get('MEWIN_ONNX_WEIGHTS', DEFAULT_WEIGHTS)).expanduser()


class OnnxDetectionTests(unittest.TestCase):
    @unittest.skipUnless(REAL_WEIGHTS.is_file(),
                         f'real ONNX weights not found: {REAL_WEIGHTS}')
    def test_real_model_identifies_rotated_tiles(self):
        model = load_model(REAL_WEIGHTS)
        cases = (('one.jpg', '1p'), ('five.jpg', '5p'))
        angles = (0, 30, 60, 90, 180, 270)

        with tempfile.TemporaryDirectory() as folder:
            for filename, expected in cases:
                with self.subTest(filename=filename):
                    with Image.open(Path(__file__).resolve().parents[1] / 'examples' / filename) as opened:
                        image = opened.convert('RGB')
                    for angle in angles:
                        with self.subTest(angle=angle):
                            source = Path(folder) / f'{Path(filename).stem}-{angle}.jpg'
                            image.rotate(angle, expand=True, fillcolor='white').save(source)
                            result = detect(model, source, REAL_WEIGHTS, .4, 640, 'cpu')
                            labels = [tile_code(model.names[item['class_id']])
                                      for item in result.detections]
                            self.assertIn(expected, labels,
                                          f'{filename} at {angle} degrees: {labels}')

    def test_duplicate_suppression_preserves_separate_tiles(self):
        boxes = np.array([[0, 0, 100, 100], [1, 1, 101, 101], [200, 0, 300, 100]])
        self.assertEqual(nms(boxes, np.array([0.9, 0.8, 0.7])), [0, 2])
        self.assertEqual(nms(np.empty((0, 4)), np.empty(0)), [])

    def test_padding_coordinates_and_non_dot_rejection(self):
        # A 200px square gets an 80px border and is scaled from 360 to 640px.
        # Model center (320,320) and size 40*640/360 correspond to [80,80,120,120].
        output = np.array([[
            [320, 320, 160], [320, 320, 160],
            [40 * 640 / 360] * 3, [40 * 640 / 360] * 3,
            [0.9, 0.8, 0.9], [0.1, 0.1, 0.95]
        ]], dtype=np.float32)
        model = OnnxDetector.__new__(OnnxDetector)
        model.names = {0: '1D', 1: '5B'}
        model.input = SimpleNamespace(name='images')
        model.session = SimpleNamespace(run=lambda *_: [output])
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'tile.png'
            Image.new('RGB', (200, 200), 'white').save(source)
            result = model.detect(source, [0], 0.4, 640, 'cpu')
            self.assertEqual(len(result.detections), 1)
            self.assertEqual(result.detections[0]['class_id'], 0)
            self.assertEqual(result.detections[0]['xyxy'], [80, 80, 120, 120])
            self.assertEqual(model.detect(source, [0], 0.99, 640, 'cpu').detections, [])
            with self.assertRaises(ValueError):
                model.detect(source, [0], 0.4, 960, 'cpu')


if __name__ == '__main__':
    unittest.main()
