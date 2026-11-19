import io
import unittest
from pathlib import Path
from unittest.mock import Mock

from PIL import Image
from web_app import prepare_image, analyze, analyze_clip, MAX_EDGE
from onnx_detector import DetectionResult


class UploadTests(unittest.TestCase):
    def encode(self, image, format='JPEG', **kwargs):
        data = io.BytesIO()
        image.save(data, format=format, **kwargs)
        return data.getvalue()

    def test_phone_photo_is_resized_instead_of_rejected(self):
        image, original = prepare_image(self.encode(Image.new('RGB', (3024, 4032))))
        self.assertEqual(original, (3024, 4032))
        self.assertEqual(image.size, (1800, MAX_EDGE))

    def test_heif_decodes_to_browser_compatible_rgb(self):
        # Decode a fixed fixture: the app never encodes HEIF.
        data = (Path(__file__).parent / 'fixtures/red.heic').read_bytes()
        image, original = prepare_image(data)
        self.assertEqual(image.mode, 'RGB')
        self.assertEqual(image.size, original)
        self.assertGreater(image.getpixel((40, 60))[0], 240)

    def test_exif_orientation(self):
        exif = Image.Exif(); exif[274] = 6
        image, _ = prepare_image(self.encode(Image.new('RGB', (60, 100)), exif=exif))
        self.assertEqual(image.size, (100, 60))

    def test_invalid_data(self):
        with self.assertRaises(OSError):
            prepare_image(b'not an image')

    def test_processed_preview_matches_detection_coordinates(self):
        def fake_detect(source, *args):
            with Image.open(source) as image:
                image = image.copy()
            return DetectionResult(image, {0: '1D'}, [{'class_id': 0, 'confidence': .8, 'xyxy': [10, 20, 50, 60]}])
        model = Mock(names={0: '1D'})
        model.detect.side_effect = fake_detect
        result = analyze(model, self.encode(Image.new('RGB', (3024, 4032))), .4)
        self.assertEqual((result['width'], result['height']), (1800, 2400))
        self.assertEqual(result['coordinate_space'], 'processed_image')
        self.assertEqual(result['detections'][0]['tile'], '1p')
        self.assertTrue(result['original_image'].startswith('data:image/jpeg;base64,'))

    def test_all_tile_families_reach_results(self):
        labels = ['1D', '5B', '9C', 'EW', 'SW', 'WW', 'NW',
                  'WD', 'GD', 'RD', '1F', '2F', '3F', '4F', '1S', '2S', '3S', '4S']
        model = Mock(names=dict(enumerate(labels)))
        def fake_detect(source, classes, *args):
            self.assertEqual(classes, list(range(len(labels))))
            return DetectionResult(Image.new('RGB', (100, 100)), model.names,
                [{'class_id': i, 'confidence': .9, 'xyxy': [1, 2, 30, 40]} for i in classes])
        model.detect.side_effect = fake_detect
        result = analyze(model, self.encode(Image.new('RGB', (100, 100))), .4)
        self.assertEqual([d['tile'] for d in result['detections']],
                         ['1p', '5s', '9m', '1z', '2z', '3z', '4z', '5z', '6z', '7z',
                          '1F', '2F', '3F', '4F', '1S', '2S', '3S', '4S'])
        self.assertEqual(result['detections'][7]['name'], 'White dragon')

    def test_clip_uses_classifier_and_exports_ranking_not_boxes(self):
        from clip_classifier import CODES, PROMPTS, MODEL_ID, REVISION
        self.assertEqual(len(set(CODES)), 42)
        self.assertEqual(len(PROMPTS), 42)
        classifier = Mock()
        classifier.classify.return_value = [{'tile': '1s', 'name': '1 bamboo', 'score': .15}]
        result = analyze_clip(classifier, self.encode(Image.new('RGB', (100, 200))))
        classifier.classify.assert_called_once()
        self.assertEqual(result['analysis_type'], 'classification')
        self.assertEqual(result['model_id'], MODEL_ID)
        self.assertEqual(result['model_revision'], REVISION)
        self.assertNotIn('detections', result)
        self.assertNotIn('confidence_threshold', result)
        self.assertNotIn('annotated_image', result)


if __name__ == '__main__':
    unittest.main()
