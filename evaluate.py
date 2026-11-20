"""Evaluate the web app's YOLO path on manually labeled image cases."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import platform

ROOT = Path(__file__).resolve().parent
VALID_CODES = {f'{n}{s}' for s in 'mps' for n in range(1, 10)} | {f'{n}z' for n in range(1, 8)} | {f'{n}{s}' for s in 'FS' for n in range(1, 5)}


def validate_manifest(path):
    manifest = json.loads(Path(path).read_text())
    cases = manifest['cases']
    seen = set()
    for case in cases:
        if case['id'] in seen:
            raise ValueError(f"Duplicate case ID: {case['id']}")
        seen.add(case['id'])
        if case['status'] not in {'ready', 'pending'} or case['split'] not in {'development', 'test'}:
            raise ValueError(f"Invalid status/split: {case['id']}")
        if case['status'] == 'pending':
            continue
        expected = case['expected']
        if not isinstance(expected, dict) or any(k not in VALID_CODES or type(v) is not int or v < 1 for k, v in expected.items()):
            raise ValueError(f"Invalid expected tile counts: {case['id']}")
        if not (ROOT / case['image']).is_file():
            raise ValueError(f"Missing image: {case['image']}")
    return manifest


def score(expected, observed):
    missing = dict(Counter(expected) - Counter(observed))
    extra = dict(Counter(observed) - Counter(expected))
    category = ('pass' if not missing and not extra else
                'false_detection' if not expected else
                'missing_and_extra' if missing and extra else
                'missed_tiles' if missing else 'extra_tiles')
    return {'passed': expected == observed, 'missing': missing, 'extra': extra,
            'failure_type': category}


def summarize(rows):
    n = len(rows)
    passed = sum(r['passed'] for r in rows)
    return {'cases': n, 'passed': passed, 'exact_hand_accuracy': passed / n if n else None}


def run(manifest, weights, confidence):
    from PIL import ImageEnhance
    from mahjong import load_model
    from web_app import analyze, prepare_image
    model = load_model(weights)
    rows = []
    for case in manifest['cases']:
        if case['status'] != 'ready':
            continue
        data = (ROOT / case['image']).read_bytes()
        def predict(payload):
            result = analyze(model, payload, confidence)
            return dict(Counter(d['tile'] for d in result['detections']))
        observed = predict(data)
        row = {**case, 'image_sha256': hashlib.sha256(data).hexdigest(),
               'observed': observed, **score(case['expected'], observed)}
        if case.get('robustness'):
            image, _ = prepare_image(data)
            dark = ImageEnhance.Brightness(image).enhance(0.7)
            buffer = io.BytesIO()
            dark.save(buffer, format='PNG')
            changed = predict(buffer.getvalue())
            row['robustness'] = {'brightness_factor': 0.7, 'observed': changed,
                                 'flipped': changed != observed,
                                 **score(case['expected'], changed)}
        else:
            row['robustness'] = None
        rows.append(row)
    summaries = {}
    for split in ('development', 'test'):
        selected = [r for r in rows if r['split'] == split]
        paired = [r for r in selected if r['robustness']]
        summaries[split] = {**summarize(selected),
            'slices': {s: summarize([r for r in selected if r['slice'] == s])
                       for s in sorted({r['slice'] for r in selected})},
            'robustness': {'pairs': len(paired),
                'flip_rate': sum(r['robustness']['flipped'] for r in paired) / len(paired) if paired else None,
                'original': summarize(paired),
                'darkened': summarize([r['robustness'] for r in paired])}}
    return {'timestamp_utc': datetime.now(timezone.utc).isoformat(),
            'python': platform.python_version(), 'weights': str(weights),
            'weights_sha256': hashlib.sha256(weights.read_bytes()).hexdigest(),
            'confidence_threshold': confidence, 'pipeline': 'web_app.analyze / YOLO / CPU / 640',
            'manifest': manifest, 'summary': summaries,
            'pending_cases': [c['id'] for c in manifest['cases'] if c['status'] == 'pending'],
            'results': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'evaluation/cases.json')
    parser.add_argument('--weights', type=Path, default=ROOT / 'models/mahjong-ar-yolov8n.onnx')
    parser.add_argument('--conf', type=float, default=0.4)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/evaluation.json')
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    try:
        if not 0 < args.conf <= 1:
            raise ValueError('Confidence must be in (0, 1].')
        manifest = validate_manifest(args.manifest)
        ready = sum(c['status'] == 'ready' for c in manifest['cases'])
        print(f'{ready} ready; {len(manifest["cases"]) - ready} pending')
        if args.validate_only:
            return
        if args.weights.suffix != '.onnx':
            raise ValueError('This evaluator supports the web app ONNX model only.')
        report = run(manifest, args.weights.resolve(), args.conf)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report['summary'], indent=2))
        print(f'Report: {args.output}')
    except (ValueError, OSError, KeyError, TypeError, ImportError) as exc:
        parser.exit(2, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
