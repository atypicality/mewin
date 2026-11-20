"""Local web interface for the Mahjong tile detector."""
import argparse
import base64
import io
import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

from PIL import Image, ImageOps, UnidentifiedImageError
from pillow_heif import register_heif_opener
from mahjong import ROOT, DEFAULT_WEIGHTS, load_model, tile_code, tile_name

register_heif_opener(thumbnails=False)
MAX_UPLOAD = 50 * 1024 * 1024
MAX_PIXELS = 80_000_000
MAX_EDGE = 2400


def prepare_image(data):
    with Image.open(io.BytesIO(data)) as source:
        if source.format not in {'JPEG', 'PNG', 'WEBP', 'HEIF'}:
            raise ValueError('Please upload a JPEG, PNG, WebP, HEIF, or HEIC image.')
        if source.width * source.height > MAX_PIXELS:
            raise ValueError('This photo exceeds 80 million pixels. Export a smaller copy.')
        original_size = source.size
        # Reduce before RGB conversion and detector padding to bound working memory.
        source.thumbnail((MAX_EDGE, MAX_EDGE), Image.Resampling.LANCZOS)
        normalized = ImageOps.exif_transpose(source).convert('RGB')
    return normalized, original_size


def image_url(image):
    output = io.BytesIO()
    image.save(output, format='JPEG', quality=90)
    return 'data:image/jpeg;base64,' + base64.b64encode(output.getvalue()).decode()


def analyze(model, data, confidence):
    if not math.isfinite(confidence) or not 0 < confidence <= 1:
        raise ValueError('Confidence must be between 1% and 100%.')
    normalized, original_size = prepare_image(data)
    buffer = io.BytesIO()
    normalized.save(buffer, format='PNG')
    buffer.seek(0)
    start = time.perf_counter()
    # The web app exposes every class in the ONNX checkpoint.
    result = model.detect(buffer, list(model.names), confidence, 640, 'cpu')
    for item in result.detections:
        code = tile_code(result.names[item['class_id']])
        item.update(tile=code, name=tile_name(code))
    result.detections.sort(key=lambda item: (item['xyxy'][1], item['xyxy'][0]))
    annotated = io.BytesIO()
    result.annotated().save(annotated, format='JPEG', quality=90)
    return {'model': 'AR-Mahjong YOLOv8n', 'model_id': 'yolo', 'analysis_type': 'detection',
            'detections': result.detections, 'confidence_threshold': confidence,
            'width': normalized.width, 'height': normalized.height,
            'original_width': original_size[0], 'original_height': original_size[1],
            'coordinate_space': 'processed_image',
            'original_image': image_url(normalized),
            'elapsed_ms': round((time.perf_counter() - start) * 1000),
            'annotated_image': 'data:image/jpeg;base64,' + base64.b64encode(annotated.getvalue()).decode()}


def analyze_clip(classifier, data):
    from clip_classifier import MODEL_ID, REVISION
    normalized, original_size = prepare_image(data)
    start = time.perf_counter()
    predictions = classifier.classify(normalized)
    return {'model': 'CLIP ViT-B/32', 'model_id': MODEL_ID, 'model_revision': REVISION,
            'analysis_type': 'classification', 'score_type': 'softmax_over_42_labels',
            'preprocessing': 'EXIF transpose, max edge 2400, fit full image on white 224x224 square, CLIP normalization',
            'predictions': predictions, 'candidate_count': len(predictions),
            'width': normalized.width, 'height': normalized.height,
            'original_width': original_size[0], 'original_height': original_size[1],
            'original_image': image_url(normalized),
            'elapsed_ms': round((time.perf_counter() - start) * 1000)}


class Handler(BaseHTTPRequestHandler):
    def respond(self, status, data, content_type='application/json'):
        if isinstance(data, dict):
            data = json.dumps(data).encode()
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if urlsplit(self.path).path == '/api/evaluation':
            report = ROOT / 'outputs' / 'evaluation.json'
            if not report.is_file():
                return self.respond(404, {'error': 'No evaluation report found. Run python evaluate.py first.'})
            return self.respond(200, json.loads(report.read_text()))
        if urlsplit(self.path).path == '/api/assistant/status':
            from gemini_assistant import status
            return self.respond(200, status())
        files = {'/': ('index.html', 'text/html; charset=utf-8'),
                 '/app.js': ('app.js', 'text/javascript; charset=utf-8'),
                 '/style.css': ('style.css', 'text/css; charset=utf-8')}
        route = urlsplit(self.path).path
        if route not in files:
            return self.respond(404, {'error': 'Not found.'})
        name, mime = files[route]
        self.respond(200, (ROOT / 'web' / name).read_bytes(), mime)

    def do_POST(self):
        url = urlsplit(self.path)
        if url.path == '/api/chat':
            return self.chat()
        if url.path not in {'/api/detect', '/api/prepare'}:
            return self.respond(404, {'error': 'Not found.'})
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= MAX_UPLOAD:
                self.close_connection = True
                return self.respond(413, {'error': 'Choose an image up to 50 MB.'})
            data = self.rfile.read(size)
            params = parse_qs(url.query)
            selection = params.get('model', ['yolo'])[0]
            if selection not in {'yolo', 'clip'}:
                raise ValueError('Choose YOLO or CLIP.')
            confidence = float(params.get('confidence', ['0.4'])[0])
            with self.server.inference_lock:
                if url.path == '/api/prepare':
                    prepared, original_size = prepare_image(data)
                    result = {'original_image': image_url(prepared),
                              'width': prepared.width, 'height': prepared.height,
                              'original_width': original_size[0], 'original_height': original_size[1]}
                elif selection == 'clip':
                    if self.server.clip_model is None:
                        from clip_classifier import ClipClassifier
                        try:
                            self.server.clip_model = ClipClassifier()
                        except Exception as exc:
                            raise RuntimeError('CLIP could not load. Run .venv/bin/python clip_classifier.py --download and check the installed dependencies.') from exc
                    result = analyze_clip(self.server.clip_model, data)
                else:
                    result = analyze(self.server.model, data, confidence)
            self.respond(200, result)
        except RuntimeError as exc:
            self.respond(503, {'error': str(exc)})
        except ValueError as exc:
            self.respond(400, {'error': str(exc)})
        except Image.DecompressionBombError:
            self.respond(400, {'error': 'This photo is too large. Export a copy under 80 million pixels.'})
        except (OSError, UnidentifiedImageError):
            self.respond(400, {'error': 'This file could not be read. Use a valid JPEG, PNG, WebP, HEIF, or HEIC image.'})
        except Exception:
            self.log_error('Detection failed')
            self.respond(500, {'error': 'Detection failed. Try another image or restart the app.'})

    def chat(self):
        from gemini_assistant import ask
        # Require same-origin JSON so another website cannot spend the local API key.
        origin = self.headers.get('Origin')
        if (self.headers.get('Content-Type', '').split(';')[0] != 'application/json'
                or (origin and origin != 'http://' + self.headers.get('Host', ''))):
            self.close_connection = True
            return self.respond(403, {'error': 'Send chat requests from this app.'})
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= 12 * 1024 * 1024:
                self.close_connection = True
                return self.respond(413, {'error': 'Chat request is too large. Use a smaller photo.'})
            payload = json.loads(self.rfile.read(size))
            if not isinstance(payload, dict):
                raise ValueError('Expected a JSON object.')
            encoded = payload.get('image')
            image = None
            if encoded is not None:
                if not isinstance(encoded, str) or not encoded.startswith('data:image/jpeg;base64,'):
                    raise ValueError('Choose a photo using the upload area first.')
                normalized, _ = prepare_image(base64.b64decode(encoded.split(',', 1)[1], validate=True))
                image = image_url(normalized).split(',', 1)[1]
            self.respond(200, ask(payload, image))
        except (ValueError, OSError, Image.DecompressionBombError):
            self.respond(400, {'error': 'Invalid chat request or image. Check your question and try again.'})
        except RuntimeError as exc:
            self.respond(503, {'error': str(exc)})
        except Exception:
            self.respond(500, {'error': 'The assistant could not complete this request. Please try again.'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    model = load_model(DEFAULT_WEIGHTS)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    server.model = model
    server.clip_model = None
    server.inference_lock = threading.Lock()
    print(f'Mahjong Studio is ready: http://127.0.0.1:{args.port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
