"""Independent, local CLIP zero-shot classification for a single tile photo."""
import argparse
from pathlib import Path

from mahjong import tile_name

MODEL_ID = 'openai/clip-vit-base-patch32'
REVISION = '3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'
MODEL_DIR = Path(__file__).resolve().parent / 'models' / 'clip-vit-base-patch32'
CODES = [f'{n}{s}' for s in 'mps' for n in range(1, 10)] + [f'{n}z' for n in range(1, 8)] + [f'{n}{s}' for s in 'FS' for n in range(1, 5)]
PROMPTS = [f'A photo of a Mahjong tile: {tile_name(code)}' + (' (bird)' if code == '1s' else '') + '.' for code in CODES]


class ClipClassifier:
    def __init__(self):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        if not (MODEL_DIR / 'pytorch_model.bin').is_file():
            raise RuntimeError('CLIP weights are missing. Run python clip_classifier.py --download, then try again.')
        self.torch = torch
        self.processor = CLIPProcessor.from_pretrained(MODEL_DIR, local_files_only=True, use_fast=False)
        self.model = CLIPModel.from_pretrained(MODEL_DIR, local_files_only=True).eval()
        torch.set_num_threads(2)
        inputs = self.processor(text=PROMPTS, return_tensors='pt', padding=True)
        with torch.inference_mode():
            features = self.model.get_text_features(**inputs)
            self.text_features = features / features.norm(dim=-1, keepdim=True)

    def classify(self, image):
        from PIL import ImageOps
        # Preserve the full image: CLIP's normal center crop would cut portrait photos.
        square = ImageOps.pad(image, (224, 224), color=(255, 255, 255))
        inputs = self.processor(images=square, return_tensors='pt')
        with self.torch.inference_mode():
            features = self.model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
            cosine = (features @ self.text_features.T)[0]
            scores = (cosine * self.model.logit_scale.exp()).softmax(dim=-1)
        predictions = [dict(tile=code, name=tile_name(code), score=round(float(scores[i]), 6),
                            cosine_similarity=round(float(cosine[i]), 6), prompt=PROMPTS[i])
                       for i, code in enumerate(CODES)]
        return sorted(predictions, key=lambda item: item['score'], reverse=True)


def download():
    from huggingface_hub import snapshot_download
    snapshot_download(MODEL_ID, revision=REVISION, local_dir=MODEL_DIR,
                      allow_patterns=['*.json', '*.txt', 'pytorch_model.bin', 'README.md'])
    print(f'CLIP installed at {MODEL_DIR}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true', required=True)
    parser.parse_args()
    download()
