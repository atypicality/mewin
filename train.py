"""Fine-tune Mahjong weights or evaluate them using a labeled YOLO dataset."""
import argparse
import json
from pathlib import Path

from mahjong import TRAINING_WEIGHTS, ROOT, load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--weights', type=Path, default=TRAINING_WEIGHTS)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--split', choices=['val', 'test'], default='test')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if args.weights.suffix != '.pt':
        parser.error('Training and evaluation require PyTorch (.pt) weights.')
    if not args.data.is_file():
        parser.error(f'Dataset configuration not found: {args.data}')
    model = load_model(args.weights)
    common = dict(data=str(args.data.resolve()), imgsz=args.imgsz, device=args.device,
                  project=str(ROOT / 'runs'), workers=0)
    if args.evaluate:
        metrics = model.val(**common, split=args.split, plots=True, name='evaluation')
        print(json.dumps(metrics.results_dict, indent=2))
    else:
        # Mirroring characters creates unrealistic tile artwork.
        model.train(**common, epochs=args.epochs, batch=args.batch, patience=20,
                    fliplr=0.0, flipud=0.0, degrees=20, seed=42, name='mahjong')
        print(f'Best weights: {model.trainer.best}')


if __name__ == '__main__':
    main()
