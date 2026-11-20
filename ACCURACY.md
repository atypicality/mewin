# Accuracy and evaluation

Run the formal YOLO evaluation from the project root with:

```powershell
.\.venv\Scripts\python.exe evaluate.py
```

The machine-readable report is written to [`outputs/evaluation.json`](./outputs/evaluation.json). When the web app is running, the **Model accuracy** panel displays that report using D3 and provides a JSON download.

## Latest local run

The report was generated with the AR-Mahjong YOLOv8n ONNX checkpoint at a 40% confidence threshold.

| Split | Exact-hand accuracy | Result |
| --- | ---: | ---: |
| Development | 83.3% | 5 / 6 cases |
| Test | 80.0% | 8 / 10 cases |

Slice results:

- Development: clear 100.0% (4/4), illustration 50.0% (1/2).
- Test: clear 100.0% (4/4), multi-tile 75.0% (3/4), difficult 50.0% (1/2).
- Brightness robustness: 0% flip rate; all four paired cases remained exact matches after darkening.

These are exact-hand matches on the repository’s manually labeled development/test cases, not a general accuracy guarantee. The test labels are still marked as pending teammate review in [`evaluation/cases.json`](./evaluation/cases.json). Confidence scores are model scores, not calibrated probabilities. Add representative photos from your own tile sets before using these numbers as a deployment benchmark.
