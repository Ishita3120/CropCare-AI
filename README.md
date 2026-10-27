# Field Diagnosis — Crop Disease Detector (Web App)

Upload a leaf photo, pick the crop, and get a severity read plus a treatment
recommendation in seconds. Built as a real, runnable Flask app with an
attractive UI — not a mockup.

## What's real vs. a placeholder

- **Real:** the Flask backend, the upload pipeline, severity scoring,
  the treatment knowledge base lookup, and the rule-based treat/no-treat
  decision engine (with dosage scaling from optional humidity/soil-moisture
  input).
- **Placeholder (clearly labeled in the UI):** disease *identification* uses
  a color-signature computer-vision heuristic (`core/image_analyzer.py`), not
  a trained neural network. It genuinely inspects the uploaded photo's pixels
  and gives sensible results on clear leaf photos, but it isn't a substitute
  for a trained CNN/ViT classifier. The function signature it exposes
  (`analyze_image(image, crop) -> AnalysisResult`) is exactly what you'd wire
  a real trained model into — see the `TODO(real model)` comment at the top
  of that file.

## Project structure

```
crop-disease-webapp/
├── app.py                    # Flask routes: / , /predict , /uploads/<file>
├── core/
│   ├── image_analyzer.py     # "model" -- heuristic today, swap for a trained CNN
│   ├── severity.py           # affected-area ratio -> low/medium/high
│   ├── knowledge_base.py     # loads data/treatments.json
│   └── decision_engine.py    # treat/no-treat + dosage, auditable reasons
├── data/
│   └── treatments.json       # sample crop/disease -> treatment table
├── templates/
│   └── index.html            # single-page UI
├── static/
│   ├── css/style.css
│   └── js/main.js            # drag-drop upload, fetch(), report rendering
├── uploads/                  # uploaded photos land here at runtime
├── tests/
│   └── test_analyzer.py
└── requirements.txt
```

## Run it

```bash
pip install -r requirements.txt
python app.py
```

Then open **http://localhost:5000** in your browser. Upload a leaf photo
(tomato, potato or wheat), optionally fill in temperature/humidity/soil
moisture, and click **Analyze leaf**.

Run the tests:

```bash
pytest tests/ -v
```

## Swapping in a real trained model

1. Train a crop/disease classifier (EfficientNet-B0, a YOLO variant, etc.)
   on your own labeled dataset.
2. In `core/image_analyzer.py`, replace the body of `analyze_image()` with
   your model's inference call, keeping the return type `AnalysisResult`
   (`disease_label`, `confidence`, `affected_area_ratio`, `leaf_area_ratio`,
   `method`) exactly as-is.
3. Nothing in `app.py`, the templates, or the frontend needs to change —
   they only depend on that return shape.
4. Expand `data/treatments.json` with more crops/diseases and validated
   dosage information as your model's label set grows.

## Notes for a real deployment

- `app.run(debug=True)` is for local development only — use a production
  WSGI server (gunicorn, uwsgi) behind a reverse proxy for real use.
- Uploaded images are kept in `uploads/`; add a cleanup job or object storage
  if you expect real traffic.
- Treatment dosages in `data/treatments.json` are **sample placeholders** —
  verify against your local agricultural extension or product labels before
  using this for real spraying decisions.
