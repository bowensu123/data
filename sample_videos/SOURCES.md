# Sample videos — sources & licenses

20 freely-redistributable clips, for demoing the AURA platform **without a VLM
deployed** (run them through the `mock` backend to populate Browse/Overview with
real video frames + the real streaming-data structure).

## Intel IoT sample videos — CC-BY 4.0
From [`intel-iot-devkit/sample-videos`](https://github.com/intel-iot-devkit/sample-videos)
(licensed CC-BY 4.0). 17 clips:

- bolt-detection.mp4, bolt-multi-size-detection.mp4
- bottle-detection.mp4, car-detection.mp4
- classroom.mp4
- driver-action-recognition.mp4
- face-demographics-walking.mp4, face-demographics-walking-and-pause.mp4
- fruit-and-vegetable-detection.mp4
- head-pose-face-detection-female.mp4, head-pose-face-detection-male.mp4,
  head-pose-face-detection-female-and-male.mp4
- one-by-one-person-detection.mp4, people-detection.mp4
- person-bicycle-car-detection.mp4
- store-aisle-detection.mp4, worker-zone-detection.mp4

## Blender open movies — CC-BY 3.0
(c) Blender Foundation, via [test-videos.co.uk](https://test-videos.co.uk).
Big Buck Bunny — https://www.bigbuckbunny.org ; Sintel — https://durian.blender.org.
3 clips:

- big-buck-bunny-10s.mp4 (720p), big-buck-bunny-1080-10s.mp4 (1080p)
- sintel-10s.mp4 (720p)

## Use in an offline / intranet demo

A ready-made mock dataset over all 20 clips is already bundled in `demo_data/` —
just point the platform at it:

```bash
python -m aura_viz --work-dir demo_data          # → http://localhost:8000  (instant browse)
```

To regenerate it yourself, or run the pipeline live as part of a demo (no VLM,
no install needed):

```bash
# Option A — CLI:
python run_pipeline.py --src sample_videos --work-dir demo_data --client mock
python -m aura_viz --work-dir demo_data

# Option B — all in the browser: python -m aura_viz, open the Run tab,
#            source = sample_videos, config = mock, Run, then Load results.
```
(After `pip install -e .` the shorter `aura-pipeline` / `aura-viz` commands work too.)
The `mock` backend produces the real chunk-wise streaming structure, dual
sliding-window truncation, supervision mask and loss weights, with the **actual
video frames** shown per timestamp — only the QA *text* is synthetic. It needs no
model, API key, GPU, or network, so it runs fully air-gapped.
