FROM python:3.12-slim

# libgl/glib for OpenCV; ffmpeg libraries come with the PyAV and torch wheels
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY solution.py run_submission.py evaluate.py ./
COPY src ./src
COPY weights ./weights

ENV YOLO_OFFLINE=1 PYTHONUNBUFFERED=1
# docker run --gpus all -v /data/test:/data/test -v $PWD/out:/out team \
#   python run_submission.py --videos /data/test --out /out/predictions.json
CMD ["python", "run_submission.py", "--videos", "/data/test", "--out", "/out/predictions.json"]
