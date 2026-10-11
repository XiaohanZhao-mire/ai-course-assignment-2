# Assignment 2 CNN Image Classifier

This extends the Assignment 1 FastAPI application with a trained CIFAR-10 image
classifier at `POST /classify`. The original `/`, `/generate`, and `/embedding`
endpoints remain available. Assignment 2 is submitted in its own repository:
https://github.com/XiaohanZhao-mire/ai-course-assignment-2

## Run with Docker

From this project folder, with Docker installed and running:

```bash
docker build -t gentext-app .
docker run --rm -p 8000:8000 gentext-app
```

Open **http://localhost:8000/docs**, expand **POST /classify**, click **Try it out**,
and upload a PNG or JPEG. The trained weights are included in
`models/cifar10_cnn.pt`; startup needs no dataset download, training, API key,
or cloud account. The first Docker build needs internet access for Python
packages and the existing spaCy English model. Linux uses CPU PyTorch wheels.
Allow several GB of free disk space for the environment and Docker build.

Example request using an included test image:

```bash
curl -X POST http://localhost:8000/classify \
  -F 'file=@examples/test_00_cat.png'
```

The JSON response contains `class_id`, `class_name`, `confidence`, and a
`probabilities` mapping for all ten classes. Confidence is the largest softmax
probability, not a calibrated guarantee of correctness. Classes, in order, are:

`airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck`.

Uploads are decoded as PNG/JPEG, EXIF-oriented, converted to RGB, resized to
64 x 64 with bilinear interpolation, and converted to tensors with values in
[0, 1]. Grayscale and RGBA uploads are converted to RGB. The endpoint accepts
images up to 5 MiB and 16 million pixels. Missing, empty, corrupt, or unsupported
images return HTTP 422; uploads over 5 MiB return HTTP 413. The classifier always
chooses among the ten CIFAR-10 classes, including for unrelated images.

## Required architecture

The architecture follows `Assignment2.pdf`, using PyTorch. The classroom
SimpleCNN is adapted to 64 x 64 inputs and 100 hidden units. No additional
convolutional layers, BatchNorm, or Dropout are added.

Shapes below omit the batch dimension and use height x width x channels.
PyTorch tensors internally use batch x channels x height x width.

| Layer | Settings | Output shape |
|---|---|---|
| Input | RGB | 64 x 64 x 3 |
| Conv1 + ReLU | 16 filters, 3 x 3, stride 1, padding 1 | 64 x 64 x 16 |
| MaxPool1 | 2 x 2, stride 2 | 32 x 32 x 16 |
| Conv2 + ReLU | 32 filters, 3 x 3, stride 1, padding 1 | 32 x 32 x 32 |
| MaxPool2 | 2 x 2, stride 2 | 16 x 16 x 32 |
| Flatten | 32 x 16 x 16 | 8,192 |
| Fully connected + ReLU | 100 units | 100 |
| Fully connected | 10 logits | 10 |

The model has **825,398 trainable parameters**. CrossEntropyLoss takes the raw
logits during training. Softmax is applied only when returning probabilities.

## Training and evaluation

CIFAR-10 contains 50,000 training images and 10,000 test images. Its native
32 x 32 images are resized to 64 x 64 to meet the assignment. The same resize
and tensor conversion are shared by training, evaluation, and API inference.
There is no augmentation or normalization beyond scaling pixels to [0, 1].

The training loop follows Module 4 Practical 3: Adam, learning rate 0.0005,
batch size 32, CrossEntropyLoss, and 10 epochs. The random seed is 42. The
final epoch is used; the test set is evaluated once after training and is not
used to select hyperparameters or checkpoints. Per-epoch training accuracy
measures predictions made during that epoch as weights are being updated.

The included model correctly classified **6,408 / 10,000 test images (64.08%)**.
Final-epoch training accuracy was **73.45%**. These are measured
results from this run, not a guarantee for arbitrary uploaded images.

Measured results are recorded in `models/metrics.json`, including the full test
confusion matrix, training history, device, package version, and checkpoint
SHA-256. `models/training_history.json` records per-epoch loss and accuracy.
The first ten test images are included under `examples/` with their original
indices and true labels; they were not selected based on correct predictions.

### Reproduce training

Python 3.12 and uv are required for local use:

```bash
uv sync --frozen
uv run python -m scripts.train_cnn --epochs 10 --batch-size 32 --device auto
```

The original dataset server can be slow. The supplied model was trained with
this byte-identical Zenodo mirror, also referenced by NVIDIA NVFlare:

```bash
uv run python -m scripts.train_cnn --epochs 10 --batch-size 32 --device auto \
  --dataset-url 'https://zenodo.org/records/15602362/files/cifar-10-python.tar.gz?download=1'
```

Both sources are checked against torchvision's official archive MD5:
`c58f30108f718f92721af3b95e74349a`. Dataset batches are also verified by
`torchvision.datasets.CIFAR10`. Retraining overwrites the model, metrics, and
example images. The dataset is downloaded to `data/` and excluded from Git and
Docker. Seeds improve repeatability; results can still differ across hardware
and numerical backends. CPU, CUDA, and Apple MPS are supported for training;
API inference always runs on CPU.

### Run locally and test

```bash
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

In another terminal:

```bash
uv run pytest -q
uv run python scripts/smoke_test.py
```

Tests verify all layer sizes, parameter counts, a real gradient update,
checkpoint provenance, API agreement with direct model inference, invalid
uploads, and the original endpoints. GitHub Actions builds and starts the
Linux Docker container, runs the tests with actual model weights, and sends
real HTTP requests. See the Actions tab for each commit's result.

## Original endpoints

```bash
curl --get http://localhost:8000/embedding --data-urlencode 'word=apple'
curl -X POST http://localhost:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"start_word":"generating","length":4}'
```

The embedding response contains the complete 300-dimensional `en_core_web_lg`
vector. Assignment 1 behavior and details are documented in
[Assignment1_README.md](docs/Assignment1_README.md).

## Files and classroom sources

| File | Purpose |
|---|---|
| `app/cnn_model.py` | Required architecture and shared preprocessing |
| `scripts/train_cnn.py` | CIFAR-10 training, held-out evaluation, checkpoint export |
| `app/image_classifier.py` | Image validation and CPU inference with saved weights |
| `app/main.py` | New `/classify` endpoint and original API endpoints |
| `models/` | Trained checkpoint and measured training/test results |
| `examples/` | First ten official test images and true labels |
| `tests/`, `scripts/smoke_test.py` | Architecture, API, and HTTP verification |
| `Dockerfile`, `pyproject.toml`, `uv.lock` | Deployment and locked dependencies |
| `docs/Assignment2_XiaohanZhao.docx` | Editable solutions to all six theory questions |
| `docs/Assignment2_CNN_Solutions.md` | Plain-text copy of the theory solutions |

Assignment2.pdf determines the exact architecture. Module 4 Practical 1 supplies
the supervised-learning setup; Practical 2 explains convolution; Practical 3
supplies the SimpleCNN and explicit training/evaluation loop. The previous
Module 3/Assignment 1 application supplies FastAPI and Docker integration.

Official references:
[PyTorch Conv2d](https://docs.pytorch.org/docs/2.8/generated/torch.nn.Conv2d.html),
[MaxPool2d](https://docs.pytorch.org/docs/2.8/generated/torch.nn.MaxPool2d.html),
[Module training/evaluation modes](https://docs.pytorch.org/docs/2.8/generated/torch.nn.Module.html),
[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html),
[FastAPI file uploads](https://fastapi.tiangolo.com/tutorial/request-files/),
[NVIDIA mirror reference](https://github.com/NVIDIA/NVFlare/commit/fb6ef2187f0c8633c997ac9d4d4a6321518ea450).
