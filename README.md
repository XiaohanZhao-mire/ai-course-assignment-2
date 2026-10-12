# Assignment 2

A PyTorch CNN trained on CIFAR-10, with a FastAPI endpoint for image classification. The trained model is included. Test accuracy: 64.08%.

## Run

With Docker running, run these commands in the project folder:

```bash
docker build -t gentext-app .
docker run --rm -p 8000:8000 gentext-app
```

Open http://localhost:8000/docs and upload a PNG or JPEG to `POST /classify`. It returns the predicted class and probabilities.

You can also try:

```bash
curl -X POST http://localhost:8000/classify -F 'file=@examples/test_00_cat.png'
```

The endpoints from Assignment 1 are still available. Theory answers are in [docs/Assignment2_XiaohanZhao.docx](docs/Assignment2_XiaohanZhao.docx).
