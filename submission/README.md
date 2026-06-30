# SpotFakePhoto

Binary classifier for detecting whether an image is an original photo (`0`) or
a photo recaptured from a screen (`1`).

## Install

```powershell
pip install -r requirements.txt
```

## Test any external image

Run this command from the submission folder:

```powershell
python predict.py "C:\full\path\to\image.jpg"
```

The command prints one probability:

- Below `0.5`: predicted original/real photo
- `0.5` or above: predicted screen-recaptured photo
- Close to `0.5`: low-confidence result

Both `best_model.pth` and `global_model.pth` must remain beside `predict.py`.

## Live camera demo

```powershell
python camera_demo.py
```

Open `http://127.0.0.1:8000`, select **Start camera**, and allow camera access.
The page also supports choosing an image from disk.

## Evaluate

Place the dataset beside the scripts with this structure:

```text
Dataset/
  Original_dataset/
  screen_data/
```

Then run:

```powershell
python experiment.py
```

The adapted model scores 92% (23/25 images) on the original validation split
and 75% (6/8 images) on an untouched holdout from the newly collected data.
The external holdout is the more realistic estimate for unrelated images.

## Retrain

```powershell
python -u train.py --epochs 25 --lr 0.0003 --patience 10
```

Retraining replaces `best_model.pth`. The submitted ensemble result uses the
included `global_model.pth` as its global-context component.
