# Echo of Hands

A hand-sign communication aid for Deaf and non-speaking users. Make a sign at
your webcam, and the app recognises it, shows the phrase on screen, and **speaks
it aloud**. Signs can be chained into a sentence and replayed as one utterance.

There are two front ends over the same trained model:

| | |
|---|---|
| **`web/`** | Browser app. Runs entirely client-side, speaks through the Web Speech API, works on a phone. No server, no uploads. |
| **`app.py`** | OpenCV desktop window. Useful for debugging landmarks and FPS. |

## What it recognises

18 one-handed static signs. Each one is either a universally understood emblem
or a genuine ASL handshape, so a new user does not have to memorise arbitrary
mappings.

| | Sign | Speaks | | Sign | Speaks |
|:-:|---|---|:-:|---|---|
| 👋 | open palm | Hello | 👌 | OK circle | I am okay |
| ✌️ | V | Peace | 🤟 | thumb+index+pinky | I love you |
| 👍 | thumb up | Yes | 📞 | thumb+pinky | Please call a doctor |
| 🚫 | index+middle to thumb | No | 👎 | thumb down | No I do not like this |
| ✋ | flat palm | Stop | 🫴 | palm up | Please give me |
| ✊ | clenched fist | I am in pain | 🆘 | fingers curled in | Come here I need you |
| ☝️ | index up | Wait a moment | 👉 | index sideways | I want that one |
| 💧 | three fingers (W) | I want water | 🤏 | thumb+index almost touching | Just a little |
| 🍽️ | fingertips pinched | I am hungry | 🚻 | thumb tucked (T) | I need the bathroom |

The vocabulary lives in
[`model/keypoint_classifier/gestures.csv`](model/keypoint_classifier/gestures.csv)
— id, label, spoken phrase, emoji, and a description of the shape. Edit that
file to change what any sign says, or to add your own.

## Scope and limits

Worth being clear about, because the difference matters:

- **This is not a sign language translator.** ASL and ISL rely on movement,
  both hands, facial expression and body position. This model classifies the
  *static shape of one hand* — think of it as a fast, hands-free phrase board.
- The classifier sees only **hand geometry normalised to the wrist**. It has no
  idea where your hand is relative to your face or chest, so signs that differ
  only by body position cannot be told apart.
- **Two-handed signs are not supported.** The feature vector is 21 landmarks
  (42 values) from a single hand.
- Reported accuracy is **97.9% on held-out frames**, but those frames come from
  the same recording sessions as the training data, so real-world accuracy on a
  different day and in different lighting will be lower.

## Setup

```bash
pip install -r requirements.txt
```

The pins in [`requirements.txt`](requirements.txt) are deliberate. MediaPipe
1.0 removed the `mp.solutions` API this project is built on, and TensorFlow
2.20 removed `tf.lite.Interpreter`; both break it outright.

> **Windows:** TensorFlow's bundled headers exceed the 260-character path
> limit. If `pip` fails with `No such file or directory ... grpc ... .h`,
> create the virtualenv somewhere short such as `C:\hgrenv` rather than inside
> the project folder.

## Running the web app

It must be served over HTTP — `file://` will not work, because the browser
blocks both the model fetch and camera access.

```bash
cd web
python -m http.server 8000
```

Then open <http://localhost:8000>.

Hold a sign until the meter fills green; the phrase is spoken and appended to
your message. Hold-to-confirm duration, confidence threshold, speech rate and
voice are all adjustable in the page.

The browser runs the same network as Python: `train_model.py` exports the dense
layer weights to `web/model_weights.json` and the page re-implements the
forward pass in JavaScript. Verified identical on all 2,470 training rows
(max probability difference 1.4e-06).

## Running the desktop app

```bash
python app.py
```

Options: `--device`, `--width`, `--height`, `--use_static_image_mode`,
`--min_detection_confidence`, `--min_tracking_confidence`, and
`--point_sign_id` (which class id drives the motion classifier; `-1` disables).

## Adding or retraining signs

**1. Record samples**

```bash
python collect_data.py
```

| Key | Action |
|---|---|
| `n` / `p` | next / previous sign |
| digits + `ENTER` | jump straight to a class id |
| `SPACE` | 3s countdown, then capture a 60-sample burst |
| `u` | undo the last burst |
| `ESC` | quit |

The top-right shows `SAVES TO -> [id] name`, which is the class your next burst
lands in. Aim for ~120 samples per sign, and **keep the hand moving slightly**
while recording — small rotations and distance changes. Sixty identical frames
teach one exact pose; sixty varied ones teach the shape.

To add a new sign, append a row to `gestures.csv` with the next free id, then
record it.

**2. Retrain**

```bash
python train_model.py
```

Reads the class count from `gestures.csv`, holds out a stratified 25% test
split, prints a per-class confusion matrix, and writes the `.keras`, `.tflite`
and `web/model_weights.json` artefacts. Refresh the web page to pick up the new
model.

## Layout

```
app.py                    desktop OpenCV app
collect_data.py           guided training-data collector
train_model.py            retrain + export for web and tflite
requirements.txt          pinned, see notes above
web/
  index.html              browser app (MediaPipe + JS inference + speech)
  model_weights.json      exported dense weights, generated by train_model.py
model/
  keypoint_classifier/    hand sign model, dataset, and gestures.csv
  point_history_classifier/   motion classifier, inherited from upstream
utils/
  cvfpscalc.py            FPS meter
  gestures.py             vocabulary loader
```

`keypoint_classification.ipynb` and `point_history_classification.ipynb` are the
original training notebooks. `train_model.py` supersedes the first one.

## Privacy

Video never leaves the machine. Hand tracking and classification both run
locally — in the browser tab for the web app. The training data
(`keypoint.csv`) contains only normalised landmark coordinates; no images or
video are stored.

## Credits

Built on [hand-gesture-recognition-mediapipe](https://github.com/kinivi/hand-gesture-recognition-mediapipe)
by [Nikita Kiselov](https://github.com/kinivi), itself a translation of
[hand-gesture-recognition-using-mediapipe](https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe)
by [Kazuhito Takahashi](https://twitter.com/KzhtTkhs).

Inherited from those projects: the MediaPipe landmark pipeline, the keypoint
preprocessing, the MLP architecture, the point-history classifier and the
training notebooks.

Added here: the 18-sign communication vocabulary and `gestures.csv` schema, the
guided collector, the retraining and web-export script, the browser app with
speech output and sentence building, and pinned dependencies.

Hand landmark detection by [MediaPipe](https://mediapipe.dev/).

## License

[Apache 2.0](LICENSE), same as the upstream project.
