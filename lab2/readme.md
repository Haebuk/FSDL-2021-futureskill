# Lab 2: 합성곱 신경망

## 이번 랩의 목표

손글씨 이미지를 텍스트로 변환하는 작업을 하고 있습니다.

이번 랩에서는, 다음 내용을 다룹니다.

- 간단한 합성곱 신경망을 이용하여 EMNIST 글자를 인식합니다.
- EMNIST 라인(줄)에 대한 데이터셋을 구축합니다.

## 시작하기전 반드시 세팅을 해주세요!

실행 전, [Lab Setup](/setup/readme.md)을 완료해주시기 바랍니다.

그 후, `fsdl-text-recognizer-2021-labs` 레퍼지토리에서 최근 작업물을 pull한 다음, 아래와 같이 작업공간으로 이동하면 됩니다.

```
git pull
cd lab2
```

## EMNIST란?

MIST는 Mini-NIST를 의미합니다. 여기서 NIST란 1980년대에 손으로 쓴 숫자와 글자의 데이터 셋을 작성한 'National Institute of Standards and Technology'의 약자입니다. 

MNIST는 이 중 숫자만 포함하기 때문에 'Mini'입니다.

EMNIST는 문자를 포함하지만, 널리 알려진 MNIST 형식으로 제공되는 원본 데이터 셋의 재포장된 형태입니다.
[링크](https://www.paperswithcode.com/paper/emnist-an-extension-of-mnist-to-handwritten)를 통해 더 자세한 설명을 살펴 볼 수 있습니다.

`notebooks/01-look-at-emnist.ipynb`에서 데이터를 살펴보겠습니다.

(`lab2`: `notebooks` 작업 경로에 유의하시기 바랍니다. 이 노트북에서 모델을 실행하지는 않지만, 모델을 통해 데이터를 탐색하고, 모델 훈련의 결과는 확인합니다.)

### 요약: 데이터 작업경로 구조


MNIST와 EMNIST 데이터를 인터넷으로부터 다운로드했습니다. 저장 경로는 어디일까요?

```
(fsdl-text-recognizer-2021) ➜  lab2 git:(main) ✗ tree -I "lab*|__pycache__" ..
..
├── data
│   ├── downloaded
│   └── raw
│       ├── emnist
│       │   ├── metadata.toml
│       │   └── readme.md
├── environment.yml
├── Makefile
├── readme.md
├── requirements
└── setup
```

EMNIST 데이터 셋을 다운로드해야 하는 방법과 그 출처에 대한 정보가 포함된 `metadata.toml` 및 `readme.md`으로 나타냅니다.

## Using a convolutional network for recognizing MNIST

We left off in Lab 1 having trained an MLP model on the MNIST digits dataset.

We can now train a CNN for the same purpose:

```sh
python3 training/run_experiment.py --model_class=CNN --data_class=MNIST --max_epochs=5 --gpus=1
```

## Doing the same for EMNIST

We can do the same on the larger EMNIST dataset:

```sh
python3 training/run_experiment.py --model_class=CNN --data_class=EMNIST --max_epochs=5 --gpus=1
```

Training the single epoch will take about 2 minutes (that's why we only do one epoch in this lab :)).
Leave it running while we go on to the next part.

## Intentional overfitting

It is very useful to be able to subsample the dataset for quick experiments and to make sure that the model is robust enough to represent the data (more on this in the Training & Debugging lecture).

This is possible by passing `--overfit_batches=0.01` (or some other fraction).
You can also provide an int `> 1` instead for a concrete number of batches.
https://pytorch-lightning.readthedocs.io/en/stable/debugging.html#make-model-overfit-on-subset-of-data

```sh
python3 training/run_experiment.py --model_class=CNN --data_class=EMNIST --max_epochs=50 --gpus=1 --overfit_batches=2
```

## Speeding up training

One way we can make sure that our GPU stays consistently highly utilized is to do data pre-processing in separate worker processes, using the `--num_workers=X` flag.

```sh
python3 training/run_experiment.py --model_class=CNN --data_class=EMNIST --max_epochs=5 --gpus=1 --num_workers=4
```

## Making a synthetic dataset of EMNIST Lines

- Synthetic dataset we built for this project
- Sample sentences from Brown corpus
- For each character, sample random EMNIST character and place on a line (optionally, with some random overlap)
- Look at: `notebooks/02-look-at-emnist-lines.ipynb`

## Homework

Edit the `CNN` and `ConvBlock` architecture in `text_recognizers/models/cnn.py` in some ways.

In particular, edit the `ConvBlock` module to be more like a ResNet block, as shown in the following image:

![](./resblock.png)

Some other things to try:

- Try adding more of the ResNet secret sauce, such as `BatchNorm`. Take a look at the official ResNet PyTorch implementation for ideas: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
- Remove `MaxPool2D`, perhaps using a strided convolution instead.
- Add some command-line arguments to make trying things a quicker process.
- A good argument to add would be for the number of `ConvBlock`s to run the input through.

Explain what you did, paste the contents of `cnn.py`, and paste your training output into the last question of Gradescope Assignment 2.

As long as you tried a couple of things, you will receive full credit.
