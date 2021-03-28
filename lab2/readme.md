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

`notebooks/01-look-at-emnist.ipynb`에서 데이터를 살펴보겠습니다. [링크](https://github.com/Haebuk/FSDL-2021-futureskill/blob/main/lab2/notebooks/01-look-at-emnist.ipynb)

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

## MNIST를 인식하기 위한 합성곱 신경망 사용

Lab1에서 MNIST 숫자 데이터 셋에 대한 MLP 모델을 적합한 후 끝냈습니다.

이제 똑같은 목적으로 CNN을 학습하겠습니다:

```sh
python3 training/run_experiment.py --model_class=CNN --data_class=MNIST --max_epochs=5 --gpus=1
```

## EMNIST도 동일하게 진행!

더 커진 EMNIST 데이터셋으로 동일하게 진행합니다:

```sh
python3 training/run_experiment.py --model_class=CNN --data_class=EMNIST --max_epochs=5 --gpus=1
```

한 epoch당 훈련에 약 2분정도의 시간이 소요 됩니다. 실행되는 동안 다음 파트로 넘어가겠습니다.

## 의도적인 과적합

데이터 셋의 표본을 사용하는 것이 빠른 실행에도 도움이 되며, 모델을 나타내는 데에도 도움이 됩니다. (Training & Debug 강의에서 자세히 다룰 예정).

다음과 같은 문구를 추가하여 적용할 수 있습니다. `--overfit_batches=0.01` (0과 1사이의 소수).

또한 `> 1`보다 큰 정수를 사용하여 특정 배치 수를 선택할 수 있습니다. `overfit_batches=2`는 2 개의 배치를 의미합니다.
[pytorch-lightning](https://pytorch-lightning.readthedocs.io/en/stable/debugging.html#make-model-overfit-on-subset-of-data) 참고

```sh
python3 training/run_experiment.py --model_class=CNN --data_class=EMNIST --max_epochs=50 --gpus=1 --overfit_batches=2
```

## 훈련 속도 증가

GPU가 지속적으로 높은 활용도를 유지하도록 하는 한 가지 방법은 '--num_workers=X' 플래그를 사용하여 별도의 작업자 프로세스에서 데이터 사전 처리를 수행하는 것입니다.

```sh
python3 training/run_experiment.py --model_class=CNN --data_class=EMNIST --max_epochs=5 --gpus=1 --num_workers=4
```

## EMNIST을 이용한 인공 데이터 생성

- 프로젝트를 위한 EMNIST 인공 데이터를 생성합니다.
- Browm 코퍼스(말뭉치)에서 표본 문장들을 인용했습니다.
- 각 글자들은 EMNIST에서 랜덤으로 추출되고 한 줄로 정렬됩니다. (중복이 될 수도 있습니다.)
- `notebooks/02-look-at-emnist-lines.ipynb`에서 확인 가능합니다. [링크](https://github.com/Haebuk/FSDL-2021-futureskill/blob/main/lab2/notebooks/02-look-at-emnist-lines.ipynb)

## 과제

`text_recognizers/models/cnn.py` 안에 있는 `CNN` 과 `ConvBlock` 구조를 자유롭게 수정하세요.   

특히 `ConvBlock` 모듈은 아래의 ResNet 블록 처럼 만들어 보세요!

![](./resblock.png)

추가로 도전해 볼 것:

- ResNet에 `BatchNorm`을 추가해보세요. 해당 정보는 공식 [ResNet PyTorch 설명](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py)에서 얻을 수 있습니다.
- `MaxPool2D`를 삭제하고, 대신 스트라이드를 사용해보세요.
- 더 빠른 실행을 위해 커맨드라인 인수를 추가해보세요.
- 예를 들어 `ConvBlock` 블록 수를 조정하는 인수를 추가하면 실행 속도에 영향을 줄 수 있습니다.

