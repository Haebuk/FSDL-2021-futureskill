# Lab 1: MNIST 인식을 위한 기본 구조

메인 프로젝트로 가기 위해, MNIST 데이터에 대해 다층 퍼셉트론(MLP)을 훈련해보겠습니다.

다음 내용들을 다룹니다:
- 프로젝트 구조
- 간단한 PyTorch MLP 모델
- PyTorch-Lightning 기반 훈련
- 다음 코드를 통한 몇 가지 실행: `python3 run_experiment.py`

## 시작하기 전, 반드시 Setup을 완료해주세요!

 실행하기 전 [리드미](/setup/readme.md)를 읽고 완료해주세요.

그 후`fsdl-text-recognizer-2021-labs` 레퍼지토리로 이동한 후에 pull을 합니다. 그리고 lab1 디렉토리로 이동합니다.

```
git pull
cd lab1/
```

## 디렉토리 구조

lab에서 점진적으로 코드베이스를 구축할 것입니다.
매주 새로운 lab이 출시되어 더 많은 코드베이스를 보여줄 것입니다.

이제 가장 기초부터 시작합니다.

```sh
(fsdl-text-recognizer-2021) ➜  lab1 git:(main) ✗ tree -I "logs|admin|wandb|__pycache__"
.
├── readme.md
├── text_recognizer
│   ├── data
│   │   ├── base_data_module.py
│   │   ├── __init__.py
│   │   ├── mnist.py
│   │   └── util.py
│   ├── __init__.py
│   ├── lit_models
│   │   ├── base.py
│   │   └── __init__.py
│   ├── models
│   │   ├── __init__.py
│   │   └── mlp.py
│   └── util.py
└── training
    ├── __init__.py
    └── run_experiment.py
```

코드베이스의 주요 분류는 `text_recognizer` 와 `training`으로 이루어져 있습니다.

첫 번째 `text_recognizer`는 개발중인 파이썬 패키지로 간주해야 하며, 어떤 방식으로든 배포될 것입니다.

두 번째 `training`는 현재 단순히 `run_experiment.py`로 구성되어 있는 `text_recognizer` 개발을 위한 지원 코드입니다.

`text_recognizer`내에도 `data`, `models`, 그리고 `lit_models`의 세부 분류가 있습니다.

이들을 차례로 알아보겠습니다.

### Data

데이터를 처리하는 코드에는 이름이 약간 겹치는 세가지 범위가 있습니다: `DataModule`, `DataLoader`, 그리고 `Dataset` 입니다.

최상단에 `DataModule` 클래스가 있습니다. 다음과 같은 항목을 다룹니다.

- raw 데이터를 다운로드 하거나 합성(인위적인) 데이터를 생성합니다.
- 필요에 따라 데이터를 처리합니다. PyTorch 모델을 통해 처리할 수 있습니다.
- 데이터를 훈련/검증/테스트 세트로 분할합니다.
- 입력값의 차원을 지정합니다. 예) `(C, H, W) 실수형 텐서`
- 타겟에 대한 정보를 지정합니다. 예) 클래스 매핑
- 훈련에 적용할 데이터의 확대 변환을 지정합니다.

위의 작업 과정에서 `DataModule`은 몇 가지 다른 클래스를 사용합니다:

1. 기본 데이터를 `torch Dataset` 래핑하여(wrap) 개별 (선택시 변환 됨) 데이터 인스턴스를 반환합니다.
2. `torch Dataset` 을 `torch DataLoader`로 래핑하여 데이터를 섞은 뒤, 샘플 배치사이즈와 함께 GPU에 전달합니다.

필요하다면 다음 링크에서 정보를 얻을 수 있습니다. [PyTorch 데이터 인터페이스](https://pytorch.org/docs/stable/data.html).

데이터 소스를 사용할 때 이전의 프레임을 사용하지 않기 위해, 간단한 기본 클래스인 [`pl.LightningDataModule`](https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html)로 부터 상속받은 `text_recognizer.data.BaseDataModule`을 정의합니다.
이 상속을 통해 데이터를 PyTorch-Lightning `Trainer`에서 매우 간단하게 사용할 수 있으며, 분산 훈련과 관련된 일반적인 문제를 피할 수 있습니다.

### Models

모델은 일반적으로 "신경망"이라고 알려진 것입니다. 즉, 입력값을 받고, 계산을 담당하는 레이어를 통해 처리하며, 출력값을 생성하는 코드입니다.

가장 중요한 것은 코드가 부분적으로 작성(신경망의 구조)되고 부분적으로 **학습된** (구조의 모든 레이어의 파라미터 또는 가중치)것 입니다.
따라서 모형의 계산은 역전파가 수반되어야 합니다.

PyTorch를 사용하기 때문에, 모든 모델은 이러한 방식으로 학습할 수 있는 `torch.nn.Module`의 하위 클래스입니다. 

### Lit Models

PyTorch-Lightning 을 학습용으로 사용하며, 이는 위에서 정의한 모델이 처리하는 모든 것을 처리할 뿐만 아니라 학습 알고리즘의 세부 사항도 지정하는 `LightningModule` 인터페이스를 정의합니다: 이 인터페이스에서 손실은 모델의 출력과 실측 정보에서 계산되어야 하며. 최적화 도구 및 학습 속도등을 사용해야 합니다.

## Training

이제 훈련에 대해 이해할 준비가 되었습니다.

`training/run_experiment.py` 는 여러 커맨드라인의 파라미터를 처리하는 스크립트입니다.

하단에 실행할 수 있는 스크립트의 예가 있습니다.

```sh
python3 training/run_experiment.py --model_class=MLP --data_class=MNIST --max_epochs=5 --gpus=1
```

`model_class` 와 `data_class` 는 우리의 인수인 반면, `max_epochs` 와 `gpus` 는 `pytorch_lightning.Trainer`로 부터 자동으로 지정된 인수입니다.
또한 다른 `Trainer` 플래그를 사용할 수 있습니다. ([문서](https://pytorch-lightning.readthedocs.io/en/latest/trainer.html#trainer-flags)를 참조하세요) 예를 들어 `--batch_size=512`를 사용할 수 있습니다.

`run_experiment.py` 스크립트는 모델 및 지정된 데이터 클래스에서 커맨드라인 플래그를 선택할 수 있습니다.
예를 들어 `text_recognizer/models/mlp.py` 스크립트는 `MLP` 클래스를 지정하고, 두 커맨드 라인 플래그를 추가합니다: `--fc1`, `--fc2`.

따라서 우리는 아래 코드를 실행할 수 있습니다.

```sh
python3 training/run_experiment.py --model_class=MLP --data_class=MNIST --max_epochs=5 --gpus=1 --fc1=4 --fc2=8
```

그리고 모델이 파라미터 수가 적어 높은 정확도를 달성하지 못하는 것을 관찰할 수 있습니다. :)

## Homework

- `training/run_experiment.py` 을 다른 하이퍼파라미터로 실행해보세요. 예)`--fc1=128 --fc2=64`
- `text_recognizers/models/mlp.py` 구조를 수정해보세요
