"""MNIST DataModule"""
import sys
sys.path.append('/content/FSDL-2021-futureskill/lab1') # 없으면 에러가 발생합니다. 데이터셋 파일의 최상단에 lab 차시 맞게 넣어주세요
print('paths: ', sys.path)
import argparse
from torch.utils.data import random_split
from torchvision.datasets import MNIST as TorchMNIST
from torchvision import transforms

from text_recognizer.data.base_data_module import BaseDataModule, load_and_print_info

DOWNLOADED_DATA_DIRNAME = BaseDataModule.data_dirname() / "downloaded"


class MNIST(BaseDataModule):
    """
    MNIST DataModule.
    Learn more at https://pytorch-lightning.readthedocs.io/en/stable/datamodules.html
    """

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.data_dir = DOWNLOADED_DATA_DIRNAME
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        self.dims = (1, 28, 28)  # dims are returned when calling `.size()` on this object.
        self.output_dims = (1,)
        self.mapping = list(range(10))

    def prepare_data(self):
        """Download train and test MNIST data from PyTorch canonical source."""
        TorchMNIST(self.data_dir, train=True, download=True)
        TorchMNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        """Split into train, val, test, and set dims."""
        mnist_full = TorchMNIST(self.data_dir, train=True, transform=self.transform)
        self.data_train, self.data_val = random_split(mnist_full, [55000, 5000])
        self.data_test = TorchMNIST(self.data_dir, train=False, transform=self.transform)


if __name__ == "__main__":
    load_and_print_info(MNIST)
