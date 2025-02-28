"""
EMNIST dataset. Downloads from NIST website and saves as .npz file if not already present.
"""
import sys
sys.path.append('/content/FSDL-2021-futureskill/lab2') # 없으면 에러가 발생합니다. 데이터셋 파일의 최상단에 lab 차시 맞게 넣어주세요
print('paths: ', sys.path)
from pathlib import Path
from typing import Sequence
import json
import os
import shutil
import zipfile

from torchvision import transforms
import h5py
import numpy as np
import toml
import torch

from text_recognizer.data.base_data_module import _download_raw_dataset, BaseDataModule, load_and_print_info
from text_recognizer.data.util import BaseDataset

NUM_SPECIAL_TOKENS = 4
SAMPLE_TO_BALANCE = True  # If true, take at most the mean number of instances per class.
TRAIN_FRAC = 0.8

RAW_DATA_DIRNAME = BaseDataModule.data_dirname() / "raw" / "emnist" # 데이터를 저장할 위치
METADATA_FILENAME = RAW_DATA_DIRNAME / "metadata.toml" 
DL_DATA_DIRNAME = BaseDataModule.data_dirname() / "downloaded" / "emnist"
PROCESSED_DATA_DIRNAME = BaseDataModule.data_dirname() / "processed" / "emnist"
PROCESSED_DATA_FILENAME = PROCESSED_DATA_DIRNAME / "byclass.h5"
ESSENTIALS_FILENAME = Path(__file__).parents[0].resolve() / "emnist_essentials.json" # 다운로드 및 처리 후 생성되는 파일의 경로와 이름


class EMNIST(BaseDataModule):
    """
    "The EMNIST dataset is a set of handwritten character digits derived from the NIST Special Database 19
    and converted to a 28x28 pixel image format and dataset structure that directly matches the MNIST dataset."
    From https://www.nist.gov/itl/iad/image-group/emnist-dataset

    The data split we will use is
    EMNIST ByClass: 814,255 characters. 62 unbalanced classes.
    """

    def __init__(self, args=None):
        super().__init__(args)
        if not os.path.exists(ESSENTIALS_FILENAME):
            _download_and_process_emnist()
        with open(ESSENTIALS_FILENAME) as f:
            essentials = json.load(f)
        self.mapping = list(essentials["characters"])
        self.inverse_mapping = {v: k for k, v in enumerate(self.mapping)} # {"<B>": 0, ...}
        self.data_train = None
        self.data_val = None
        self.data_test = None
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.dims = (1, *essentials["input_shape"])  # Extraction
        # Extra dimension is added by ToTensor()
        self.output_dims = (1,)

    def prepare_data(self): # init과 겹침
        if not os.path.exists(PROCESSED_DATA_FILENAME):
            _download_and_process_emnist()
        with open(ESSENTIALS_FILENAME) as f:
            essentials = json.load(f) 

    def setup(self, stage: str = None): # None 이면 둘 다 하는거구나
        if stage == "fit" or stage is None:
            with h5py.File(PROCESSED_DATA_FILENAME, "r") as f:
                self.x_trainval = f["x_train"][:]
                self.y_trainval = f["y_train"][:].squeeze().astype(int)

            data_trainval = BaseDataset(self.x_trainval, self.y_trainval, transform=self.transform)
            train_size = int(TRAIN_FRAC * len(data_trainval))
            val_size = len(data_trainval) - train_size
            self.data_train, self.data_val = torch.utils.data.random_split(
                data_trainval, [train_size, val_size], generator=torch.Generator().manual_seed(42)
            )

        if stage == "test" or stage is None:
            with h5py.File(PROCESSED_DATA_FILENAME, "r") as f:
                self.x_test = f["x_test"][:]
                self.y_test = f["y_test"][:].squeeze().astype(int)
            self.data_test = BaseDataset(self.x_test, self.y_test, transform=self.transform)

    def __repr__(self):
        basic = f"EMNIST Dataset\nNum classes: {len(self.mapping)}\nMapping: {self.mapping}\nDims: {self.dims}\n"
        if self.data_train is None and self.data_val is None and self.data_test is None:
            return basic # load하기 전

        x, y = next(iter(self.train_dataloader()))
        data = (
            f"Train/val/test sizes: {len(self.data_train)}, {len(self.data_val)}, {len(self.data_test)}\n"
            f"Batch x stats: {(x.shape, x.dtype, x.min(), x.mean(), x.std(), x.max())}\n"
            f"Batch y stats: {(y.shape, y.dtype, y.min(), y.max())}\n"
        )
        return basic + data # load후 포함


def _download_and_process_emnist():
    metadata = toml.load(METADATA_FILENAME)
    _download_raw_dataset(metadata, DL_DATA_DIRNAME)
    _process_raw_dataset(metadata["filename"], DL_DATA_DIRNAME)


def _process_raw_dataset(filename: str, dirname: Path):
    print("Unzipping EMNIST...")
    curdir = os.getcwd()
    os.chdir(dirname)
    zip_file = zipfile.ZipFile(filename, "r")
    zip_file.extract("matlab/emnist-byclass.mat")

    from scipy.io import loadmat  # pylint: disable=import-outside-toplevel

    # NOTE: If importing at the top of module, would need to list scipy as prod dependency.

    print("Loading training data from .mat file")
    data = loadmat("matlab/emnist-byclass.mat")
    x_train = data["dataset"]["train"][0, 0]["images"][0, 0].reshape(-1, 28, 28).swapaxes(1, 2)
    y_train = data["dataset"]["train"][0, 0]["labels"][0, 0] + NUM_SPECIAL_TOKENS # 오른쪽으로 4칸씩 shift
    x_test = data["dataset"]["test"][0, 0]["images"][0, 0].reshape(-1, 28, 28).swapaxes(1, 2)
    y_test = data["dataset"]["test"][0, 0]["labels"][0, 0] + NUM_SPECIAL_TOKENS
    # NOTE that we add NUM_SPECIAL_TOKENS to targets, since these tokens are the first class indices

    if SAMPLE_TO_BALANCE:
        print("Balancing classes to reduce amount of data")
        x_train, y_train = _sample_to_balance(x_train, y_train)
        x_test, y_test = _sample_to_balance(x_test, y_test)

    print("Saving to HDF5 in a compressed format...")
    PROCESSED_DATA_DIRNAME.mkdir(parents=True, exist_ok=True) # make parent dir if needed, do not make dir if it already exists
    with h5py.File(PROCESSED_DATA_FILENAME, "w") as f:
        f.create_dataset("x_train", data=x_train, dtype="u1", compression="lzf")
        f.create_dataset("y_train", data=y_train, dtype="u1", compression="lzf")
        f.create_dataset("x_test", data=x_test, dtype="u1", compression="lzf")
        f.create_dataset("y_test", data=y_test, dtype="u1", compression="lzf")

    print("Saving essential dataset parameters to text_recognizer/datasets...")
    mapping = {int(k): chr(v) for k, v in data["dataset"]["mapping"][0, 0]} # dataset mapping
    characters = _augment_emnist_characters(mapping.values())
    essentials = {"characters": characters, "input_shape": list(x_train.shape[1:])} # batch size not needed
    with open(ESSENTIALS_FILENAME, "w") as f: 
        json.dump(essentials, f) #wirte essentials to ESSENTIALS_FILENAME

    print("Cleaning up...")
    shutil.rmtree("matlab") # Remove downloaded data
    os.chdir(curdir) # 원래 위치로


def _sample_to_balance(x, y):
    """Because the dataset is not balanced, we take at most the mean number of instances per class."""
    np.random.seed(42)
    num_to_sample = int(np.bincount(y.flatten()).mean()) # 각 라벨당 몇 개 씩의 X가 존재하는지 세고, 그 평균(=라벨당 X의 수)을 정수로 반환
    all_sampled_inds = []
    for label in np.unique(y.flatten()): # y에 존재하는 라벨 마다..
        inds = np.where(y == label)[0] # return nonzero indicies that are labeled as label
        sampled_inds = np.unique(np.random.choice(inds, num_to_sample)) # 
        all_sampled_inds.append(sampled_inds)
    ind = np.concatenate(all_sampled_inds)
    x_sampled = x[ind]
    y_sampled = y[ind]
    return x_sampled, y_sampled


def _augment_emnist_characters(characters: Sequence[str]) -> Sequence[str]:
    """Augment the mapping with extra symbols."""
    # Extra characters from the IAM dataset
    iam_characters = [
        " ",
        "!",
        '"',
        "#",
        "&",
        "'",
        "(",
        ")",
        "*",
        "+",
        ",",
        "-",
        ".",
        "/",
        ":",
        ";",
        "?",
    ]

    # Also add special tokens:
    # - CTC blank token at index 0
    # - Start token at index 1
    # - End token at index 2
    # - Padding token at index 3
    # NOTE: Don't forget to update NUM_SPECIAL_TOKENS if changing this!
    return ["<B>", "<S>", "<E>", "<P>", *characters, *iam_characters]


if __name__ == "__main__":
    load_and_print_info(EMNIST)
