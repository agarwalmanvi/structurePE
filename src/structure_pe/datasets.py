import logging
import os
import string
from random import Random

import numpy as np
import pytorch_lightning as pl
import torch.utils.data
from confugue import configurable
from torch.utils.data import DataLoader

LOGGER = logging.getLogger(__name__)

##################################################################################################

KEYS = [
    "C:maj", "A:min",  # no sharps or flats
    "G:maj", "E:min",  # sharps
    "D:maj", "B:min",
    "A:maj", "F#:min",  # 6, 7
    "E:maj", "C#:min",  # 8, 9
    "B:maj", "G#:min",  # 10, 11
    "F#:maj", "D#:min",  # 12, 13
    "C#:maj", "A#:min",  # 14, 15
    "F:maj", "D:min",  # flats
    "Bb:maj", "G:min",  # 18, 19
    "Eb:maj", "C:min",  # 20, 21
    "Ab:maj", "F:min",  # 22, 23
    "Db:maj", "Bb:min",  # 24, 25
    "Gb:maj", "Eb:min",  # 26, 27
    "Cb:maj", "Ab:min"  # 28, 29
]
KEYS = dict(zip(KEYS, list(range(len(KEYS)))))
KEYS["Gb:min"] = 7
KEYS["Db:min"] = 9
# Others may be needed to be added if the validation set contains alternate key names
n_embeddings_KEYS = 30

# all possible labels
LABELS = list(string.ascii_lowercase) + list(string.ascii_uppercase)
LABELS = dict(zip(LABELS, list(range(len(LABELS)))))
n_embeddings_LABELS = len(LABELS)

LETTERS = string.ascii_lowercase
LETTERS = dict(zip(LETTERS, list(range(len(LETTERS)))))

n_embeddings_CHORD_IDS = 395

lowest_pitch = np.array([53, 29, 27])
highest_pitch = np.array([98, 102, 106])
pitch_range = highest_pitch - lowest_pitch + 1  # array([46, 74, 80])
pitch_dim = int(np.sum(pitch_range))  # 200
master_idx = np.cumsum(np.concatenate((np.array([0]), pitch_range)))  # array([  0,  46, 120, 200])

##################################################################################################
##################################################################################################


@configurable()
class PianoRollPop909DS(torch.utils.data.Dataset):
    """
    Import the preprocessed Pop909 dataset, which has already been transformed into
    a piano roll representation and just convert it to a torch tensor of the right shapes
    """

    def __init__(
            self,
            root: str,
            idx_list=None,
            include_versions=True,
            shuffle_seed: int = 42,
            sample_length: int = 1,
            import_structure: bool = False,
            task: str = "next_note_pred",
            # ds_len: int = None,
            **kwargs
    ):
        super().__init__()

        if idx_list is None:
            raise NotImplementedError

        self.import_structure = import_structure
        self.root = root
        self.sample_length = sample_length
        self.task = task

        print("Shuffling with seed: ", shuffle_seed)
        self.shuffle_gen = Random(shuffle_seed)

        # first convert the list of indices to a list of strings
        idx_list = ["{0:0=3d}".format(i) for i in idx_list]

        # populate a list of filenames, so we can use it to choose a file
        self.filenames = []
        for idx in idx_list:

            idx_root = os.path.join(self.root, idx, "len_" + str(self.sample_length))

            # if the main directory exists, get the filenames
            if os.path.isdir(idx_root):
                self.import_sample_filenames(idx_root)

            else:
                print("\t Did not find folder: ", idx_root)

            if include_versions:

                versions_dirpath = os.path.join(idx_root, "versions")
                # if versions dir exists, get the filenames
                if os.path.isdir(versions_dirpath):
                    self.import_sample_filenames(versions_dirpath)
                else:
                    print("\t Versions folder does not exist : ", versions_dirpath)

        # shuffle just once
        self.shuffle_gen.shuffle(self.filenames)

        print("Files found: ", len(self.filenames))
        print("Importing structure? ", import_structure)

    def import_sample_filenames(self, root_dir):
        """
        Loop over root_dir and select those filenames that are legit samples
        """

        npz_files = []

        if os.path.isdir(root_dir):

            for f in os.listdir(root_dir):
                full_path = os.path.join(root_dir, f)
                if os.path.isfile(full_path) and "sample" in os.path.basename(full_path):
                    # only select file paths that correspond to actual samples
                    npz_files.append(full_path)

        self.filenames.extend(npz_files)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fpath = self.filenames[idx]
        # one single sample of shape dim x (seq_len+1)
        data = np.load(fpath)["data"]
        data = torch.transpose(torch.tensor(data).to(dtype=torch.float), 0, 1)
        # shape: seq_len x dim
        if self.task == "next_note_pred":
            x = data[:-1, :]
            y = data[1:, :]
        elif self.task == "accompaniment_generation":
            x = data[:-1, :256]
            y = data[:-1, 256:]
        else:
            raise NotImplementedError
        seq_len = x.shape[0]

        if not self.import_structure:
            return (x, y), {}, fpath
        else:

            split_fname = os.path.basename(fpath).split("_")
            music_idx, split_idx = split_fname[0], split_fname[1]

            f = np.load(
                os.path.join(os.path.dirname(fpath), music_idx + "_" + split_idx + "_struct_num.npz")
            )

            # fix the shapes to reflect: seq_len (full seq except last timestep) x dim

            # tempo is already converted into an integer value corresponding to the bucket
            tempo_bucket = int(f["tempo_bucket"])
            # shape: seq_len x 1
            tempo_bucket = np.expand_dims(np.repeat(tempo_bucket, seq_len), -1)

            chord_ids = f['chord_ids']
            # shape: seq_len x 1
            chord_ids = torch.transpose(torch.tensor(chord_ids).to(dtype=torch.int), 0, 1)[:seq_len, :]

            melody = f["melody"]
            # shape: seq_len x 1
            melody = torch.transpose(torch.tensor(melody).to(dtype=torch.int), 0, 1)[:seq_len, :]

            annotation_1 = f["annotation_1"]
            # shape: seq_len x 1
            annotation_1 = torch.transpose(torch.tensor(annotation_1).to(dtype=torch.int), 0, 1)[:seq_len, :]

            measure_order = f["measure_order"]
            # shape: seq_len x 1
            measure_order = torch.transpose(torch.tensor(measure_order).to(dtype=torch.int), 0, 1)[:seq_len, :]
            note_order = f["note_order"]
            # shape: seq_len x 1
            note_order = torch.transpose(torch.tensor(note_order).to(dtype=torch.int), 0, 1)[:seq_len, :]

            onset = f["onset"]
            onset = torch.transpose(torch.tensor(onset).to(dtype=torch.float), 0, 1)[:seq_len, :]
            if self.task == "accompaniment_generation":
                # slice it further if required
                onset = onset[:, :256]

            structure_dict = {
                "annotation_1": annotation_1,
                "melody": melody,
                "tempo_bucket": tempo_bucket,
                "chord_ids": chord_ids,
                "measure_order": measure_order,
                "note_order": note_order,
                "onset": onset
            }

            return (x, y), structure_dict, fpath


@configurable
class PianoRollPop909DM(pl.LightningDataModule):
    """
    root should contain the folders 001-909 of all the songs and
    a file split.npz for specifying which songs go into which dataset
    """

    def __init__(
            self,
            batch_size: int,
            num_workers: int,
            shuffle: bool,
            root: str,
            sample_lengths: str = "1,1,1",
            use_test_set: bool = True,
            task: str = "next_note_pred",
            **kwargs
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.persistent_workers = True if self.num_workers > 0 else False
        print("Using ", self.num_workers, " workers; Persistance: ", self.persistent_workers)

        self.use_test_set = use_test_set

        print("Choosing sample lengths: ", sample_lengths)
        try:
            sample_lengths = [int(i) for i in sample_lengths.split(",")]
        except:
            sample_lengths = sample_lengths.split(",")

        # import splits
        f = np.load(os.path.join(root, "split.npz"))
        train_idx, val_idx = f["train_idx"], f["val_idx"]

        print("\nNumber of songs for training data: ", len(train_idx))
        print("Number of songs for validation data: ", len(val_idx))

        print("\nInitializing training set...")
        self.trainset = self._cfg['pytorch_dataset']['train'].configure(PianoRollPop909DS,
                                                                        idx_list=train_idx,
                                                                        root=root,
                                                                        sample_length=sample_lengths[0],
                                                                        task=task,
                                                                        )
        print("\nInitializing validation set...")
        self.validset = self._cfg['pytorch_dataset']['val'].configure(PianoRollPop909DS,
                                                                      idx_list=val_idx,
                                                                      root=root,
                                                                      sample_length=sample_lengths[1],
                                                                      task=task,
                                                                      )
        if self.use_test_set:
            test_idx = f["test_idx"]
            print("Number of songs for testing data: ", len(test_idx))

            print("\nInitializing testing set...")

            self.testset = self._cfg['pytorch_dataset']['test'].configure(PianoRollPop909DS,
                                                                          idx_list=test_idx,
                                                                          root=root,
                                                                          sample_length=sample_lengths[2],
                                                                          task=task,
                                                                          )

    def train_dataloader(self) -> DataLoader:
        loader = DataLoader(
            self.trainset,
            shuffle=self.shuffle,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=True
        )
        return loader

    def val_dataloader(self) -> DataLoader:
        loader = DataLoader(
            self.validset,
            shuffle=self.shuffle,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=True
        )
        return loader

    def test_dataloader(self) -> DataLoader:
        if self.use_test_set:
            loader = DataLoader(
                self.testset,
                shuffle=self.shuffle,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                persistent_workers=self.persistent_workers,
                pin_memory=True
            )
            return loader
        else:
            print("You said you didn't want the test set!")
            raise NotImplementedError