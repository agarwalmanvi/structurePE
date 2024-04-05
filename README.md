# Structure-informed Positional Encoding for Music Generation

This is the repository for our paper:<br>

> M. Agarwal, C. Wang, G. Richard. "Structure-informed Positional Encoding for Music Generation," in _ICASSP 2024_, DOI: 10.1109/ICASSP48485.2024.10448149. <br>

In this work, we propose a _structure-informed positional encoding_ framework for symbolic music generation with Transformers.

On the [companion website](https://www.manviagarwal.com/projects/structurepe/), you can find generated samples as well as 
the supplementary dataset that provides the correct alignment positions for structural labels. 


## Installation

Before you begin, you will need to install [Poetry](https://python-poetry.org/docs/).

Use `poetry install` to create a virtual environment using the `poetry.lock` file. <br>
You can also delete the `poetry.lock` file and run `poetry install` to install the bare minimum requirements, 
which are provided in the `pyproject.toml` file. This can take a while. <br>
You can activate the virtual environment using `poetry shell`.  After activating the shell for the first time, 
you can run `pip install neptune` to install [Neptune](https://docs.neptune.ai/) and `pip install confugue` to install [Confugue](https://confugue.readthedocs.io/en/latest/). 
If you are using Neptune for the first time, you probably also want to follow their 
documentation in order to set your API token correctly. Then, you should change line 44 
in `src/structure_pe/train_pop909.py` to your `username` and `project`:
```python
project = "username/project"
```
However, note that our system is built with PyTorch Lightning, so you can simply change the `train_pop909.py` script to use 
your favourite logging platform and it should still work out of the box.

To seamlessly use any RPE variant, we provide drop-in replacements for the appropriate PyTorch classes 
and functions, which can be found under `src/torch_replace`. For example, `src/torch_replace/nn/modules/transformer.py` needs to 
replace `path/to/torch/nn/models/transformer.py`. You can find the path to the source files of your PyTorch installation by running, 
for example, `import torch; print(torch.__path__)`.

## Dataset

You can find the preprocessed dataset for training length = 512 under `data/pop909/data/$IDX/len_0.5`, where `$IDX` gives 
the song index `001-909`.
You can run preprocessing for other training lengths by running:
```python
cd /path/to/data/pop909
python preprocessing.py --multiple {0.5|1|3}
```
where you can use `0.5` for length 512, `1` for length 1024 and `3` for length 3072.

## Training

You can launch training by running, for example:
```python
python path/to/train_pop909.py --model-dir path/to/exp/pop909/ape/next_note_prediction/512 --lr 0.0001 --root path/to/data/pop909/data
```
The command line options are:
1. `--model-dir`: directory where the correct `config.yaml` file lives (make sure this file is called `config.yaml`!)
2. `--name`: (optionally) provide a name for this run
3. `--lr`: learning rate
4. `--root`: directory where all the data lives

For example, you can train the model for StructureAPE with sinusoidal embedding using a learning rate of 0.001 with:
```python
python $HOME/structurePE/src/structure_pe/train_pop909.py --model-dir $HOME/structurePE/exp/pop909/ape_structure_sine/next_note_prediction/512 --lr 0.001 --root $HOME/structurePE/data/pop909/data
```

## Inference and Metrics

Once the model is trained, you can run inference using `exp/pop909/inference.py`:
```python
python inference.py
```
This should save `.npz` files in `exp/pop909/samples`.

Finally, you should run the binarization using the functions in `exp/pop909/binarization.py`. For example, for each `.npz` file, 
you could run:
```python
from binarization import get_prob, use_threshold, get_tgt, get_mse

f = np.load(npz_file)
probabilities = get_prob(f)
binary_arr = use_threshold(0.5, probabilities)

X, Y = get_tgt(f['fpath'], "next_note_prediction")
mse = get_mse(Y, binary_arr)
```
The metrics that were used in the paper are given in `exp/pop909/metrics.py`.

## Checkpoints

Because the models trained with Structure RPE and Nonstationary StructureRPE require big GPUs (>40GB for `batchsize=1` and `accumulate_grad_batches=8`), 
we release the checkpoints for training lengths 512 and 1024. You can find the checkpoints on [Zenodo](https://zenodo.org/records/10932363).
In `ckpts/README.md`, you can find a table describing the model configuration that corresponds to each model name.