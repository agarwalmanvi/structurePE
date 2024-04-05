import logging
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from confugue import configurable
from torch import nn

from structure_pe.nn import (
    PositionalEncoding, StructureAPE, SineStructureAPE,
    BaselineStructureAPE,
    BaselineStructureRPE,
    MusicTransformerRPE,
    StructureBiasRPE, StructureBiasSineRPE,
    StructureBiasSineNonStationaryRPE
)
from structure_pe.datasets import n_embeddings_KEYS, n_embeddings_LABELS, n_embeddings_CHORD_IDS

LOGGER = logging.getLogger(__name__)


@configurable()
class StructureFormer(pl.LightningModule):
    def __init__(
            self,
            d_in: int,
            d_out: int,
            d_model: int,
            pe_type: str = "ape",
            dropout: float = 0.1,
            lr: float = 1e-4,
    ):
        super(StructureFormer, self).__init__()

        self.d_model = d_model
        self.lr = lr

        self.input2transformer = torch.nn.Linear(
            in_features=d_in,
            out_features=d_model
        )

        pe_in_attention = False
        rpe_arg = None

        if self._cfg.get('add_positional_encoding', True):
            print("\tInitializing model WITH PE")
            self.pe_type = pe_type

            if "ape" in self.pe_type:
                self.pe = self.init_ape(pe_type=self.pe_type, d_model=d_model)

            # # # Relative Positional Encoding # # #
            elif "rpe" in self.pe_type:
                print("\t\tUsing Relative Positional Encoding\n")
                pe_in_attention = True
                rpe_arg = self.pe_type

            else:
                print("\t\tCould not recognize PE type!")
                raise NotImplementedError

        else:
            self.pe_type = None
            self.pe = None
            print("\tInitializing model WITHOUT PE")

        self.decoder = self._cfg['decoder'].configure(
            TransformerDecoder,
            d_model=d_model,
            dropout=dropout,
            pe_in_attention=pe_in_attention,
            rpe_type=rpe_arg,
        )

        self.transformer2output = torch.nn.Linear(
            in_features=d_model,
            out_features=d_out
        )

        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.fwd_loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")

    def init_ape(self, pe_type, d_model):
        """Initialize an APE module and return it"""

        if pe_type == "ape":

            print("\t\tUsing Absolute Positional Encoding")
            return self._cfg['positional_encoding'].configure(PositionalEncoding, d_model=d_model).cpu()

        elif pe_type == "structure_ape":

            print("\t\tUsing Structure APE")
            structure_dims = self._cfg.get('structure_dims')
            structure_dims['key'] = n_embeddings_KEYS
            structure_dims['annotation'] = n_embeddings_LABELS
            structure_dims['chord_ids'] = n_embeddings_CHORD_IDS
            return self._cfg['positional_encoding'].configure(StructureAPE, d_model=d_model,
                                                              structure_dims=structure_dims).cpu()

        elif pe_type == "sine_structure_ape":

            print("\t\tUsing Sine Structure APE")
            return self._cfg['positional_encoding'].configure(SineStructureAPE, d_model=d_model).cpu()

        elif pe_type == "structure_ape_baseline":
            print("\t\tUsing Structure APE Baseline")
            return self._cfg['positional_encoding'].configure(
                BaselineStructureAPE, d_model=d_model
            ).cpu()

        else:

            raise NotImplementedError

    def configure_optimizers(self):
        optimizer = self._cfg['optimizer'].configure(
            torch.optim.Adam,
            lr=self.lr,
            params=self.parameters(),
        )

        @configurable
        def make_scheduler(*, _cfg):
            scheduler_dict = dict(_cfg.get())
            scheduler_dict['scheduler'] = _cfg['scheduler'].configure(
                optimizer=optimizer)
            return scheduler_dict

        schedulers = self._cfg['schedulers'].configure_list(make_scheduler)
        return [optimizer], schedulers

    def training_step(self, batch, batch_idx):
        # shape: batch x seq_len x (track x pitches)
        (x, y), structure, fpath = batch

        # Apply curriculum learning based on increasing length
        curriculum = self._cfg.get('length_curriculum', {})
        if self.trainer.training and curriculum and self.global_step <= curriculum['steps']:
            max_len = (
                    (curriculum['max'] - curriculum['min'])
                    * self.global_step // curriculum['steps']
                    + curriculum['min'])
            max_len = max(max_len, curriculum['min'])
            x = x[:, :max_len, :]
            y = y[:, :max_len, :]
            if structure != {}:
                for k, v in structure.items():
                    structure[k] = v[:, :max_len, :]

        # log training length
        self.log('train/length', x.size(1))

        logits = self.get_logits(x, structure, batch_idx)

        # compute loss
        train_loss = self.loss_fn(logits, y)
        # loss to logger
        self.log("train/loss", train_loss)
        return train_loss

    def validation_step(self, batch, batch_idx):
        (x, y), structure, _ = batch
        # log validation length
        self.log('val/length', x.shape[1])
        logits = self.get_logits(x, structure, batch_idx)
        val_loss = self.loss_fn(logits, y)
        self.log("val/loss", val_loss)
        return val_loss

    def test_step(self, batch, batch_idx, *args, **kwargs):
        savedir = Path(self._cfg['savedir'].get())
        (x, y), structure, fpath = batch
        logits = self.get_logits(x, structure, batch_idx)
        val_loss = self.fwd_loss_fn(logits, y)
        val_loss = torch.mean(val_loss, dim=(0, 2))
        np.savez_compressed(
            savedir / ("batch_"+str(batch_idx)+".npz"),
            fpath=fpath,
            logits=logits.detach().cpu().numpy(),
            val_loss=val_loss.detach().cpu().numpy()
        )

    def get_logits(self, x, structure, batch_idx):

        # compute logits
        out = self.input2transformer(x)

        # # # If using an APE module, apply it before it goes into the Transformer # # #
        if self.pe_type is not None:
            if self.pe_type == "ape":
                out = self.pe(
                    out
                )
            elif self.pe_type == "structure_ape":
                out = self.pe(
                    out, structure
                )
            elif self.pe_type == "sine_structure_ape":
                out = self.pe(out, structure)

        if self.pe_type is not None and "rpe" in self.pe_type:
            out = self.decoder(out, input_raw=x, structure=structure)
        else:
            out = self.decoder(out, structure=None)
        out = self.transformer2output(out)
        return out


@configurable()
class TransformerDecoder(torch.nn.Module):
    def __init__(self, n_layer, n_head, d_model, d_ff, dropout=0.1, activation='relu', batch_first=True,
                 ###
                 pe_in_attention=False,
                 rpe_type=None,
                 pe_seq_len=None,
                 ###
                 ):
        super(TransformerDecoder, self).__init__()
        self.n_layer = n_layer
        self.n_head = n_head
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        self.activation = activation

        self.pe_in_attention = pe_in_attention
        self.rpe_type = rpe_type

        self.init_rpe(pe_seq_len)

        self.decoder_layers = nn.ModuleList()
        for l in range(self.n_layer):
            self.decoder_layers.append(
                torch.nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_head,
                    dim_feedforward=d_ff,
                    dropout=dropout,
                    activation=activation,
                    batch_first=batch_first,
                )
            )

    def init_rpe(self, pe_seq_len):

        if self.pe_in_attention:

            if self.rpe_type == "music_transformer_rpe":
                d_head = self.d_model // self.n_head
                self.rpe_layer = self._cfg['positional_encoding'].configure(
                    MusicTransformerRPE,
                    emb_dim=d_head,
                    max_pos=pe_seq_len // 4
                )
                self.rpe_weights = torch.nn.ModuleList([
                    torch.nn.Linear(d_head, d_head) for _ in range(self.n_layer)
                ])

            elif self.rpe_type == "structure_rpe_baseline":
                self.rpe_layer = self._cfg['positional_encoding'].configure(
                    BaselineStructureRPE,
                    out_features=self.d_model
                )
                self.rpe_weights = torch.nn.ModuleList([
                    torch.nn.Linear(self.d_model, self.d_model) for _ in range(self.n_layer)
                ])

            elif self.rpe_type == "structure_rpe":
                self.rpe_layer = self._cfg['positional_encoding'].configure(
                    StructureBiasRPE,
                    emb_dim=self.d_model,
                )
                self.rpe_weights = torch.nn.ModuleList([
                    torch.nn.Linear(self.d_model, self.d_model) for _ in range(self.n_layer)
                ])

            elif self.rpe_type == "structure_rpe_sine":
                self.rpe_layer = self._cfg['positional_encoding'].configure(
                    StructureBiasSineRPE,
                    emb_dim=self.d_model,
                )
                self.rpe_weights = torch.nn.ModuleList([
                    torch.nn.Linear(self.d_model, self.d_model) for _ in range(self.n_layer)
                ])

            elif self.rpe_type == "structure_rpe_sine_nonstationary":
                self.rpe_layer = self._cfg['positional_encoding'].configure(
                    StructureBiasSineNonStationaryRPE,
                    emb_dim=self.d_model,
                )
                self.rpe_weights = torch.nn.ModuleList([
                    torch.nn.Linear(self.d_model, self.d_model) for _ in range(self.n_layer)
                ])

            else:
                raise NotImplementedError

        else:
            print("\t\tNote: NOT using RPE!\n\n")


    def forward(self, x, input_raw=None, structure=None):
        # x.shape: batch x seq_len x emb_dim
        with torch.no_grad():
            seq_len = x.size(1)
            lengths = torch.arange(1, seq_len + 1, device=x.device)
            indices = torch.arange(seq_len, device=x.device)
            # matrix with everything above diagonal is set to True due to required format for attn_mask
            attn_mask = ~(indices.view(1, -1) < lengths.view(-1, 1))

        out = x

        if self.pe_in_attention:

            # construct PE embeddings
            if (
                    self.rpe_type == "structure_rpe" or
                    self.rpe_type == "structure_rpe_sine" or
                    self.rpe_type == "structure_rpe_sine_nonstationary" or
                    self.rpe_type == "structure_rpe_nonstationary"
            ):
                assert structure is not None
                pe_embeddings_common = self.rpe_layer(
                    structure=structure
                )
            elif self.rpe_type == "music_transformer_rpe":
                pe_embeddings_common = self.rpe_layer(
                    seq_len_q=out.shape[1],
                    seq_len_k=out.shape[1],
                    dvc=out.device
                )

            elif self.rpe_type == "structure_rpe_baseline":
                assert input_raw is not None
                pe_embeddings_common = self.rpe_layer(
                    x=input_raw,
                    onset=structure["onset"]
                )

        for l in range(self.n_layer):

            if self.pe_in_attention:

                # use PE embeddings in attention matrices
                if (
                        self.rpe_type == "music_transformer_rpe" or
                        self.rpe_type == "structure_rpe_baseline" or
                        self.rpe_type == "structure_rpe" or
                        self.rpe_type == "structure_rpe_sine" or
                        self.rpe_type == "structure_rpe_sine_nonstationary" or
                        self.rpe_type == "structure_rpe_nonstationary"
                ):
                    pe_embeddings = [
                        self.rpe_weights[l](pe_embeddings_) for pe_embeddings_ in pe_embeddings_common
                    ]
                    dict_kwargs = {
                        "src": out,
                        "pe_embeddings": pe_embeddings,
                        "src_mask": attn_mask
                    }
            else:
                dict_kwargs = {
                    "src": out,
                    "src_mask": attn_mask
                }

            out = self.decoder_layers[l](
                **dict_kwargs
            )


        return out
