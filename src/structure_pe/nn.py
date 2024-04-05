import math

import torch
from confugue import configurable
from torch import nn

Tensor = torch.Tensor



def create_angles(base, emb_dim):
    i = torch.arange(1, emb_dim + 1)
    # precompute coefficient: B^(2k/d)
    angles = 1 / torch.pow(base, (2 * (i // 2)) / emb_dim)
    # reshape to: 1 x d_model
    angles = torch.unsqueeze(angles, dim=0)
    # self.register_buffer('angles_'+name, angles)

    return angles


class Radians(nn.Module):
    def __init__(self, angles):
        super(Radians, self).__init__()
        self.angles = torch.nn.Parameter(angles, requires_grad=False)

    def forward(self):
        raise NotImplementedError





##################################################################################################
##################################### Baselines ##################################################
##################################################################################################


class PositionalEncoding(nn.Module):
    """
    Vanilla Absolulte Positional Encoding from Vaswani et al., 2017.
    """

    def __init__(self, d_model, max_pos=5000, **kwargs):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_pos, d_model)
        position = torch.arange(0, max_pos, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe.shape = (1, max_pos, d_model)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, **kwargs):
        pe = self.pe[:, :x.size(1), :]
        return x + pe.to(x.device)


# adapted from: https://github.com/gazelle93/Transformer-Various-Positional-Encoding
class MusicTransformerRPE(nn.Module):
    """
    Relative Positional Encoding with contextual bias formulation from Shaw et al. 2018
    """

    def __init__(self, emb_dim, max_pos=512, **kwargs):
        super(MusicTransformerRPE, self).__init__()
        self.max_position = max_pos
        self.embeddings_table = torch.nn.Embedding(max_pos * 2 + 1, emb_dim)

    def forward(self, seq_len_q, seq_len_k, dvc, **kwargs):
        range_vec_q = torch.arange(seq_len_q).to(device=dvc)
        range_vec_k = torch.arange(seq_len_k).to(device=dvc)
        relative_matrix = range_vec_k[None, :] - range_vec_q[:, None]
        clipped_relative_matrix = torch.clamp(relative_matrix, -self.max_position, self.max_position)
        relative_position_matrix = clipped_relative_matrix + self.max_position
        # seq_len x seq_len x emb_dim
        embeddings = self.embeddings_table(relative_position_matrix)

        return [embeddings]


class BaselineStructureAPE(nn.Module):
    """
    Structure APE with learnable embeddings and two methods: sum and weighted sum
    """

    def __init__(
            self,
            d_model,
            max_measure,
            **kwargs
    ):
        super(BaselineStructureAPE, self).__init__()

        self.note_order_embedding = torch.nn.Embedding(64, d_model)
        self.max_measure = max_measure
        self.measure_order_embedding = torch.nn.Embedding(self.max_measure, d_model)

    def forward(self, x, structure_dict):
        note_order = torch.squeeze(self.note_order_embedding(structure_dict["note_order"]), 2)
        measure_order = torch.squeeze(self.measure_order_embedding(
            torch.clamp(structure_dict["measure_order"], min=0, max=self.max_measure)
        ), 2)

        pe = torch.sum(torch.stack([note_order, measure_order], dim=0), dim=0)

        # add elementwise
        return x + pe.to(x.device)


class BaselineStructureRPE(nn.Module):
    def __init__(
            self,
            in_features,
            out_features
    ):
        super().__init__()
        self.relative_pitch_weights = torch.nn.Linear(
            in_features=in_features,
            out_features=out_features
        )
        self.relative_onset_weights = torch.nn.Linear(
            in_features=in_features,
            out_features=out_features
        )

    def get_relative_matrix(self, inp):
        # inp.shape: b x t x e
        b, t, e = inp.shape
        inp_ = torch.permute(inp, (0, 2, 1)).reshape(b * e, t)
        relative_matrix = torch.unsqueeze(inp_, -1) - torch.unsqueeze(inp_, -2)
        relative_matrix = torch.permute(relative_matrix.view(b, e, t, t), (0, 2, 3, 1))
        return relative_matrix

    def forward(self, x, onset):
        # x.shape: b x t x d_model
        relative_pitch = self.relative_pitch_weights(self.get_relative_matrix(x))
        relative_onset = self.relative_onset_weights(self.get_relative_matrix(onset))
        return [relative_pitch, relative_onset]

##################################################################################################
##################################### Our Methods ################################################
##################################################################################################



class StructureAPE(nn.Module):
    """
    Structure APE with learnable embeddings
    """

    def __init__(
            self,
            d_model,
            structure_dims,
            include_structures="chord_ids,tempo_bucket,melody,annotation",
            aggregate_method="sum",
            **kwargs
    ):
        super(StructureAPE, self).__init__()

        self.include_structures = include_structures.split(",")

        print("\t\tIncluding structures: ", self.include_structures)
        assert len(include_structures) > 0

        self.aggregate_method = aggregate_method

        if "chord_ids" in self.include_structures:
            self.chord_id_embedding = torch.nn.Embedding(structure_dims["chord_ids"], d_model)
        if "tempo_bucket" in self.include_structures:
            self.tempo_emb = torch.nn.Embedding(structure_dims["tempo_bucket"], d_model)
        if "melody" in self.include_structures:
            self.melody_emb = torch.nn.Embedding(structure_dims["melody"], d_model)
        if "annotation" in self.include_structures:
            self.annotations_emb = torch.nn.Embedding(structure_dims["annotation"], d_model)

    def forward(self, x, structure_dict):
        pe_add = []
        if "chord_ids" in self.include_structures:
            chord_ids_out = torch.squeeze(self.chord_id_embedding(structure_dict["chord_ids"]), 2)
            pe_add.append(chord_ids_out)
        if "tempo_bucket" in self.include_structures:
            # squeezing out the "original dim" dimension of 1
            tempo_out = torch.squeeze(self.tempo_emb(structure_dict["tempo_bucket"]), 2)
            pe_add.append(tempo_out)
        if "melody" in self.include_structures:
            # squeeze out the extra dimension introduced by retrieving the embeddings
            melody_out = torch.squeeze(self.melody_emb(structure_dict["melody"]), 2)
            pe_add.append(melody_out)
        if "annotation" in self.include_structures:
            annotation_1 = torch.squeeze(self.annotations_emb(structure_dict["annotation_1"]), 2)
            pe_add.append(annotation_1)

        # n_structures x b x seq_len x d_model
        structures_stacked = torch.stack(pe_add, dim=0)
        pe = torch.sum(structures_stacked, dim=0)

        return x + pe.to(x.device)


class SineStructureAPE(nn.Module):
    """
    Structure APE with sinusoidal embeddings
    """

    def __init__(
            self,
            d_model,
            seq_len,
            include_structures="chord_ids,tempo_bucket,melody,annotation",
            aggregate_method="sum",
            same_base=False,
            **kwargs
    ):
        super().__init__()

        self.include_structures = include_structures.split(",")
        n_structures = len(self.include_structures)

        print("\t\tIncluding structures: ", self.include_structures)
        assert len(include_structures) > 0

        self.aggregate_method = aggregate_method

        if same_base:
            self.bases = [10001 for _ in range(4)]
        else:
            self.bases = [
                7920,  # melody
                9919,  # tempo
                10001,  # chord IDS
                11339  # annotations
            ]

        print("\t\tBases : ", self.bases)

        self.angles = torch.nn.ModuleList([
            Radians(create_angles(b, d_model)) for b in self.bases
        ])

    def apply_encoding(self, inp, idx):
        # Apply to annotation
        angle_rads = inp * self.angles[idx].angles  # (b, t, 1) * (1, d_model) -> (b, t, d_model)
        # apply sin to even indices in the array; 2i
        angle_rads[:, :, 0::2] = torch.sin(angle_rads.clone()[:, :, 0::2])
        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, :, 1::2] = torch.cos(angle_rads.clone()[:, :, 1::2])

        return angle_rads

    def forward(self, x, structure_dict, **kwargs):
        # x.shape: b x t x 384

        structures_stacked = []

        if "annotation" in self.include_structures:

            # annotation.shape: b x t x 1
            annotation_1 = structure_dict['annotation_1']

            angle_rads = self.apply_encoding(annotation_1, 3)
            structures_stacked.append(angle_rads.to(x.device))

        if "chord_ids" in self.include_structures:
            # chord_ids.shape: b x t x 1
            chord_ids = structure_dict['chord_ids']

            angle_rads = self.apply_encoding(chord_ids, 2)
            structures_stacked.append(angle_rads.to(x.device))

        if "melody" in self.include_structures:
            # melody.shape: b x t x 1
            melody = structure_dict['melody']

            angle_rads = self.apply_encoding(melody, 0)
            structures_stacked.append(angle_rads.to(x.device))

        if "tempo_bucket" in self.include_structures:
            # tempo.shape: b x t x 1
            tempo = structure_dict['tempo_bucket']

            angle_rads = self.apply_encoding(tempo, 1)
            structures_stacked.append(angle_rads.to(x.device))

        # n_structures x b x t x d_model
        structures_stacked = torch.stack(structures_stacked, dim=0)
        pe = torch.sum(structures_stacked, dim=0)

        return x + pe.to(x.device)

@configurable
class StructureBiasRPE(nn.Module):
    """
    Stationary RPE with structural information; using learnable embeddings
    """

    def __init__(
            self,
            emb_dim,  # must be equal to d_model and not d_model // nhead
            max_pos="128,395,52",
            structure_order="melody,chord_ids,annotation",
            **kwargs
    ):
        super(StructureBiasRPE, self).__init__()

        self.emb_dim = emb_dim

        self.max_position = [int(i) for i in max_pos.split(",")]
        self.structure_order = structure_order.split(",")

        self.embeddings_tables = torch.nn.ModuleList([
            torch.nn.Embedding(
                m * 2 + 1, self.emb_dim
            ) for m in self.max_position
        ])
        self.weights = torch.nn.ModuleList([
            torch.nn.Linear(self.emb_dim, self.emb_dim) for _ in self.max_position
        ])

    def forward(self, structure, **kwargs):

        embeddings = []

        for structure_name, structure_ID in structure.items():

            if (
                    structure_name == "melody" or
                    structure_name == "annotation_1" or
                    structure_name == "chord_ids"
            ):

                if "annotation" in structure_name:
                    s_idx = self.structure_order.index("annotation")
                else:
                    s_idx = self.structure_order.index(structure_name)
                # structure_ID.shape: b x t x 1
                # relative_matrix.shape: b x t x t
                max_pos_structure = self.max_position[s_idx]
                relative_matrix = structure_ID - structure_ID.transpose(-1, -2)
                clipped_relative_matrix = torch.clamp(relative_matrix, -max_pos_structure, max_pos_structure)
                relative_position_matrix = clipped_relative_matrix + max_pos_structure
                # b x t x t x e
                emb = self.embeddings_tables[s_idx](relative_position_matrix)
                emb = self.weights[s_idx](emb)

                embeddings.append(emb)

        return embeddings


@configurable
class StructureBiasSineRPE(nn.Module):
    """
    Stationary RPE with structural information; using sinusoidal embeddings
    """

    def __init__(
            self,
            emb_dim,
            same_base=False,
            include_structures="melody,chord_ids,annotation",
            **kwargs
    ):
        super(StructureBiasSineRPE, self).__init__()
        self.emb_dim = emb_dim
        self.include_structures = include_structures.split(",")

        if same_base:
            self.bases = [10001 for _ in range(4)]
        else:
            self.bases = [
                7920,  # melody
                9919,  # tempo
                10001,  # chord IDS
                11339  # annotations
            ]

        self.angles = torch.nn.ModuleList([
            Radians(create_angles(b, self.emb_dim)) for b in self.bases
        ])
        self.weights = torch.nn.ModuleList([
            torch.nn.Linear(self.emb_dim, self.emb_dim) for _ in self.bases
        ])

    def apply_encoding(self, inp, idx):
        """Apply sinusoidal encoding to the supplied structural input"""

        angle_rads = torch.matmul(inp.to(torch.float),
                                  self.angles[idx].angles)  # (b, t, t, 1) * (1, d_model) -> (b, t, t, d_model)
        # apply sin to even indices in the array; 2i
        angle_rads[:, :, :, 0::2] = torch.sin(angle_rads.clone()[:, :, :, 0::2])
        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, :, :, 1::2] = torch.cos(angle_rads.clone()[:, :, :, 1::2])

        return angle_rads

    def apply_rpe(self, inp, idx):
        # inp.shape: (b, t, 1)
        # we don't consider any max position argument and just embed it as is
        # (b, t, t)
        relative_matrix = inp - inp.transpose(-1, -2)
        # (b, t , t, 1)
        relative_position_matrix = torch.unsqueeze(relative_matrix, -1)
        # (b, t, t, e)
        relative_position_matrix_embedding = self.apply_encoding(relative_position_matrix, idx)
        relative_position_matrix_embedding = self.weights[idx](relative_position_matrix_embedding)

        return relative_position_matrix_embedding

    def forward(self, structure, **kwargs):
        embedded_structures = []

        if "melody" in self.include_structures:
            melody = structure["melody"]
            embedded_structures.append(
                self.apply_rpe(melody, 0)
            )

        if "chord_ids" in self.include_structures:
            chord_ids = structure["chord_ids"]
            embedded_structures.append(
                self.apply_rpe(chord_ids, 2)
            )
        if "annotation" in self.include_structures:
            anno_1 = structure["annotation_1"]
            embedded_structures.append(
                self.apply_rpe(anno_1, 3)
            )
        return embedded_structures


@configurable
class StructureBiasSineNonStationaryRPE(nn.Module):
    """
    Stationary RPE with structure-related information; using sinusoidal embedding
    """

    def __init__(
            self,
            emb_dim,
            same_base=False,
            non_stationary="chord_ids,annotation",
            **kwargs
    ):
        super(StructureBiasSineNonStationaryRPE, self).__init__()
        self.emb_dim = emb_dim
        self.non_stationary = non_stationary.split(",")
        print("Using structures for non-stationary kernel: ", self.non_stationary)

        if same_base:
            self.bases = [10001 for _ in range(2+len(self.non_stationary))]
        else:
            self.bases = [
                10001,  # chord IDS
                11339  # annotations
            ]
            if "chord_ids" in self.non_stationary:
                self.bases.append(9919) # time lags: chord IDs
            if "annotation" in self.non_stationary:
                self.bases.append(24149) # time lags: annotations

        self.angles = torch.nn.ModuleList([
            Radians(create_angles(b, self.emb_dim)) for b in self.bases
        ])
        self.weights = torch.nn.ModuleList([
            torch.nn.Linear(self.emb_dim, self.emb_dim) for _ in self.bases
        ])

    def apply_encoding(self, inp, idx):
        """Apply sinusoidal encoding to the supplied structural input"""

        angle_rads = torch.matmul(inp.to(torch.float),
                                  self.angles[idx].angles)  # (b, t, t, 1) * (1, d_model) -> (b, t, t, d_model)
        # apply sin to even indices in the array; 2i
        angle_rads[:, :, :, 0::2] = torch.sin(angle_rads.clone()[:, :, :, 0::2])
        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, :, :, 1::2] = torch.cos(angle_rads.clone()[:, :, :, 1::2])

        return angle_rads

    def apply_rpe(self, inp, idx):
        # inp.shape: (b, t, 1)
        # we don't consider any max position argument and just embed it as is
        # (b, t, t)
        relative_matrix = inp - inp.transpose(-1, -2)
        # (b, t , t, 1)
        relative_position_matrix = torch.unsqueeze(relative_matrix, -1)
        # (b, t, t, e)
        relative_position_matrix_embedding = self.apply_encoding(relative_position_matrix, idx)
        relative_position_matrix_embedding = self.weights[idx](relative_position_matrix_embedding)

        return relative_position_matrix_embedding

    def get_time(self, annotation_arr):

        batch_size, seq_len = annotation_arr.shape

        changes_batch = torch.nonzero(annotation_arr[:, :-1] != annotation_arr[:, 1:], as_tuple=False)

        time_within_sections_per_sample = []

        for sample_idx in range(batch_size):
            tlines_sample = changes_batch[changes_batch[:, 0] == sample_idx][:, 1]
            tlines_sample = torch.cat(
                (
                    torch.tensor([0.0], device=annotation_arr.device),
                    tlines_sample,
                    torch.tensor([seq_len], device=annotation_arr.device)
                )
            )
            section_lens = torch.tensor(
                [tlines_sample[i + 1] - tlines_sample[i] for i in range(len(tlines_sample) - 1)])
            time_within_sections = torch.cat([torch.arange(l) for l in section_lens])
            time_within_sections_per_sample.append(time_within_sections)

        time_within_sections_per_sample = torch.stack(time_within_sections_per_sample, dim=0)
        assert time_within_sections_per_sample.shape[0] == batch_size
        assert time_within_sections_per_sample.shape[1] == seq_len
        time_within_sections_per_sample = torch.unsqueeze(time_within_sections_per_sample, -1)

        t_minus_s = time_within_sections_per_sample - time_within_sections_per_sample.transpose(-1, -2)
        t_minus_t0 = time_within_sections_per_sample - torch.zeros_like(time_within_sections_per_sample).transpose(-1,
                                                                                                                   -2)

        return t_minus_s.to(annotation_arr.device), t_minus_t0.to(annotation_arr.device)

    def apply_nonstationary_kernel(self, structure, idx):
        # b, t, t
        mask = structure != structure.transpose(-1, -2)

        # b, t, t
        t_minus_s, t_minus_t0 = self.get_time(torch.squeeze(structure, -1))
        # b, t, t, 1
        t_minus_s, t_minus_t0 = torch.unsqueeze(t_minus_s, -1), torch.unsqueeze(t_minus_t0, -1)
        # b, t, t, e
        t_minus_s, t_minus_t0 = self.apply_encoding(t_minus_s, idx), self.apply_encoding(t_minus_t0, idx)
        # b, t, t, e
        t_minus_s, t_minus_t0 = self.weights[idx](t_minus_s), self.weights[idx](t_minus_t0)
        # whereever the sections don't match, set it to 0
        t_minus_s[mask] = 0.0
        t_minus_t0[mask] = 0.0

        return t_minus_s, t_minus_t0

    def forward(self, structure, **kwargs):

        embedded_structures = []

        chord_ids = structure["chord_ids"]
        embedded_structures.append(
            self.apply_rpe(chord_ids, 0)
        )

        anno_1 = structure["annotation_1"]
        embedded_structures.append(
            self.apply_rpe(anno_1, 1)
        )

        if len(self.non_stationary) > 1:

            t_minus_s_chords, t_minus_t0_chords = self.apply_nonstationary_kernel(chord_ids, 2)
            embedded_structures.append(t_minus_s_chords)
            embedded_structures.append(t_minus_s_chords)

            t_minus_s_annotations, t_minus_t0_annotations = self.apply_nonstationary_kernel(anno_1, 3)
            embedded_structures.append(t_minus_t0_annotations)
            embedded_structures.append(t_minus_s_annotations)

        else:
            if "chord_ids" in self.non_stationary:
                t_minus_s_chords, t_minus_t0_chords = self.apply_nonstationary_kernel(chord_ids, 2)
                embedded_structures.append(t_minus_s_chords)
                embedded_structures.append(t_minus_s_chords)
            elif "annotation" in self.non_stationary:
                t_minus_s_annotations, t_minus_t0_annotations = self.apply_nonstationary_kernel(anno_1, 2)
                embedded_structures.append(t_minus_t0_annotations)
                embedded_structures.append(t_minus_s_annotations)
            else:
                raise NotImplementedError

        return embedded_structures
