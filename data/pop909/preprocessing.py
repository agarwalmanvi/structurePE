import argparse
import math
import os
import re
from pathlib import Path

import muspy
import numpy as np
import pandas as pd
import torch
from mido import MidiFile

from structure_pe.datasets import LABELS, LETTERS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--multiple', type=float, default=1.0, required=True)

    args = parser.parse_args()

    # # # # # Change the following paths ! # # # # #
    dir_ = Path("/home/magarwal/repos/POP909-Dataset/POP909/")  # work
    struct_dir_ = Path("/home/magarwal/repos/hierarchical-structure-analysis/POP909/")  # work

    out_ = Path("./data")

    nb_tracks = 3
    nb_pitches = 128
    base_seq_length = 1024
    if args.multiple == 0.5:
        seq_length_multiple = 0.5
    else:
        seq_length_multiple = int(args.multiple)
    print("Seq length multiple: ", seq_length_multiple)
    pred_seq_length = int(base_seq_length * seq_length_multiple) + 1
    if seq_length_multiple == 0.5:
        overlap = 64
    elif seq_length_multiple == 1:
        overlap = 128
    elif seq_length_multiple == 3:
        overlap = 384
    print("Using overlap = ", overlap)
    RESOLUTION = 16
    CHORD_DB = np.load("./data/chord_db.npz")['chord_db']

    f = pd.read_csv("./data/startpos.csv")
    pos = f["Position"].to_numpy()
    song_id = f["ID"].to_numpy()

    MIN_TEMPO = 30
    MAX_TEMPO = 150
    tempo_id = [0] * 30 + [1] * 30 + [2] * 30 + [3] * 30
    tempos = list(range(MIN_TEMPO, MAX_TEMPO))
    tempo_buckets = dict(zip(tempos, tempo_id))

    def read_music(mname):
        # import muspy file and adjust resolution
        m = muspy.read(mname)
        m.adjust_resolution(16)

        return m

    def music_to_samples(music, name, start_pos):
        '''Process a piece of music into a pianoroll matrix'''

        trackname_to_trackidx = {"melody": 0, "bridge": 1, "piano": 2}

        pianorolls = [None, None, None]
        earliest_note = 10000
        max_len = 0

        note_times = [None, None, None]
        note_pitches = [None, None, None]

        for id_, track in enumerate(music.tracks):

            note_times[id_] = np.array([note.time for note in track.notes])
            note_pitches[id_] = np.array([note.pitch for note in track.notes])

            track_idx = trackname_to_trackidx[track.name.lower().strip()]

            new_music_obj = muspy.Music(
                resolution=16,
                tracks=[track]
            )

            # boolean array of dim: time x dim
            pr = muspy.to_pianoroll_representation(music=new_music_obj, encode_velocity=False).astype(float)
            pianorolls[track_idx] = pr

            first_note_time = np.nonzero(np.sum(pr, axis=-1))[0][0]
            if first_note_time < earliest_note:
                earliest_note = first_note_time
            if pr.shape[0] > max_len:
                max_len = pr.shape[0]

        add_time = np.abs(start_pos) if start_pos < 0 else 0
        for idx in range(len(note_times)):
            note_times[idx] = note_times[idx] - earliest_note + add_time

        sample = np.zeros((nb_tracks, nb_pitches, max_len - earliest_note))
        for pr_idx, pr in enumerate(pianorolls):
            # convert to dim x time; make the pianoroll start from the first note that appears
            insert_pr = np.transpose(pr[earliest_note:, :])
            pr_len = insert_pr.shape[-1]
            sample[pr_idx, :, :pr_len] = insert_pr

        sample = torch.from_numpy(sample)
        # reshape to (tracks x pitches) x time
        sample = torch.flatten(sample, start_dim=0, end_dim=1)
        sample = sample.numpy()

        return sample, (note_times, note_pitches)

    def fetch_tempo(dirname):
        '''Return an int for the tempo for the whole song'''

        tempo_f = dirname / "tempo.txt"

        with open(tempo_f) as file:
            tempos_ = [line.rstrip() for line in file]

        tempos_ = [int(i) for i in tempos_ if len(i) > 0]
        # return the first available tempo
        tempo = np.clip(tempos_[0], MIN_TEMPO, MAX_TEMPO - 1)
        tempo_bucket = tempo_buckets[tempo]
        return tempo_bucket

    def get_ID(chord_vector):
        chord_exists = (chord_vector == CHORD_DB).all(axis=1)
        chord_id = np.nonzero(chord_exists)[0][0]

        return chord_id


    def fetch_chords(dirname):
        '''NOTE: Chords are quantized to a quarter note'''

        chord_f = dirname / "finalized_chord.txt"

        with open(chord_f) as file:
            chords_ = [line.rstrip() for line in file]

        chord_id_by_quarter_note = []

        for chord_idx, chord in enumerate(chords_):

            # # # Get a vectorial representation for the chord # # #
            # this is a non-empty chord
            if chord[0] != "N":

                # throw away the name of the chord and process the rest
                s = chord[chord.index('['):]
                # get the pitches as a binary vector indicating which pitches are present
                pitches = s[1:s.index(']')].replace(" ", "").split(',')
                pitches = [int(i) for i in pitches]
                pitch_vec = np.zeros(12)
                pitch_vec[pitches] = 1.0

                # get the root note and the duration
                rest = s[s.index(']') + 2:].split(' ')
                # map root note from 0-11 (original) to 1-12, 0 is reserved for "no root note" when the chord is absent
                root_note = int(rest[0]) + 1
                n_quarter_notes = int(rest[1])

                # 13-dimensional chord vector for this beat
                chord_vec = np.concatenate(([root_note], pitch_vec))

            else:
                print("\t\tFound a case of no chord!")
                n_quarter_notes = chord.split(" ")[-1]
                print("\t\tLasts for ", n_quarter_notes, " quarter notes.")
                try:
                    n_quarter_notes = int(n_quarter_notes)
                    pitch_vec = np.zeros(12)
                    root_note = 0

                    # 13-dimensional chord vector for empty chord
                    chord_vec = np.concatenate(([root_note], pitch_vec))

                    print("\t\tDesigned chord for no chord")

                except:

                    print("does not compute |[@_@]|")
                    raise NotImplementedError

            # # # Extract index from a master list # # #
            chord_id_ = get_ID(chord_vec)

            for _ in range(n_quarter_notes):
                chord_id_by_quarter_note.append(chord_id_)

        # shape: n_quarter_notes
        all_chord_ids = np.array(chord_id_by_quarter_note)
        # shape: (n_quarter_notes x 16)
        all_chord_ids = np.repeat(a=all_chord_ids, repeats=RESOLUTION)

        # return chord IDs
        return all_chord_ids



    def fetch_melody(dirname):
        '''
        Quantized to sixteenth note in the annotations.
        For a resolution of 16 ticks per quarter note, we need to repeat each entry 4 times
        '''

        melody_f = dirname / "melody.txt"

        with open(melody_f) as file:
            melodies_ = [line.rstrip() for line in file]

        melody_by_16th_note = []

        for melody_ in melodies_:
            # for each pitch create an array for the required duration
            pitch, n_16th_notes = [int(i) for i in melody_.split(" ")]
            pitch_arr = np.full(shape=n_16th_notes * 4, fill_value=pitch)
            melody_by_16th_note.append(pitch_arr)

        # merge into one massive vector of shape: (n_16th_notes x 4)
        melody_by_16th_note = np.concatenate(melody_by_16th_note)

        return melody_by_16th_note



    def fetch_annotations(dirname):
        annotations = []

        annotator_idx = '1'

        structure_f = dirname / ("human_label" + annotator_idx + ".txt")

        with open(structure_f, "r") as file:
            # read the first line of the file
            structure = file.readline().rstrip()

        strlist = re.split('(?<=\D)(?=\d)|(?<=\d)(?=\D)', structure)
        expanded_structure = []

        expanded_letters = []
        expanded_cases = []

        for i in strlist:
            try:
                count = int(i)
                expanded_structure.extend([current_label] * count)

                expanded_letters.extend([current_letter] * count)
                expanded_cases.extend([current_case] * count)
            except:
                current_label = LABELS[i]

                current_letter = LETTERS[i.lower()]
                current_case = int(i.isupper())

        # make a big vector of shape: n_bars
        expanded_structure = np.array(expanded_structure)
        # We assume a 4/4 time signature because of earlier filtering
        # shape: (n_bars x n_beats_per_measure x 16)
        expanded_structure = np.repeat(expanded_structure, 4 * RESOLUTION)

        annotations.append(expanded_structure)

        # return annotations of both labellers
        return annotations

    def fetch_structure(dirname, annotations_n_ticks, start_pos):
        '''
        Given a dirname related to a music_idx, fetch all the structures relevant to this piece of music
        :arg: dirname: path to the dir where all the structural annotations live
        :arg: annotations_n_ticks: size of time dimension of pianoroll (max_len) - start_pos for how many ticks are expected from the annotations

        returns a dict of structures:
        tempo: a single int for all timesteps
        chord_ids: one int for each timestep
        chord_root_note: vector of ints for each timestep
        chord_pitches: 12-dim binary vectors for each timestep
        melody: vector of ints for each timestep
        annotations: vector of ints for each timestep
        '''

        # shape : 1
        tempo_bucket = fetch_tempo(dirname)

        # When the annotations come in, they are not the right length.
        # So, we will make sure all the annotations match the time dimension of the pianoroll

        # chord_ids.shape: chord_n_ticks
        chord_ids = fetch_chords(dirname)
        no_chord_id = get_ID(np.zeros(13))
        if len(chord_ids) < annotations_n_ticks:
            # pad the sequence, with the correct value!
            chord_ids = np.pad(array=chord_ids, pad_width=(0, annotations_n_ticks - len(chord_ids)),
                               constant_values=no_chord_id)
        else:
            # otherwise slice to shape: annotations_n_ticks
            chord_ids = chord_ids[:annotations_n_ticks]
        # pad the beginning with "no chord" if start pos is non zero
        if start_pos > 0:
            chord_ids = np.pad(array=chord_ids, pad_width=(start_pos, 0), constant_values=no_chord_id)
        # reshape to: 1 x n_ticks
        chord_ids = np.expand_dims(chord_ids, 0)

        # For the melody, we use the MIDI_note_number = 0 to signify rest notes as specified in the README for the
        # hierarchical structure analysis dataset for Pop909

        # shape: melody_n_ticks
        melody = fetch_melody(dirname)
        if len(melody) < annotations_n_ticks:
            melody = np.pad(array=melody, pad_width=(0, annotations_n_ticks - len(melody)), constant_values=0)
        else:
            # otherwise slice to shape: annotations_n_ticks
            melody = melody[:annotations_n_ticks]
        if start_pos > 0:
            melody = np.pad(array=melody, pad_width=(start_pos, 0), constant_values=0)
        # shape: 1 x n_ticks
        melody = np.expand_dims(melody, 0)

        # shape:
        # annotations: annotations_n_ticks,
        annotations = fetch_annotations(dirname)

        for annotation_idx in range(len(annotations)):
            annotation = annotations[annotation_idx]
            if len(annotation) < annotations_n_ticks:
                annotation = np.pad(array=annotation, pad_width=(0, annotations_n_ticks - len(annotation)),
                                    constant_values=LABELS["x"])
            else:
                annotation = annotation[:annotations_n_ticks]
            if start_pos > 0:
                annotation = np.pad(array=annotation, pad_width=(start_pos, 0), constant_values=LABELS["x"])
            annotation = np.expand_dims(annotation, 0)
            annotations[annotation_idx] = annotation

        # Structure APE baseline: shape: 1 x annotation_n_ticks
        total_ticks = annotations_n_ticks + start_pos
        nbars = math.floor(total_ticks / 64)
        remainder = total_ticks - (nbars * 64)
        measure_order = np.concatenate(
            (np.repeat(np.arange(nbars), 64), np.repeat(nbars, remainder))
        )
        assert len(measure_order) == total_ticks
        measure_order = np.expand_dims(measure_order, 0)

        note_order_one_bar = np.tile(np.arange(16), 4)
        note_order = np.concatenate(
            (np.tile(note_order_one_bar, nbars), note_order_one_bar[:remainder])
        )
        assert len(note_order) == total_ticks
        note_order = np.expand_dims(note_order, 0)

        # send back all the results as well as the updated chord database
        return {
            "tempo_bucket": tempo_bucket,
            "chord_ids": chord_ids,
            "melody": melody,
            "annotation_1": annotations[0],
            "measure_order": measure_order,
            "note_order": note_order
        }



    def skip_actions(musicname, music, seq_length, n_skipped):
        print("\nSkipping:  ", musicname)
        if music.get_end_time() < seq_length:
            print("\t Reason: Music is not long enough!")
        elif len(music.tracks) < 3:
            print("\t Reason: Music has ", len(music.tracks), " tracks.")
        n_skipped += 1
        return n_skipped

    def yield_samples(pianoroll, structures, note_infos, seq_length):
        # shape : dim x time

        n_ticks = pianoroll.shape[-1]

        note_times, note_pitches = note_infos
        onset_pianorolls = [np.zeros((128, n_ticks)) for _ in range(3)]
        for idx_pr in range(len(note_times)):
            note_times_pr, note_pitches_pr = note_times[idx_pr], note_pitches[idx_pr]
            mask_ = note_times_pr < n_ticks
            note_times_pr = note_times_pr[mask_]
            note_pitches_pr = note_pitches_pr[mask_]
            onset_pianorolls[idx_pr][note_pitches_pr, note_times_pr] = 1.0

        onset_pianorolls = np.concatenate(onset_pianorolls, axis=0)

        # slice up the sample into the appropriate length with overlaps
        # list of dim x n_ticks slices
        all_pianorolls_ = []
        all_onset_pianorolls_= []
        # list of dicts with the correct structural annotations
        all_structures_ = []
        idx = 0

        while (idx + seq_length) <= n_ticks:

            # generate the sliced pianoroll
            slice_pianoroll = pianoroll[:, idx:idx + pred_seq_length]
            all_pianorolls_.append(slice_pianoroll)

            slice_onset_pianoroll = onset_pianorolls[:, idx:idx + pred_seq_length]
            all_onset_pianorolls_.append(slice_onset_pianoroll)

            # generate the sliced structures
            slices_structures = {}
            for k, s in structures.items():
                # if it's key or tempo, it just stays constant throughout the sample
                if "tempo" in k:
                    slices_structures[k] = s
                # otherwise, slice it up
                else:
                    slices_structures[k] = s[:, idx:idx + pred_seq_length]
            all_structures_.append(slices_structures)

            idx = idx + pred_seq_length - overlap

        # return as a list of tuples
        return list(zip(all_pianorolls_, all_onset_pianorolls_, all_structures_))

    def save_samples(out_dir, music_idx, samples):

        # save
        os.makedirs(out_dir, exist_ok=True)

        # save each sample in a different file
        for sample_idx, sample in enumerate(samples):
            sample_basefname = music_idx + "_" + str(sample_idx)
            pianoroll, onset_pianoroll, structure = sample

            # save pianoroll as a binary array
            np.savez_compressed(out_dir / (sample_basefname + "_sample"), data=pianoroll)

            # save the numerical vectors of the structure
            np.savez_compressed(out_dir / (sample_basefname + "_struct_num"),
                                tempo_bucket=structure["tempo_bucket"],
                                chord_ids=structure["chord_ids"],
                                melody=structure["melody"],
                                annotation_1=structure["annotation_1"],
                                measure_order=structure["measure_order"],
                                note_order=structure["note_order"],
                                onset=onset_pianoroll
                                )

    print("Here we go...")

    total_samples = 0
    skipped = 0

    for start_pos_id, idx in enumerate(song_id):

        start_pos = pos[start_pos_id]
        music_idx = "{0:0=3d}".format(idx)

        #####################################################
        # # # # # # # # # # GET THE FILE NAME # # # # # # # #
        #####################################################

        mfname = (music_idx + ".mid")
        mpath = dir_ / music_idx / mfname
        if idx % 5 == 0:
            print("Processing: ", mfname)
        music = read_music(mpath)

        #####################################################
        # # # # # # # # # # # # PROCESS IT! # # # # # # # # #
        #####################################################

        # if the music is too short or does not have enough tracks, skip it and print some information
        if (music.get_end_time() < pred_seq_length) or (len(music.tracks) < 3):

            skipped = skip_actions(mfname, music, pred_seq_length, skipped)

        # otherwise, let's turn it into samples
        else:

            return_val = music_to_samples(music, mfname, start_pos)

            pianoroll, note_infos = return_val

            # if time needs to be added at the start, add zeros to the pianoroll
            if start_pos < 0:
                pianoroll = np.concatenate(
                    (
                        np.zeros(
                            (pianoroll.shape[0], np.abs(start_pos))
                        ),
                        pianoroll
                    ),
                    axis=-1
                )
            n_timesteps = pianoroll.shape[-1]
            if start_pos < 0:
                annotations_n_timesteps = int(n_timesteps)
            else:
                annotations_n_timesteps = int(n_timesteps - start_pos)
            n_beats = n_timesteps / RESOLUTION

            # fetch the structural information
            structure_dirname = struct_dir_ / music_idx
            if start_pos < 0:
                all_structures = fetch_structure(structure_dirname, annotations_n_timesteps, 0)
            else:
                all_structures = fetch_structure(structure_dirname, annotations_n_timesteps, start_pos)

            # at this point all the structural information should be of shape: n_dim x n_ticks
            # such that n_ticks for the pianoroll and structural annotations are all the same

            samples = yield_samples(pianoroll, all_structures, note_infos, pred_seq_length)

            n_samples = len(samples)
            total_samples += n_samples

            # save
            out_dir = out_ / music_idx / ("len_" + str(seq_length_multiple))
            save_samples(out_dir, music_idx, samples)

    print("Total samples extracted: ", total_samples)
    print("Skipped samples: ", skipped)

if __name__ == '__main__':
    main()
