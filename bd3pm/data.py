import os

import mido
import numpy as np
import torch
from mido import Message, MidiFile, MidiTrack
from torch.utils.data import Dataset

BINS_PER_BEAT = 4
BEATS_PER_BAR = 4
NUM_BARS = 8
PITCH_MIN = 21
NUM_PITCHES = 88
SEQ_LEN = BINS_PER_BEAT * BEATS_PER_BAR * NUM_BARS

KEY_SHIFTS = {
    'C': 0, 'Db': -1, 'D': -2, 'Eb': -3, 'E': -4, 'F': -5, 'F#': -6, 'Gb': -6, 'G': 5, 'Ab': 4, 'A': 3,
    'Bb': 2, 'B': 1, 'Cm': 0, 'C#m': -1, 'Dm': -2, 'D#m': -3, 'Ebm': -3, 'Em': -4, 'Fm': -5, 'F#m': -6,
    'Gbm': -6, 'Gm': 5, 'G#m': 4, 'Abm': 4, 'Am': 3, 'Bbm': 2, 'Bm': 1,
}


def downbeat_steps(mid):
    tpb = mid.ticks_per_beat
    time_sigs = []
    for track in mid.tracks:
        tick = 0
        for msg in track:
            tick += msg.time
            if msg.type == 'time_signature':
                time_sigs.append((tick, msg.numerator, msg.denominator))
    time_sigs.sort(key=lambda x: x[0])
    max_tick = max([sum(msg.time for msg in tr) for tr in mid.tracks] + [0])
    ticks, tick, idx, num, den = [], 0, 0, 4, 4
    while tick < max_tick:
        if idx < len(time_sigs) and tick >= time_sigs[idx][0]:
            _, num, den = time_sigs[idx]
            idx += 1
        ticks.append(tick)
        ticks_per_bar = num * tpb * (4 / den)
        tick += int(ticks_per_bar)
        if ticks_per_bar <= 0:
            break
    return sorted(set(int((t / tpb) * BINS_PER_BEAT) for t in ticks))


def midi_to_roll(path):
    mid = MidiFile(path)
    key_events = []
    for track in mid.tracks:
        tick = 0
        for msg in track:
            tick += msg.time
            if msg.type == 'key_signature':
                key_events.append((tick, KEY_SHIFTS.get(msg.key, 0)))
    key_events.sort(key=lambda x: x[0])

    def shift_at(tick):
        s = 0
        for t, k in key_events:
            if t <= tick:
                s = k
            else:
                break
        return s

    ticks_per_bin = mid.ticks_per_beat / BINS_PER_BEAT
    note_track = mid.tracks[1] if len(mid.tracks) > 1 else mid.tracks[0]
    total_ticks = sum(msg.time for msg in note_track)
    roll = np.zeros((int(round(total_ticks / ticks_per_bin)) + 1, NUM_PITCHES, 2), dtype=np.uint8)

    def close(start, pitch_idx, end):
        if end > start:
            roll[start, pitch_idx, 0] = 1
            if end - start > 1:
                roll[start + 1:end, pitch_idx, 1] = 1
        elif end == start:
            roll[start, pitch_idx, 0] = 1

    tick, active = 0, {}
    for msg in note_track:
        tick += msg.time
        b = int(round(tick / ticks_per_bin))
        if msg.type == 'note_on' and msg.velocity > 0:
            pitch_idx = msg.note + shift_at(tick) - PITCH_MIN
            if 0 <= pitch_idx < NUM_PITCHES:
                if msg.note in active:
                    close(active[msg.note][0], active[msg.note][1], b)
                active[msg.note] = (b, pitch_idx)
        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
            if msg.note in active:
                close(active[msg.note][0], active[msg.note][1], b)
                del active[msg.note]
    return roll


def roll_to_midi(roll, path, min_duration=1):
    tpb = 480
    scale = tpb / BINS_PER_BEAT
    mid = MidiFile(ticks_per_beat=tpb)
    track = MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(120), time=0))
    events, active = [], {}
    for tick, pitch_idx in zip(*np.nonzero(roll[:, :, 0])):
        pitch = int(pitch_idx + PITCH_MIN)
        if pitch in active:
            events.append((int(tick * scale), 'note_off', pitch, 0))
        events.append((int(tick * scale), 'note_on', pitch, 80))
        active[pitch] = int(tick)
        end = int(tick) + 1
        while end < roll.shape[0] and roll[end, pitch_idx, 0] != 1 and roll[end, pitch_idx, 1] != 0:
            end += 1
        if end - tick < min_duration:
            continue
        events.append((int(end * scale), 'note_off', pitch, 0))
        active.pop(pitch, None)
    events.sort(key=lambda e: (e[0], 0 if e[1] == 'note_off' else 1))
    now = 0
    for time, kind, pitch, vel in events:
        track.append(Message(kind, note=pitch, velocity=vel, time=max(time - now, 0)))
        now = time
    mid.save(path)


def song_windows(path):
    steps_per_bar = BINS_PER_BEAT * BEATS_PER_BAR
    roll = midi_to_roll(path)
    out = []
    for start in downbeat_steps(MidiFile(path)):
        end = start + SEQ_LEN
        if end > roll.shape[0]:
            if start >= roll.shape[0]:
                continue
            chunk = np.zeros((SEQ_LEN, NUM_PITCHES, 2), dtype=np.bool_)
            chunk[:roll.shape[0] - start] = roll[start:]
        else:
            chunk = roll[start:end]
        if np.sum(chunk) > 10 and np.sum(chunk[:steps_per_bar]) and np.sum(chunk[-steps_per_bar:]):
            out.append(np.asarray(chunk, dtype=np.bool_))
    return out


def roll_from_midi(path, seq_len=SEQ_LEN):
    roll = midi_to_roll(path)
    active = np.logical_or(roll[..., 0] > 0.5, roll[..., 1] > 0.5).astype(np.float32)
    if active.shape[0] < seq_len:
        active = np.concatenate([active, np.zeros((seq_len - active.shape[0], NUM_PITCHES), np.float32)])
    return active[:seq_len]


class PianoRollDataset(Dataset):
    def __init__(self, path, augment=False):
        with np.load(path) as data:
            self.data = data['samples']
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = torch.from_numpy(self.data[idx]).float()
        if self.augment:
            pitches = torch.where(sample[..., 0] > 0.5)[1]
            if len(pitches) > 0:
                lo, hi = max(-5, -pitches.min().item()), min(6, 87 - pitches.max().item())
                if lo <= hi:
                    shift = torch.randint(lo, hi + 1, (1,)).item()
                    if shift != 0:
                        shifted = torch.zeros_like(sample)
                        if shift > 0:
                            shifted[:, shift:, :] = sample[:, :-shift, :]
                        else:
                            shifted[:, :shift, :] = sample[:, -shift:, :]
                        sample = shifted
        return sample.view(sample.shape[0], -1)
