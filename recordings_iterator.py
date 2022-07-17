import os
import wave
from collections import defaultdict

from matplotlib import pyplot as plt

directory = "datasets/one_shot_percussive_sounds"


def check_sample_rates():
    sample_rate_cnt = defaultdict(lambda: 0, dict())
    for subdirectory in os.listdir(directory):
        subdirectory = os.path.join(directory, subdirectory)
        for file in os.listdir(subdirectory):
            filename = os.fsdecode(file)
            if filename.endswith(".wav"):
                f = wave.open(os.path.join(subdirectory, filename), "rb")
                sample_rate_cnt[f.getframerate()] += 1
            else:
                continue

    print(f"Counting sample rates: {sample_rate_cnt}")


def check_sample_count():
    sample_count_cnt = defaultdict(lambda: 0, dict())
    for subdirectory in os.listdir(directory):
        subdirectory = os.path.join(directory, subdirectory)
        for file in os.listdir(subdirectory):
            filename = os.fsdecode(file)
            if filename.endswith(".wav"):
                f = wave.open(os.path.join(subdirectory, filename), "rb")
                n_frames = f.getnframes()
                sample_count_cnt[20*(n_frames//20)/f.getframerate()] += 1
            else:
                continue

    print(sample_count_cnt)
    plt.bar(list(sample_count_cnt.keys()), sample_count_cnt.values(), width=0.002)
    plt.gcf().set_dpi(600)
    plt.title("Recordings lengths histogram")
    plt.xlabel("Count")
    plt.ylabel("Duration [s]")
    print("Showing plot 'Recordings lengths histogram'")
    plt.show()


# check_sample_rates()
check_sample_count()