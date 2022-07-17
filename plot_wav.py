import wave

import numpy as np
from matplotlib import pyplot as plt
from pyaudio import PyAudio

FILENAME = r"datasets\one_shot_percussive_sounds\1\488.wav"
backslash_char = "\\"

chunk = 16000

f = wave.open(FILENAME, "rb")
p = PyAudio()
stream = p.open(format=p.get_format_from_width(f.getsampwidth()),
                channels=f.getnchannels(),
                rate=f.getframerate(),
                output=True)
# read data
# data = f.readframes(chunk)
data = np.frombuffer(f.readframes(np.inf), dtype=np.int16)


def zero_pad_data(data):
    return np.pad(data, (chunk//20, chunk - data.size), 'constant')


data = zero_pad_data(data)
stream.write(data)
# signal = np.fromstring(string=data, dtype=float)

# stop stream
stream.stop_stream()
stream.close()

# close PyAudio
p.terminate()

plt.figure(1)

plot_a = plt.subplot(211)
plot_a.plot(data / max(abs(data.min()), abs(data.max())), lw=0.5, c='k')
plot_a.set_ylabel('Energy')
plot_a.set_xlim((0, chunk))
plot_a.set_ylim((-1., 1.))

plot_b = plt.subplot(212)
plot_b.specgram(data, NFFT=2 ** 10, Fs=f.getframerate(), noverlap=(2 ** 10) - 1, scale='dB')
plot_b.set_xlabel('Time')
plot_b.set_ylabel('Frequency')
plot_b.set_xlim((0., 1.))
plot_b.set_ylim((0, f.getframerate() / 20))

plt.suptitle(f"Plots for {FILENAME.split(backslash_char)[-1]}")
plt.gcf().set_dpi(300)
plt.show()
