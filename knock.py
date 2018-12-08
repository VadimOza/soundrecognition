from scipy.io.wavfile import read

fs, data = read('knock.wav')

data_size = len(data)

min_val = 5000

focus_size = int(0.15 * fs)
focuses = []
distances = []
idx = 0
while idx < len(data):
    if data[idx] > min_val:
        mean_idx = idx + focus_size // 2
        focuses.append(float(mean_idx) / data_size)
        if len(focuses) > 1:
            last_focus = focuses[-2]
            actual_focus = focuses[-1]
            distances.append(actual_focus - last_focus)
        idx += focus_size
    else:
        idx += 1

print(focuses)
print(distances)
