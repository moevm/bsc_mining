import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.fftpack import fft


c = 299792458

# chirp sequence frequency
f_chirp = 50*1e3 #Hz

# ramp frequency
f_r = 200*1e6 #Hz
T_r = 1/f_chirp # duration of one cycle
S = f_r/T_r

n_r = 150 # number of chirps
T_M = T_r*n_r

# sample settings
f_s = 50e6 #50 MHz
n_s = int(T_r*f_s)

f_0 = 77.7*1e9

# some helpful
w_0 = 2*np.pi*f_0
lambda_0 = c/f_0

def f_transmitted(t):
    return f_0 + S*(t%T_r)

r_0 = 50 # initial distance
v_veh = 36/3.6 # velocity

def get_range(t):
    return r_0+v_veh*t

def itr(t):
    r = get_range(t)
    w_itr = 2*f_0*v_veh/c + 2*S*r/c
    # we do t%T_r because the eq. above only valid within the ramp
    v = np.cos(2*np.pi*w_itr*(t%T_r) +2*r*2*np.pi*f_0/c)
    return v



t_sample = np.linspace(0, T_M, n_r*n_s)

v_sample = itr(t_sample)

plt.figure(figsize=(15,5))
plt.plot(t_sample, v_sample, "+")
plt.xlim(0, 0.1*T_r)
plt.xlabel("t [s]")
plt.title("Визуализация семплов на временном отрезке [0, 0.1 * $T_r$]");
plt.show()


table = np.zeros((n_r, n_s))

for chirp_nr in range(n_r):
    table[chirp_nr, :] = v_sample[(chirp_nr*n_s):(n_s*(chirp_nr+1))]

table_df = pd.DataFrame(data=table, 
                        columns=["sample_%03d"%i for i in range(n_s)], 
                        index=["chirp_%03d"%i for i in range(n_r)])


chirp0_samples = table_df.iloc[0].values
chirp0_magnitude = fft(chirp0_samples)

# frequencies found by FFT, will be used later
frequencies = np.arange(0, n_s//2)*f_s/n_s

def freq_to_range(f):
    return f*c/(2*S)

ranges = np.around(freq_to_range(frequencies), decimals = 1)


plt.figure(figsize=(10,5))
plt.plot(ranges, 2.0/n_s*np.abs(chirp0_magnitude[0:n_s//2]))
plt.plot(ranges, 2.0/n_s*np.abs(chirp0_magnitude[0:n_s//2]), "k+")
plt.xlabel("Расстояние $r$ [m]")
plt.title("Полученное расстояние (БПФ 1 чирпа)")
plt.show()
print(freq_to_range(frequencies)[np.argmax(2.0/n_s*np.abs(chirp0_magnitude[0:n_s//2]))]) # out of range value in peak

range_table = np.zeros((n_r, n_s//2), dtype=np.csingle)
 
for chirp_nr in range(n_r):
    chirp_ad_values = table_df.iloc[chirp_nr].values
    chirp_fft = fft(chirp_ad_values) # FFT
    range_table[chirp_nr, :] = 2.0/n_s*chirp_fft[:n_s//2]

velocity_table = np.zeros((n_r, range_table.shape[1]), dtype=np.csingle)

for r in range(range_table.shape[1]):
    range_bin_magn = range_table[:, r]
    range_bin_fft = fft(range_bin_magn)
    velocity_table[:, r]= 2.0/n_r*range_bin_fft


def angle_freq_to_velocity(w):
    return w*c/(4*np.pi*f_0)

omega_second = 2*np.pi*np.concatenate((np.arange(0, n_r//2), np.arange(-n_r//2, 0)[::1]))*f_chirp/n_r

velocities = np.around(angle_freq_to_velocity(omega_second), decimals = 1)
plt.figure(figsize=(15,10))
plt.imshow(np.abs(velocity_table))
plt.xticks(range(ranges.size)[::20], ranges[::20]);
plt.yticks(range(velocities.size)[::10], velocities[::10]);
plt.xlim([0, 200])
plt.xlabel("расстояние $r$ [m]")
plt.ylabel("скорость $\\dot r = v$ [m/s]");
plt.title("Результат двумерного БПФ последовательности чирпов - $r, \\dot r$ map")
plt.show()
