import matplotlib.pyplot as plt
import pandas as pd
from scipy.fftpack import fft
import numpy as np

class Fmcw:
    def __init__(self, x, y, v, max_v, yaw, a, omega, show_animation, ar_path, vr_path):
        self.x = float(x)
        self.y = float(y)
        self.v = float(v)
        self.a = float(a)
        self.omega = float(omega)
        self.max_v = float(max_v)
        self.yaw = np.deg2rad(float(yaw))
        self.show_animation = show_animation
        self.ar_path = ar_path
        self.vr_path = vr_path

        self.c = 299792458

        self.angle_dots = [] 
        self.dist_dots = []
        self.rad_v_dots = []

        # chirp sequence frequency
        self.f_chirp = 50*1e3 #Hz

        # ramp frequency
        self.f_r = 200*1e6 #Hz
        self.T_r = 1/self.f_chirp # duration of one cycle
        self.m_w = self.f_r/self.T_r

        self.n_r = 150 # number of chirps
        self.T_M = self.T_r*self.n_r

        self.f_s = 50e6 #50 MHz
        self.n_s = int(self.T_r*self.f_s)

        self.f_0 = 77.7*1e9

        self.d = self.c / (8 * self.f_r)

    def update(self, dt):
        self.x += self.v * np.cos(self.yaw)  * dt
        self.y += self.v * np.sin(self.yaw)*  dt
        self.yaw += self.omega * dt
        self.v += self.a * dt
        if self.v >= self.max_v:
            self.v = self.max_v

    def get_range(self, t, r, v):
        return r+v*t

    def itr(self, t, ri, vi):
        r = self.get_range(t, ri, vi)
        w_itr = 2*self.f_0*vi/self.c + 2*self.m_w*r/self.c
        # we do t%T_r because the eq. above only valid within the ramp
        v = np.cos(2*np.pi*w_itr*(t%self.T_r) + 2*r*2*np.pi*self.f_0/self.c)
        return v

    def sum_v(self, chirp_nr):
        v = 0
        for i in range(len(self.dist_dots)):
            v += self.arr_v_sample[i][(chirp_nr*self.n_s):(self.n_s*(chirp_nr+1))]
        return v

    def sum_v_for_a(self, indexes):
        v = 0
        for i in range(0, len(indexes)):
            v += self.arr_v_sample[i][0:self.n_s]
        return v

    def freq_to_range(self, f):
        return f*self.c/(2*self.m_w)

    def angle_freq_to_velocity(self, w):
        return w*self.c/(4*np.pi*self.f_0)

    def find_angle_range_map(self, time):
        angles = np.arange(-80, 81, 1)
        angle_table = []
        for a in angles:
            indexes = [i for i,x in enumerate(self.angle_dots) if x == a]
            
            if len(indexes) != 0:
                t_sample = np.linspace(0, self.T_M, self.n_r*self.n_s)
                self.arr_v_sample = []
                for i in indexes:
                    self.arr_v_sample.append(self.itr(t_sample, self.dist_dots[i], self.rad_v_dots[i]))

                samples = self.sum_v_for_a(indexes)

                chirp_magnitude = fft(samples)

                angle_table.append(2.0/self.n_s*np.abs(chirp_magnitude[0:self.n_s//2]))
            else:
                angle_table.append([0.0] * (self.n_s//2))

        frequencies = np.arange(0, self.n_s//2)*self.f_s/self.n_s
        ranges = np.around(self.freq_to_range(frequencies), decimals = 1)

        table_a = pd.DataFrame(data=angle_table, 
                                columns=["r = %.2f"%i for i in ranges], 
                                index=["a = %.2f"%i for i in angles])
                                
        table_a.to_csv(self.ar_path + str(round(time, 1)) + 'sec.csv')

        if self.show_animation:
            plt.figure(3)
            plt.cla()
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            plt.imshow(angle_table)
            plt.xticks(range(ranges.size)[::20], ranges[::20], rotation=90);
            plt.yticks(range(angles.size)[::10], angles[::10]);
            plt.xlim([0, 200])
            plt.xlabel("расстояние $r$ [m]")
            plt.ylabel("азимут $θ$ [°]");
            plt.title("Карта азимута-расстояния объектов")

    def find_velocity_range_map(self, time):
        t_sample = np.linspace(0, self.T_M, self.n_r*self.n_s)
        self.arr_v_sample = []
        for i in range(len(self.dist_dots)):
            self.arr_v_sample.append(self.itr(t_sample, self.dist_dots[i], self.rad_v_dots[i]))

        table = np.zeros((self.n_r, self.n_s))

        for chirp_nr in range(self.n_r):
            table[chirp_nr, :] = self.sum_v(chirp_nr)

        table_df = pd.DataFrame(data=table, 
                                columns=["sample_%03d"%i for i in range(self.n_s)], 
                                index=["chirp_%03d"%i for i in range(self.n_r)])

        table_df.head(10)

        frequencies = np.arange(0, self.n_s//2)*self.f_s/self.n_s

        ranges = np.around(self.freq_to_range(frequencies), decimals = 1)

        range_table = np.zeros((self.n_r, self.n_s//2), dtype=np.csingle)

        for chirp_nr in range(self.n_r):
            chirp_ad_values = table_df.iloc[chirp_nr].values
            chirp_fft = fft(chirp_ad_values) # FFT
            range_table[chirp_nr, :] = 2.0/self.n_s*chirp_fft[:self.n_s//2]

        velocity_table = np.zeros((self.n_r, range_table.shape[1]), dtype=np.csingle)
        for r in range(range_table.shape[1]):
            range_bin_magn = range_table[:, r]
            range_bin_fft = fft(range_bin_magn)
            velocity_table[:, r]= 2.0/self.n_r*range_bin_fft

        omega_second = 2*np.pi*np.concatenate((np.arange(0, self.n_r//2), np.arange(-self.n_r//2, 0)[::1]))*self.f_chirp/self.n_r
        velocities = np.around(self.angle_freq_to_velocity(omega_second), decimals = 1)
        
        table_v = pd.DataFrame(data=np.abs(velocity_table), 
                                columns=["r = %.2f"%i for i in ranges], 
                                index=["v = %.2f"%i for i in velocities])

        table_v.to_csv(self.vr_path + str(round(time, 1)) + 'sec.csv')

        if self.show_animation:
            plt.figure(2)
            plt.cla() 
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            plt.imshow(np.abs(velocity_table), cmap = plt.get_cmap('RdYlBu'))
            plt.xticks(range(ranges.size)[::20], ranges[::20], rotation=90)
            plt.yticks(range(velocities.size)[::10], velocities[::10])
            plt.xlim([0, 200])
            plt.xlabel("расстояние $r$ [m]")
            plt.ylabel("скорость $v$ [m/s]");
            plt.title("Карта скорости-расстояния объектов")
