# BUG 1024
import os
import sys
import wave
import time
import ui_cn_1
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from scipy import signal
from PyQt5.QtCore import *
from scipy.io import wavfile
from PyQt5.QtWidgets import *
from PyQt5.QtMultimedia import *
from QCandyUi import CandyWindow
from PyQt5.QtGui import QTextCursor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QComboBox, QCheckBox

matplotlib.rcParams['font.family'] = 'STSong'
matplotlib.use('Qt5Agg')

# 处理函数
class ProcessFunction(object):
    # 归一化幅度
    def Audio_TimeDomain(self, feature):  # 时域
        f = wave.open(feature.path, "rb")  # 打开音频文件
        params = f.getparams()
        nchannels, sampwidth, framerate, nframes = params[:4]
        # nchannels通道数
        # sampwidth量化位数
        # framerate采样频率
        # nframes采样点数
        str_data = f.readframes(nframes)  # 返回Bytes对象表示的采样点数设置的音频
        f.close()  # 关闭音频文件
        # 将字符串转换为数组，得到一维的short类型的数组
        wave_data = np.fromstring(str_data, dtype=np.short)  # dtype = s
        # 赋值的归一化
        wave_data = wave_data * 1.0 / (max(abs(wave_data)))
        # 整合左声道和右声道的数据
        # numpy.reshape(a, newshape, order='C') [采样点数, 通道数] 返回值为 *采样点数* 维的数组，每组有 *通道数* 个
        wave_data = np.reshape(wave_data, [nframes, 1])
        # 最后通过采样点数和取样频率计算出每个取样的时间
        time = np.arange(0, nframes) * (1.0 / framerate)
        # 输出音频信息
        feature.textBrowser_3.append("音频信息: " + str(nchannels))
        feature.textBrowser_3.append("通道数: " + str(nchannels))
        feature.textBrowser_3.append("采样频率: " + str(framerate) + " Hz")
        feature.textBrowser_3.append("采样点数: " + str(nframes))
        feature.textBrowser_3.append("采样时间: " + str(nframes / framerate) + " seconds")



        # 进度条设置10
        feature.progressBar.setValue(10)
        feature.progressBar_2.setValue(10)

        # 设置画布
        ax = feature.fig1.add_subplot(111)
        # 调整图像大小
        ax.cla()
        # 画出归一化幅度图像
        ax.plot(time, wave_data[:, 0], color='#dc143c')
        ax.set_title('归一化幅度', fontsize=10)
        ax.set_xlabel('时间[sec]', fontsize=10)

        feature.fig1.subplots_adjust(left=0.1, bottom=0.215, right=None, top=None, wspace=None, hspace=None)
        feature.canvas1.draw()


        #取一段时间进行分析

        # 设置画布
        bx = feature.fig8.add_subplot(111)
        # 调整图像大小
        bx.cla()
        # 画出归一化幅度图像
        StartTime0 = np.int(feature.StartTime)
        DurationTime0 = np.int(feature.DurationTime)
        StartTime1 = np.int((framerate / 1000) * StartTime0)
        DurationTime1 = np.int((StartTime1 + DurationTime0 * (framerate / 1000)))
        # 传入数组的变量必须是整型，而乘除法返回的是浮点，用int()转换
        # print(StartTime1)
        # print(DurationTime1)
        # 实现20ms的幅度显示
        bx.plot(time[StartTime1:DurationTime1], wave_data[StartTime1:DurationTime1, 0], color='#5014DC')
        bx.set_title('归一化幅度', fontsize=10)
        bx.set_xlabel('时间[sec]', fontsize=10)

        feature.fig8.subplots_adjust(left=0.1, bottom=0.215, right=None, top=None, wspace=None, hspace=None)
        feature.canvas8.draw()

        # 进度条设置20
        feature.progressBar.setValue(20)
        feature.progressBar_2.setValue(20)

    # 频域，FFT
    def Audio_FrequencyDomain(self, feature):
        # STFT图像绘制
        sampling_freq, audio = wavfile.read(feature.path)  # 返回采样频率和一维数组，对应8-bit integer PCM编码
        T = 20  # 短时傅里叶变换的时长 单位 ms
        fs = sampling_freq  # 采样频率
        N = len(audio)  # 采样点的个数
        audio = audio * 1.0 / (max(abs(audio))) # 归一化

        # 计算并绘制STFT的大小
        # nperseg ：int, optional
        # 段长. 默认256 .
        # f：ndarray
        # Array of sample frequencies.
        # t：ndarray
        # Array of segment times.
        # Zxx：ndarray
        # x的STFT. 一般情况下,Zxx 的最后一个轴对应段落时间 .
        f, t, Zxx = signal.stft(audio, fs, nperseg=np.int(T * fs / 1000))
        # 设置进度条30
        feature.progressBar.setValue(30)
        feature.progressBar_2.setValue(30)

        ax = feature.fig5.add_subplot(111)
        feature.fig5.subplots_adjust(left=None, bottom=0.215, right=None, top=None, wspace=None, hspace=None)
        ax.cla()
        ax.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=0.1)  # 明显划分出边缘，表现为颜色深浅
        ax.set_title('短时傅里叶变换STFT', fontsize=10)
        ax.set_xlabel('时间 [sec]', fontsize=10)
        ax.set_ylabel('频率 [Hz]', fontsize=10)
        feature.canvas5.draw()

        feature.progressBar.setValue(40)
        feature.progressBar_2.setValue(40)

        # 绘制FFT图像
        fft_signal = np.fft.fft(audio)
        fft_signal = abs(fft_signal)  # 取绝对值
        # 建立频率轴
        # 复信号没有负频率，以fs为采样速率的信号，fft的频谱结果是从[0,fs]的。
        fft_signal = np.fft.fftshift(fft_signal)  # 移动0频率点，画图
        # 与上同理
        fft_signal = fft_signal[int(fft_signal.shape[0] / 2):]
        # 频率轴的间隔
        freqInteral = (sampling_freq / len(fft_signal))

        Freq = np.arange(0, sampling_freq / 2, sampling_freq / (2 * len(fft_signal)))
        # 设置进度条50
        feature.progressBar.setValue(50)
        feature.progressBar_2.setValue(50)
        # 计算最高频率
        highFreq = (np.argmax(fft_signal[int(len(fft_signal) / 2):len(fft_signal)])) * freqInteral
        feature.textBrowser_3.append("FFT : 最高频率为: " + str(highFreq))

        ax = feature.fig3.add_subplot(111)
        # 调整图像大小
        ax.cla()
        ax.plot(Freq, fft_signal, color='#dc143c')
        ax.set_title('FFT 图像', fontsize=10)
        ax.set_xlabel('频率 [Hz]', fontsize=10)
        ax.set_ylabel('增益', fontsize=10)
        feature.fig3.subplots_adjust(left=None, bottom=0.215, right=None, top=None, wspace=None, hspace=None)
        feature.canvas3.draw()

        cx = feature.fig10.add_subplot(111)
        # 调整图像大小
        cx.cla()
        cx.plot(Freq, fft_signal, color='#5014DC')
        cx.set_title('FFT 图像', fontsize=10)
        cx.set_xlabel('频率 [Hz]', fontsize=10)
        cx.set_ylabel('强度 [dB]', fontsize=10)
        feature.fig10.subplots_adjust(left=None, bottom=0.215, right=None, top=None, wspace=None, hspace=None)
        feature.canvas10.draw()

        # 设置进度条60
        feature.progressBar.setValue(60)
        feature.progressBar_2.setValue(60)

    # 语谱图
    def Audio_SpectrogramDomain(self, feature):
        # 语谱图绘制
        f = wave.open(feature.path, "rb")
        params = f.getparams()
        nchannels, sampwidth, framerate, nframes = params[:4]
        strData = f.readframes(nframes)  # 读取音频，字符串格式
        waveData = np.fromstring(strData, dtype=np.int16)  # 将字符串转化为int
        waveData = waveData * 1.0 / (max(abs(waveData)))  # wave幅值归一化
        waveData = np.reshape(waveData, [nframes, 1]).T  # 合并多声道，.T:数组转置，横坐标为时间 ，变成一维数组，下面会用
        f.close()
        feature.progressBar.setValue(75)
        feature.progressBar_2.setValue(75)

        ax = feature.fig6.add_subplot(111)
        feature.fig6.subplots_adjust(left=None, bottom=0.215, right=None, top=None, wspace=None, hspace=None)
        ax.cla()
        #  matplotlib.pyplot.specgram(x信号，一维数组, Fs=None采样频率, sides=None返回单边频谱, scale_by_freq=None 按密度缩放频率)
        ax.specgram(waveData[0], Fs=framerate, scale_by_freq=True, sides='default')
        ax.set_title('语谱图', fontsize=10)
        ax.set_xlabel('时间 [sec]', fontsize=10)
        ax.set_ylabel('频率 [Hz]', fontsize=10)
        feature.canvas6.draw()
        feature.progressBar.setValue(85)
        feature.progressBar_2.setValue(85)

    # 设计IIR滤波器
    def IIR_Designer(self, feature):
        if str(feature.iirType) == 'Butterworth':  # 巴特沃斯 双线性变换法间接设计模拟滤波器
            fs = float(feature.fs)
            feature.textBrowser_3.append(str(feature.filterType))
            feature.textBrowser_3.append(str(feature.iirType))
            # 对于带通带阻分开讨论
            if str(feature.filterType) == "Bandpass" or str(feature.filterType) == "bandstop":

                # 频率预畸
                wp = str(feature.An_wp).split()  # 切分开之后再转换为array
                wp0 = float(wp[0]) * (2 * np.pi / fs)
                wp1 = float(wp[1]) * (2 * np.pi / fs)

                # 双线性变换
                wp[0] = (2 * fs) * np.tan(wp0 / 2)
                wp[1] = (2 * fs) * np.tan(wp1 / 2)

                omega_p = [float(wp[0]), float(wp[1])]
                wst = str(feature.An_wst).split()  # 切分开之后再转换为序列
                wst0 = float(wst[0]) * (2 * np.pi / fs)
                wst1 = float(wst[1]) * (2 * np.pi / fs)

                wst[0] = (2 * fs) * np.tan(wst0 / 2)  # 双线性变换
                wst[1] = (2 * fs) * np.tan(wst1 / 2)

                omega_st = [wst[0], wst[1]]
            else:
                wp = float(feature.An_wp) * (2 * np.pi / fs)
                wst = float(feature.An_wst) * (2 * np.pi / fs)
                # 频率预畸
                omega_p = (2 * fs) * np.tan(wp / 2)
                omega_st = (2 * fs) * np.tan(wst / 2)
            # 改为浮点型输入
            feature.Rp = float(feature.Rp)
            feature.As = float(feature.As)

            # 设计阶数
            N, Wn = signal.buttord(omega_p, omega_st, feature.Rp, feature.As, True)
            # buttord模拟滤波器设计（连续时间线性时不变系统LTI）
            # butter 返回值b,a：分子num分母den
            feature.filts = signal.lti(*signal.butter(N, Wn, btype=str(feature.filterType), analog=True))
            # bilinear 返回 ：变换后的数字滤波器 *传递函数* 的分母 变换后的数字滤波器 *传递函数* 的分子
            # 这里是一个函数式的分子分母，类型为N维数组
            feature.filtz = signal.lti(*signal.bilinear(feature.filts.num, feature.filts.den, fs))
            # 经过lti线性时不变系统，得到零、极点，在分子分母，一维数组
            feature.z, feature.p = signal.bilinear(feature.filts.num, feature.filts.den, fs)
            # 求频率响应
            wz, hz = signal.freqz(feature.filtz.num, feature.filtz.den)

            ax = feature.fig2.add_subplot(111)
            ax.cla()
            # x轴对数表示，画滤波器图
            ax.semilogx(wz * fs / (2 * np.pi), 20 * np.log10(np.abs(hz).clip(1e-15)))
            ax.set_xlabel('Hz')
            ax.set_ylabel('dB')
            ax.set_title('Butterworth')
            feature.fig2.subplots_adjust(left=None, bottom=0.2, right=None, top=None, wspace=None, hspace=None)
            feature.canvas2.draw()

            # 绘制零极点图
            ax = feature.fig4.add_subplot(111)
            ax.cla()
            z1, p1, k1 = signal.tf2zpk(feature.z, feature.p)  # 零点、极点、增益
            c = np.vstack((feature.p, feature.z))
            Max = (abs(c)).max()  # 获取最大值
            a = feature.p / Max  # 归一化
            b = feature.z / Max
            # 在算法仿真中，由于是全精度的计算，曲线往往比较理想。但在把算法写入硬件时，由于资源限制，必须要进行量化
            Ra = (a * (2 ** ((N - 1) - 1))).astype(int)  # 量化切割
            Rb = (b * (2 ** ((N - 1) - 1))).astype(int)
            z2, p2, k2 = signal.tf2zpk(Rb, Ra)
            # 参数方程画圆
            theta = np.arange(0, 2 * np.pi, 0.01)
            x = np.cos(theta)
            y = np.sin(theta)
            ax.plot(x, y, color='black')  # 画圆
            for i in p1:
                ax.plot(np.real(i), np.imag(i), 'bx')  # 量化前的极点，蓝色叉
            for i in z1:
                ax.plot(np.real(i), np.imag(i), 'bo')  # 量化前零点
            for i in p2:
                ax.plot(np.real(i), np.imag(i), 'rx')  # 量化后极点，红色叉
            for i in z2:
                ax.plot(np.real(i), np.imag(i), 'ro')  # 量化后零点
            ax.set_xlim(-1.8, 1.8)
            ax.set_ylim(-1.2, 1.2)
            ax.grid()
            ax.set_title("%d bit 量化" % N)  # N为量化位数，N越多，误差越小。
            feature.fig4.subplots_adjust(left=None, bottom=0.2, right=None, top=None, wspace=None, hspace=None)
            feature.canvas4.draw()

        if str(feature.iirType) == 'Chebyshev I':  # 切比雪夫一型

            fs = float(feature.fs)
            feature.textBrowser_3.append(str(feature.filterType))
            feature.textBrowser_3.append(str(feature.iirType))
            if str(feature.filterType) == "Bandpass" or str(feature.filterType) == "bandstop":
                # 如果是带通带阻需要输入四组频率数据
                # 频率预畸
                wp = str(feature.An_wp).split()  # 切分开之后再转换为array
                wp0 = float(wp[0]) * (2 * np.pi / fs)
                wp1 = float(wp[1]) * (2 * np.pi / fs)

                # 双线性变换
                wp[0] = (2 * fs) * np.tan(wp0 / 2)
                wp[1] = (2 * fs) * np.tan(wp1 / 2)

                omega_p = [float(wp[0]), float(wp[1])]
                wst = str(feature.An_wst).split()  # 切分开之后再转换为array
                wst0 = float(wst[0]) * (2 * np.pi / fs)
                wst1 = float(wst[1]) * (2 * np.pi / fs)

                # 双线性变换
                wst[0] = (2 * fs) * np.tan(wst0 / 2)
                wst[1] = (2 * fs) * np.tan(wst1 / 2)

                omega_st = [wst[0], wst[1]]
            else:
                wp = float(feature.An_wp) * (2 * np.pi / fs)
                wst = float(feature.An_wst) * (2 * np.pi / fs)
                # 频率预畸
                omega_p = (2 * fs) * np.tan(wp / 2)
                omega_st = (2 * fs) * np.tan(wst / 2)

            if len(str(feature.Rp).split()) > 1:  # 纹波参数
                Rpinput = str(feature.Rp).split()
                feature.Rp = float(Rpinput[0])
                feature.As = float(feature.As)
                rp_in = float(Rpinput[1])
            else:
                feature.Rp = float(feature.Rp)
                feature.As = float(feature.As)
                rp_in = 0.1 * feature.Rp

            # N, Wn = signal.cheb1ord(wp, wst, feature.Rp, feature.As, True)
            N, Wn = signal.cheb1ord(omega_p, omega_st, feature.Rp, feature.As, True)
            feature.filts = signal.lti(*signal.cheby1(N, rp_in, Wn, btype=str(feature.filterType),
                                                      analog=True))  # 切比雪夫是还有一个纹波参数
            feature.filtz = signal.lti(*signal.bilinear(feature.filts.num, feature.filts.den, fs))

            feature.z, feature.p = signal.bilinear(feature.filts.num, feature.filts.den, fs)

            wz, hz = signal.freqz(feature.filtz.num, feature.filtz.den)

            ax = feature.fig2.add_subplot(111)
            ax.cla()
            ax.semilogx(wz * fs / (2 * np.pi), 20 * np.log10(np.abs(hz).clip(1e-15)))
            ax.set_xlabel(' Hz')
            ax.set_ylabel(' dB')
            ax.set_title('切比雪夫I型')
            feature.fig2.subplots_adjust(left=None, bottom=0.2, right=None, top=None, wspace=None, hspace=None)
            feature.canvas2.draw()

            # 绘制零极点图
            ax = feature.fig4.add_subplot(111)
            ax.cla()  # 删除原图
            z1, p1, k1 = signal.tf2zpk(feature.z, feature.p)  # zero, pole and gain
            c = np.vstack((feature.p, feature.z))
            Max = (abs(c)).max()  # find the largest value
            a = feature.p / Max  # normalization
            b = feature.z / Max
            Ra = (a * (2 ** ((N - 1) - 1))).astype(int)  # quantizan and truncate
            Rb = (b * (2 ** ((N - 1) - 1))).astype(int)
            z2, p2, k2 = signal.tf2zpk(Rb, Ra)
            # 参数方程画圆
            theta = np.arange(0, 2 * np.pi, 0.01)
            x = np.cos(theta)
            y = np.sin(theta)
            ax.plot(x, y, color='black')
            for i in p1:
                ax.plot(np.real(i), np.imag(i), 'bx')  # pole before quantization
            for i in z1:
                ax.plot(np.real(i), np.imag(i), 'bo')  # zero before quantization
            for i in p2:
                ax.plot(np.real(i), np.imag(i), 'rx')  # pole after quantization
            for i in z2:
                ax.plot(np.real(i), np.imag(i), 'ro')  # zero after quantization
            ax.set_xlim(-1.8, 1.8)
            ax.set_ylim(-1.2, 1.2)
            ax.grid()
            ax.set_title("%d bit 量化" % N)
            feature.fig4.subplots_adjust(left=None, bottom=0.2, right=None, top=None, wspace=None, hspace=None)
            feature.canvas4.draw()

        if str(feature.iirType) == 'Chebyshev II':  # 切比雪夫二型

            fs = float(feature.fs)
            feature.textBrowser_3.append(str(feature.filterType))
            feature.textBrowser_3.append(str(feature.iirType))
            if str(feature.filterType) == "Bandpass" or str(feature.filterType) == "bandstop":
                # 如果是带通带阻需要输入四组频率数据
                # 频率预畸
                wp = str(feature.An_wp).split()  # 切分开之后再转换为array
                wp0 = float(wp[0]) * (2 * np.pi / fs)
                wp1 = float(wp[1]) * (2 * np.pi / fs)

                # 双线性变换
                wp[0] = (2 * fs) * np.tan(wp0 / 2)
                wp[1] = (2 * fs) * np.tan(wp1 / 2)

                omega_p = [float(wp[0]), float(wp[1])]
                wst = str(feature.An_wst).split()  # 切分开之后再转换为array
                wst0 = float(wst[0]) * (2 * np.pi / fs)
                wst1 = float(wst[1]) * (2 * np.pi / fs)
                # 双线性变换

                wst[0] = (2 * fs) * np.tan(wst0 / 2)
                wst[1] = (2 * fs) * np.tan(wst1 / 2)

                omega_st = [wst[0], wst[1]]
            else:
                wp = float(feature.An_wp) * (2 * np.pi / fs)
                wst = float(feature.An_wst) * (2 * np.pi / fs)
                # 频率预畸
                omega_p = (2 * fs) * np.tan(wp / 2)
                omega_st = (2 * fs) * np.tan(wst / 2)
            if len(str(feature.As).split()) > 1:  # 纹波参数
                Asinput = str(feature.As).split()
                feature.As = float(Asinput[0])
                feature.Rp = float(feature.Rp)
                rs_in = float(Asinput[1])
            else:
                feature.Rp = float(feature.Rp)
                feature.As = float(feature.As)
                rs_in = 0.1 * feature.As
            N, Wn = signal.cheb2ord(omega_p, omega_st, feature.Rp, feature.As, True)
            feature.filts = signal.lti(*signal.cheby2(N, rs_in, Wn, btype=str(feature.filterType),
                                                      analog=True))  # 切比雪夫是还有一个纹波参数
            feature.filtz = signal.lti(*signal.bilinear(feature.filts.num, feature.filts.den, fs))

            feature.z, feature.p = signal.bilinear(feature.filts.num, feature.filts.den, fs)

            wz, hz = signal.freqz(feature.filtz.num, feature.filtz.den)

            ax = feature.fig2.add_subplot(111)
            ax.cla()
            ax.semilogx(wz * fs / (2 * np.pi), 20 * np.log10(np.abs(hz).clip(1e-15)))
            ax.set_xlabel('Hz')
            ax.set_ylabel('dB')
            ax.set_title('切比雪夫II型')
            feature.fig2.subplots_adjust(left=None, bottom=0.2, right=None, top=None, wspace=None, hspace=None)
            feature.canvas2.draw()

            # 绘制零极点图
            ax = feature.fig4.add_subplot(111)
            ax.cla()
            z1, p1, k1 = signal.tf2zpk(feature.z, feature.p)  # zero, pole and gain
            c = np.vstack((feature.p, feature.z))
            Max = (abs(c)).max()  # find the largest value
            a = feature.p / Max  # normalization
            b = feature.z / Max
            Ra = (a * (2 ** ((N - 1) - 1))).astype(int)  # quantizan and truncate
            Rb = (b * (2 ** ((N - 1) - 1))).astype(int)
            z2, p2, k2 = signal.tf2zpk(Rb, Ra)
            # 参数方程画圆
            theta = np.arange(0, 2 * np.pi, 0.01)
            x = np.cos(theta)
            y = np.sin(theta)
            ax.plot(x, y, color='black')
            for i in p1:
                ax.plot(np.real(i), np.imag(i), 'bx')  # pole before quantization
            for i in z1:
                ax.plot(np.real(i), np.imag(i), 'bo')  # zero before quantization
            for i in p2:
                ax.plot(np.real(i), np.imag(i), 'rx')  # pole after quantization
            for i in z2:
                ax.plot(np.real(i), np.imag(i), 'ro')  # zero after quantization
            ax.set_xlim(-1.8, 1.8)
            ax.set_ylim(-1.2, 1.2)
            ax.grid()
            ax.set_title("%d bit 量化" % N)
            feature.fig4.subplots_adjust(left=None, bottom=0.2, right=None, top=None, wspace=None, hspace=None)
            feature.canvas4.draw()

        if str(feature.iirType) == 'Cauer/elliptic':
            fs = float(feature.fs)
            feature.textBrowser_3.append(str(feature.filterType))
            feature.textBrowser_3.append(str(feature.iirType))
            if str(feature.filterType) == "Bandpass" or str(feature.filterType) == "bandstop":
                # 如果是带通带阻需要输入四组频率数据
                # 频率预畸
                wp = str(feature.An_wp).split()  # 切分开之后再转换为array
                wp0 = float(wp[0]) * (2 * np.pi / fs)
                wp1 = float(wp[1]) * (2 * np.pi / fs)

                # 双线性变换
                wp[0] = (2 * fs) * np.tan(wp0 / 2)
                wp[1] = (2 * fs) * np.tan(wp1 / 2)

                omega_p = [float(wp[0]), float(wp[1])]
                wst = str(feature.An_wst).split()  # 切分开之后再转换为array
                wst0 = float(wst[0]) * (2 * np.pi / fs)
                wst1 = float(wst[1]) * (2 * np.pi / fs)

                # 双线性变换
                wst[0] = (2 * fs) * np.tan(wst0 / 2)
                wst[1] = (2 * fs) * np.tan(wst1 / 2)

                omega_st = [wst[0], wst[1]]
            else:
                wp = float(feature.An_wp) * (2 * np.pi / fs)
                wst = float(feature.An_wst) * (2 * np.pi / fs)
                # 频率预畸
                omega_p = (2 * fs) * np.tan(wp / 2)
                omega_st = (2 * fs) * np.tan(wst / 2)

            Asinput = str(feature.As).split()
            Rpinput = str(feature.Rp).split()
            feature.As = float(Asinput[0])
            feature.Rp = float(Rpinput[0])
            rs_in = float(Asinput[1])
            rp_in = float(Rpinput[1])

            feature.Rp = float(feature.Rp)
            feature.As = float(feature.As)
            N, Wn = signal.ellipord(omega_p, omega_st, feature.Rp, feature.As, True)
            feature.filts = signal.lti(*signal.ellip(N, rp_in, rs_in, Wn, btype=str(feature.filterType),
                                                     analog=True))
            feature.filtz = signal.lti(*signal.bilinear(feature.filts.num, feature.filts.den, fs))

            feature.z, feature.p = signal.bilinear(feature.filts.num, feature.filts.den, fs)

            wz, hz = signal.freqz(feature.filtz.num, feature.filtz.den)

            ax = feature.fig2.add_subplot(111)
            ax.cla()
            ax.semilogx(wz * fs / (2 * np.pi), 20 * np.log10(np.abs(hz).clip(1e-15)))
            ax.set_xlabel('Hz')
            ax.set_ylabel('dB')
            ax.set_title('考尔/椭圆滤波器')
            feature.fig2.subplots_adjust(left=None, bottom=0.2, right=None, top=None, wspace=None, hspace=None)
            feature.canvas2.draw()

            # 绘制零极点图
            ax = feature.fig4.add_subplot(111)
            ax.cla()
            z1, p1, k1 = signal.tf2zpk(feature.z, feature.p)  # zero, pole and gain
            c = np.vstack((feature.p, feature.z))
            Max = (abs(c)).max()  # find the largest value
            a = feature.p / Max  # normalization
            b = feature.z / Max
            Ra = (a * (2 ** ((N - 1) - 1))).astype(int)  # quantizan and truncate
            Rb = (b * (2 ** ((N - 1) - 1))).astype(int)
            z2, p2, k2 = signal.tf2zpk(Rb, Ra)
            # 参数方程画圆
            theta = np.arange(0, 2 * np.pi, 0.01)
            x = np.cos(theta)
            y = np.sin(theta)
            ax.plot(x, y, color='black')
            for i in p1:
                ax.plot(np.real(i), np.imag(i), 'bx')  # pole before quantization
            for i in z1:
                ax.plot(np.real(i), np.imag(i), 'bo')  # zero before quantization
            for i in p2:
                ax.plot(np.real(i), np.imag(i), 'rx')  # pole after quantization
            for i in z2:
                ax.plot(np.real(i), np.imag(i), 'ro')  # zero after quantization
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-1.2, 1.2)
            ax.grid()
            ax.set_title("%d bit 量化" % N)
            feature.fig4.subplots_adjust(left=None, bottom=0.2, right=None, top=None, wspace=None, hspace=None)
            feature.canvas4.draw()

        feature.textBrowser_3.append("*********")
        feature.textBrowser_3.setText("滤波器参数")
        feature.textBrowser_3.append("*********")
        feature.textBrowser_3.append("滤波类型=" + str(feature.filterType))
        feature.textBrowser_3.append("IIR类型=" + str(feature.iirType))
        feature.textBrowser_3.append("阶数=" + str(N))
        feature.textBrowser_3.append("分子系数=" + str(feature.z))
        feature.textBrowser_3.append("分母系数=" + str(feature.p))
        feature.textBrowser_3.append()

    # 应用IIR滤波器
    def apply_IIR(self, feature):
        f = wave.open(feature.path, "rb")
        params = f.getparams()
        nchannels, sampwidth, framerate, nframes = params[:4]
        # nchannels通道数
        # sampwidth量化位数
        # framerate采样频率
        # nframes采样点数
        str_data = f.readframes(nframes)
        f.close()
        # 将字符串转换为数组，得到一维的short类型的数组
        wave_data = np.fromstring(str_data, dtype=np.short)
        # 赋值的归一化
        maximum = max(abs(wave_data))
        wave_data = wave_data * 1.0 / (maximum)
        # 整合左声道和右声道的数据
        wave_data = np.reshape(wave_data, [nframes, nchannels])

        # 最后通过采样点数和取样频率计算出每个取样的时间
        time = np.arange(0, nframes) * (1.0 / framerate)
        # print(time)
        # 均匀线性分布
        t = np.linspace(0, nframes / framerate, nframes, endpoint=False)

        # 应用滤波器scipy.signal.filtfilt(b, a, x, axis=- 1, padtype='odd',
        # padlen=None, method='pad', irlen=None)
        # TODO method gust 还是 pad
        feature.yout = signal.filtfilt(feature.z, feature.p, wave_data[:, 0], method='gust')

        StartTime0 = np.int(feature.StartTime)
        DurationTime0 = np.int(feature.DurationTime)
        StartTime1 = np.int((feature.SampleFreq / 1000) * StartTime0)
        DurationTime1 = np.int((StartTime1 + DurationTime0 * (feature.SampleFreq / 1000)))

        ax = feature.fig5.add_subplot(111)
        # 调整图像大小
        ax.cla()
        ax.plot(t, feature.yout, color='#dc143c')
        ax.set_title('滤波后波形图', fontsize=10)
        ax.set_xlabel('时间 [sec]', fontsize=10)

        feature.fig5.subplots_adjust(left=None, bottom=0.215, right=None, top=None, wspace=None, hspace=None)
        feature.canvas5.draw()

        dx = feature.fig8.add_subplot(111)
        # 调整图像大小
        dx.plot(t[StartTime1:DurationTime1], feature.yout[StartTime1:DurationTime1], color='#dc143c')
        dx.set_title('归一化幅度', fontsize=10)
        dx.set_xlabel('时间[sec]', fontsize=10)

        feature.fig8.subplots_adjust(left=0.1, bottom=0.215, right=None, top=None, wspace=None, hspace=None)
        feature.canvas8.draw()

        # 绘制出时域的图像之后，再到频率分析
        # FFT变换#
        fft_signal = np.fft.fft(feature.yout)
        fft_signal = np.fft.fftshift(abs(fft_signal))
        fft_signal = fft_signal[int(fft_signal.shape[0] / 2):]
        # 建立频率轴
        Freq = np.arange(0, framerate / 2, framerate / (2 * len(fft_signal)))

        # 绘图
        ax = feature.fig6.add_subplot(111)
        # 调整图像大小
        ax.cla()
        ax.plot(Freq, fft_signal, color='#dc143c')
        ax.set_title('FFT 图像', fontsize=10)
        ax.set_xlabel('频率 [Hz]', fontsize=10)
        ax.set_ylabel('强度 [dB]', fontsize=10)
        feature.fig6.subplots_adjust(left=None, bottom=0.215, right=None, top=None, wspace=None, hspace=None)
        feature.canvas6.draw()

        ex = feature.fig10.add_subplot(111)
        # 调整图像大小
        ex.plot(Freq, fft_signal, color='#dc143c')
        ex.set_title('FFT 图像', fontsize=10)
        ex.set_xlabel('频率 [Hz]', fontsize=10)
        ex.set_ylabel('强度 [dB]', fontsize=10)
        feature.fig10.subplots_adjust(left=None, bottom=0.215, right=None, top=None, wspace=None, hspace=None)
        feature.canvas10.draw()

        # 写音频文件###

        feature.yout = feature.yout * maximum  # 去归一化
        feature.yout = feature.yout.astype(np.short)  # 强制转换short类型
        f = wave.open(feature.saveDatepath_IIR, "wb")  # 创建只写保存文件
        f.setnchannels(nchannels)  # 设置声道数
        f.setsampwidth(sampwidth)  # 设置采样字节长度
        f.setframerate(framerate)  # 设置采样频率
        f.setnframes(nframes)  # 设置总帧数
        f.writeframes(feature.yout.tostring())  # 写入 音频
        f.close()

        feature.process_flag = 1  # 代表本次处理完毕

    # 设计FIR滤波器
    def FIR_Designer(self, feature):
        kaiser_para = 0.85
        if feature.filterType_FIR == 'Lowpass':
            numtaps = int(feature.filter_length)  # 滤波器长度取整形
            fcut = feature.f2 * 2 / feature.fs_FIR
            if str(feature.firType) == 'kaiser':
                width = kaiser_para
            else:
                width = None
            # scipy.signal.firwin(numtaps, cutoff, width=None, window='hamming',
            # pass_zero=True, scale=True, nyq=None, fs=None)
            feature.FIR_b = signal.firwin(numtaps, fcut, width=width, window=str(feature.firType))
        if feature.filterType_FIR == 'Highpass':
            numtaps = int(feature.filter_length)
            fcut = feature.f2 * 2 / feature.fs_FIR
            if str(feature.firType) == 'kaiser':
                width = kaiser_para
            else:
                width = None
            feature.FIR_b = signal.firwin(numtaps, fcut, width=width, window=str(feature.firType), pass_zero=False)
        if feature.filterType_FIR == 'Bandpass':
            numtaps = int(feature.filter_length)
            fcut = [feature.f1 * 2 / feature.fs_FIR, feature.f2 * 2 / feature.fs_FIR]
            if str(feature.firType) == 'kaiser':
                width = kaiser_para
            else:
                width = None
            feature.FIR_b = signal.firwin(numtaps, fcut, width=width, window=str(feature.firType), pass_zero=False)
        if feature.filterType_FIR == 'Bandstop':
            numtaps = int(feature.filter_length)
            fcut = [feature.f1 * 2 / feature.fs_FIR, feature.f2 * 2 / feature.fs_FIR]
            if str(feature.firType) == 'kaiser':
                width = kaiser_para
            else:
                width = None
            feature.FIR_b = signal.firwin(numtaps, fcut, width=width, window=str(feature.firType))  #

        feature.textBrowser_3.append("FIR滤波类型:" + feature.filterType_FIR)
        feature.textBrowser_3.append("FIR类型" + str(feature.firType))
        feature.textBrowser_3.append("     *****设计完成*****    ")
        # 绘制频率响应：
        wz, hz = signal.freqz(feature.FIR_b)

        ax = feature.fig2.add_subplot(111)
        ax.cla()
        ax.semilogx(wz * feature.fs_FIR / (2 * np.pi), 20 * np.log10(np.abs(hz).clip(1e-15)))
        ax.set_xlabel('Hz')
        ax.set_ylabel('dB')
        ax.set_title(str(feature.firType))
        feature.fig2.subplots_adjust(left=None, bottom=0.2, right=None, top=None, wspace=None, hspace=None)
        feature.canvas2.draw()

        # 绘制零极点图
        # 只有零点没有极点
        ax = feature.fig4.add_subplot(111)
        ax.cla()
        fir_a = np.zeros(numtaps)  # 生成与阶数相等的全为0的数组
        fir_a[numtaps - 1] = 1  # 最后一位设为1
        z1, p1, k1 = signal.tf2zpk(feature.FIR_b, fir_a)  # 从线性滤波器的分子、分母表示，返回零、极点、增益
        c = np.vstack((fir_a, feature.FIR_b))
        Max = (abs(c)).max()  # 寻找最大值
        a = fir_a / Max  # 归一化
        b = feature.FIR_b / Max  # 归一化
        Ra = (a * (2 ** ((numtaps - 1) - 1))).astype(int)  # 量化
        Rb = (b * (2 ** ((numtaps - 1) - 1))).astype(int)
        z2, p2, k2 = signal.tf2zpk(Rb, Ra)  # 量化后的零点、极点、增益
        # 参数方程画圆
        theta = np.arange(0, 2 * np.pi, 0.01)
        x = np.cos(theta)
        y = np.sin(theta)
        ax.plot(x, y, color='black')
        for i in p1:
            ax.plot(np.real(i), np.imag(i), 'bx')  # pole before quantization
        for i in z1:
            ax.plot(np.real(i), np.imag(i), 'bo')  # zero before quantization
        for i in p2:
            ax.plot(np.real(i), np.imag(i), 'rx')  # pole after quantization
        for i in z2:
            ax.plot(np.real(i), np.imag(i), 'ro')  # zero after quantization
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.2, 1.2)
        ax.grid()
        ax.set_title("%d 点量化" % numtaps)
        feature.fig4.subplots_adjust(left=None, bottom=0.2, right=None, top=None, wspace=None, hspace=None)
        feature.canvas4.draw()
        feature.textBrowser_3.setText("滤波器参数")
        feature.textBrowser_3.append("*********")
        feature.textBrowser_3.append("滤波类型=" + str(feature.filterType_FIR))
        feature.textBrowser_3.append("FIR类型" + str(feature.firType))
        feature.textBrowser_3.append("阶数(滤波器长度)=" + str(numtaps))
        feature.textBrowser_3.append("分子系数=" + str(feature.FIR_b))
        feature.textBrowser_3.append("分母系数=" + str(fir_a))

    # 应用FIR滤波器
    def apply_FIR(self, feature):

        f = wave.open(feature.path, "rb")
        params = f.getparams()
        nchannels, sampwidth, framerate, nframes = params[:4]
        # nchannels通道数
        # sampwidth量化位数
        # framerate采样频率
        # nframes采样点数
        str_data = f.readframes(nframes)
        f.close()
        # 将字符串转换为数组，得到一维的short类型的数组
        wave_data = np.fromstring(str_data, dtype=np.short)
        # 赋值的归一化
        maximum = max(abs(wave_data))
        wave_data = wave_data * 1.0 / (maximum)
        # 整合左声道和右声道的数据
        wave_data = np.reshape(wave_data, [nframes, nchannels])

        # 最后通过采样点数和取样频率计算出每个取样的时间
        time = np.arange(0, nframes) * (1.0 / framerate)

        t = np.linspace(0, nframes / framerate, nframes, endpoint=False)

        feature.yout = signal.filtfilt(feature.FIR_b, 1, wave_data[:, 0])

        StartTime0 = np.int(feature.StartTime)
        DurationTime0 = np.int(feature.DurationTime)
        StartTime1 = np.int((framerate / 1000) * StartTime0)
        DurationTime1 = np.int((StartTime1 + DurationTime0 * (framerate / 1000)))
        # 传入数组的变量必须是整型，而乘除法返回的是浮点

        ax = feature.fig5.add_subplot(111)
        # 调整图像大小
        ax.cla()
        ax.plot(t, feature.yout, color='#dc143c')
        ax.set_title('滤波后波形图', fontsize=10)
        ax.set_xlabel('时间 [sec]', fontsize=10)

        feature.fig5.subplots_adjust(left=None, bottom=0.213, right=None, top=None, wspace=None, hspace=None)
        feature.canvas5.draw()

        dx = feature.fig8.add_subplot(111)
        # 调整图像大小
        dx.plot(t[StartTime1:DurationTime1], feature.yout[StartTime1:DurationTime1], color='#dc143c')
        dx.set_title('归一化幅度', fontsize=10)
        dx.set_xlabel('时间[sec]', fontsize=10)

        feature.fig8.subplots_adjust(left=0.1, bottom=0.215, right=None, top=None, wspace=None, hspace=None)
        feature.canvas8.draw()

        # 绘制出时域的图像之后，再到频率分析
        # FFT变换#
        fft_signal = np.fft.fft(feature.yout)
        fft_signal = np.fft.fftshift(abs(fft_signal))[int(fft_signal.shape[0] / 2):]
        # 建立频率轴
        Freq = np.arange(0, framerate / 2, framerate / (2 * len(fft_signal)))

        # 绘图
        ax = feature.fig6.add_subplot(111)
        # 调整图像大小
        ax.cla()
        ax.plot(Freq, fft_signal, color='#dc143c')
        ax.set_title('FFT 图像', fontsize=10)
        ax.set_xlabel('频率 [Hz]', fontsize=10)
        ax.set_ylabel('强度', fontsize=10)
        feature.fig6.subplots_adjust(left=None, bottom=0.214, right=None, top=None, wspace=None, hspace=None)
        feature.canvas6.draw()

        ex = feature.fig10.add_subplot(111)
        # 调整图像大小
        # ex.cla()
        ex.plot(Freq, fft_signal, color='#dc143c')
        ex.set_title('FFT 图像', fontsize=10)
        ex.set_xlabel('频率 [Hz]', fontsize=10)
        ex.set_ylabel('强度 [dB]', fontsize=10)
        feature.fig10.subplots_adjust(left=None, bottom=0.214, right=None, top=None, wspace=None, hspace=None)
        feature.canvas10.draw()

        # feature.precessed_Audio=feature.filtz.output(wave_data,time,X0=None)#求系统的零状态响应
        # feature.precessed_Audio =feature.precessed_Audio.tostring()
        # feature.process_flag=1#标志位为1，代表处理好了，否则的话就代表没有
        # 写音频文件
        feature.yout = feature.yout * maximum  # 去归一化
        feature.yout = feature.yout.astype(np.short)
        f = wave.open(feature.saveDatepath_FIR, "wb")  # 写入文件
        f.setnchannels(nchannels)
        f.setsampwidth(sampwidth)
        f.setframerate(framerate)
        f.setnframes(nframes)
        f.writeframes(feature.yout.tostring())
        f.close()
        feature.process_flag = 1  # 代表本次处理完毕

# 定义一个信号，与槽函数绑定，每当发送这个信号时，就将调用绑定的槽函数
class Signal(QObject):
    text_update = pyqtSignal(str)

    # 在TextBrower_3上打印控制台输出
    def write(self, text):
        self.text_update.emit(str(text))  # 定义一个发送信号的函数 →886行
        # loop = QEventLoop()
        # QTimer.singleShot(100, loop.quit)
        # loop.exec_()
        QApplication.processEvents()

# 因为界面py文件和逻辑控制py文件分开的，所以在引用的时候要加上文件名再点出对象
class MyMainForm(QMainWindow, ui_cn_1.Ui_MainWindow):

    def __init__(self, parent=None):
        super(MyMainForm, self).__init__(parent)  # 从父类继承
        self.Process = ProcessFunction()  # process对象包含所有的信号处理函数及其画图
        self.saveDatepath_IIR = os.getcwd().replace('\\', '/') + "/ProcessedSignal/sweepiir.wav"  # iir保存位置
        self.saveDatepath_FIR = os.getcwd().replace('\\', '/') + "/ProcessedSignal/sweepfir.wav"  # fir保存位置

        self.setupUi(self)  # setupUi是Ui_FilterDesigner类里面的一个方法，这里的self是两个父类的子类的一个实例

        # 实时显示输出, 将控制台的输出重定向到界面中
        sys.stdout = Signal()
        sys.stdout.text_update.connect(self.updatetext)  # 控制台输出，这个信号 绑定到updatetext上

        self.progressBar.setValue(0)  # 进度条初始化为0
        self.progressBar_2.setValue(0)  # 进度条初始化为0
        # 标志位初始化
        self.process_flag = 0  # 处理完毕标志位
        self.isPlay = 0  # 播放器播放标志位
        self.isPlay_IIR = 0  # 播放器播放标志位
        self.isPlay_FIR = 0

        # 播放器的设定

        self.player = QMediaPlayer(self)  # 这个播放器是播放原声的
        self.player_IIR = QMediaPlayer(self)  # 定义两个对象出来，这个负责播放处理过后的

        # 处理前的进度条
        self.horizontalSlider_3.sliderMoved[int].connect(
            lambda: self.player.setPosition(self.horizontalSlider_3.value()))
        self.horizontalSlider_3.setStyle(QStyleFactory.create('Fusion'))

        # 处理后的进度条
        self.horizontalSlider_4.sliderMoved[int].connect(
            lambda: self.player_IIR.setPosition(self.horizontalSlider_4.value()))
        self.horizontalSlider_4.setStyle(QStyleFactory.create('Fusion'))

        self.timer = QTimer(self)
        self.timer.start(1000)  # 定时器设定为1s，超时过后链接到playRefresh刷新页面
        self.timer.timeout.connect(self.playRefresh)  # 

        # 菜单栏的事件绑定
        self.actionFile.triggered.connect(self.onFileOpen)  # 菜单栏的action打开文件
        self.actionQuit.triggered.connect(self.close)  # 菜单栏的退出action
        self.Timelayout_()  # 时间域的四个图窗布局

        # 分析信号配置
        self.dial_2.setValue(20)  # 默认音量大小为20
        self.dial_2.valueChanged.connect(self.changeVoice0)  # 音量圆盘控制事件绑定,如果值被改变就调起事件
        self.pushButton_analyse.clicked.connect(self.AnalyseStart)  # 给pushButton_3添加一个点击事件
        self.pushButton_5.clicked.connect(self.playMusic)

        # 设计应用滤波器配置
        self.dial_2.setValue(20)  # 默认音量大小为20
        self.dial_2.valueChanged.connect(self.changeVoice)  # 音量圆盘控制事件绑定,如果值被改变就调起事件
        self.dial_3.setValue(20)  # 默认音量大小为20
        self.dial_3.valueChanged.connect(self.changeVoice)
        self.pushButton_18.clicked.connect(self.designIIR)  # 点击开始设计IIR滤波器按钮之后，调用函数
        self.pushButton_21.clicked.connect(self.applyIIR)  # 点击应用滤波器
        self.pushButton_4.clicked.connect(self.playMusic)
        self.pushButton_5.clicked.connect(self.playIIRaudio)
        self.pushButton_20.clicked.connect(self.designFIR)  # 点击开始设计IIR滤波器按钮之后，调用函数
        self.pushButton_19.clicked.connect(self.applyFIR)  # 点击应用滤波器

    # 输出错误信息到UI内文字框
    def updatetext(self, text):
        """
            更新textBrowser
        """
        cursor = self.textBrowser_3.textCursor()  # textCursor 类提供了访问和修改 QTextDocuments的 API
        cursor.movePosition(QTextCursor.End)  # Move to the end of the document.
        self.textBrowser_3.append(text)
        self.textBrowser_3.setTextCursor(cursor)
        self.textBrowser_3.ensureCursorVisible()  # Ensures that the cursor is visible by scrolling the text edit if necessary.

    # 绑定图表
    def Timelayout_(self):
        self.fig1 = plt.figure()
        self.canvas1 = FigureCanvas(self.fig1)
        layout = QVBoxLayout()  # 垂直布局
        layout.addWidget(self.canvas1)
        self.graphicsView.setLayout(layout)  # 设置好布局之后调用函数

        self.fig2 = plt.figure()
        self.canvas2 = FigureCanvas(self.fig2)
        layout = QVBoxLayout()  # 垂直布局
        layout.addWidget(self.canvas2)
        self.graphicsView_2.setLayout(layout)  # 设置好布局之后调用函数

        self.fig3 = plt.Figure()
        self.canvas3 = FigureCanvas(self.fig3)
        layout = QVBoxLayout()  # 垂直布局
        layout.addWidget(self.canvas3)
        self.graphicsView_3.setLayout(layout)  # 设置好布局之后调用函数

        self.fig4 = plt.Figure()
        self.canvas4 = FigureCanvas(self.fig4)
        layout = QVBoxLayout()  # 垂直布局
        layout.addWidget(self.canvas4)
        self.graphicsView_4.setLayout(layout)  # 设置好布局之后调用函数

        self.fig5 = plt.Figure()
        self.canvas5 = FigureCanvas(self.fig5)
        layout = QVBoxLayout()  # 垂直布局
        layout.addWidget(self.canvas5)
        self.graphicsView_5.setLayout(layout)  # 设置好布局之后调用函数

        self.fig6 = plt.Figure()
        self.canvas6 = FigureCanvas(self.fig6)
        layout = QVBoxLayout()  # 垂直布局
        layout.addWidget(self.canvas6)
        self.graphicsView_6.setLayout(layout)  # 设置好布局之后调用函数

        self.fig8 = plt.Figure()
        self.canvas8 = FigureCanvas(self.fig8)
        layout = QVBoxLayout()  # 垂直布局
        layout.addWidget(self.canvas8)
        self.graphicsView_8.setLayout(layout)  # 设置好布局之后调用函数

        self.fig10 = plt.Figure()
        self.canvas10 = FigureCanvas(self.fig10)
        layout = QVBoxLayout()  # 垂直布局
        layout.addWidget(self.canvas10)
        self.graphicsView_10.setLayout(layout)  # 设置好布局之后调用函数

    # 以下是UI按钮绑定的函数
    # 打开文件
    def onFileOpen(self):  # 打开文件
        self.path, _ = QFileDialog.getOpenFileName(self, '打开文件', '', '音乐文件 (*.wav)')

        if self.path:  # 选中文件之后就选中了需要播放的音乐，并同时显示出来
            self.isPlay = 0  # 每次打开文件的时候就需要暂停播放，无论是否在播放与否
            self.isPlay_IIR = 0

            self.player.pause()
            self.player_IIR.pause()
            # 951 2 to 3; 956 3 to 4
            self.player.setMedia(QMediaContent(QUrl(self.path)))  # 选中需要播放的音乐
            self.horizontalSlider_3.setMinimum(0)
            self.horizontalSlider_3.setMaximum(self.player.duration())
            self.horizontalSlider_3.setValue(self.horizontalSlider_3.value() + 1000)
            self.horizontalSlider_3.setSliderPosition(0)

            self.horizontalSlider_4.setMinimum(0)
            self.horizontalSlider_4.setMaximum(self.player.duration())
            self.horizontalSlider_4.setValue(self.horizontalSlider_3.value() + 1000)
            self.horizontalSlider_4.setSliderPosition(0)

            self.label_23.setText("当前文件:  " + os.path.basename(self.path))

    # 分析打开的文件按钮
    def AnalyseStart(self):  # 这里对应的是打开文件，并点击按钮
        try:
            if self.path:  # 要必须在打开文件之后才允许进行处理
                self.textBrowser_3.append("*********文件 :" + str(os.path.basename(self.path)) + "*********")
                self.progressBar.setValue(0)  # 每次允许处理时进度条归0
                self.progressBar_2.setValue(0)  # 进度条初始化为0
                self.StartTime = self.lineEdit_6.text()
                self.DurationTime = self.lineEdit_7.text()
                self.Process.Audio_TimeDomain(self)  # 把实例传入进去
                self.progressBar.setValue(33)
                self.progressBar_2.setValue(33)
                self.Process.Audio_FrequencyDomain(self)
                self.progressBar.setValue(66)
                self.progressBar_2.setValue(66)
                self.Process.Audio_SpectrogramDomain(self)
                self.progressBar.setValue(100)
                self.progressBar_2.setValue(100)
                self.textBrowser_3.append("分析成功!")
                self.textBrowser_3.append(
                    "---------  " + str(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime())) + "  ---------")

        except Exception as e:
            print(e)
            self.textBrowser_3.setText("尝试打开文件时出错")

    # 播放音乐
    def playMusic(self):
        try:
            if self.path:  # 这个path是当前的路径，如果path变了，那么就意味着更换了文件
                if not self.isPlay:  # 如果isPlay=0，那就说明播放器并没有打开，且此时按下了播放按钮，就开始播放
                    self.player.play()
                    self.isPlay = 1  # 播放之后同时置为1，代表播放器目前正在播放
                else:
                    self.player.pause()
                    self.isPlay = 0  # 暂停之后同时置为0，代表播放器目前没有播放
        except Exception as e:
            print(e)
            self.textBrowser_3.setText("播放音频时出错")

    # 进度条更新
    def playRefresh(self):
        if self.isPlay:
            self.horizontalSlider_4.setMinimum(0)
            self.horizontalSlider_4.setMaximum(self.player_IIR.duration())
            self.horizontalSlider_4.setValue(self.horizontalSlider_4.value() + 1000)

            self.horizontalSlider_3.setMinimum(0)
            self.horizontalSlider_3.setMaximum(self.player.duration())
            self.horizontalSlider_3.setValue(self.horizontalSlider_3.value() + 1000)

        elif self.isPlay_IIR:
            self.horizontalSlider_4.setMinimum(0)
            self.horizontalSlider_4.setMaximum(self.player_IIR.duration())
            self.horizontalSlider_4.setValue(self.horizontalSlider_4.value() + 1000)
            if self.horizontalSlider_4.value() == 5000:  # 排除bug 停止计数以重新写入文件
                self.player_IIR.stop()
                self.isPlay_IIR = 0

        # ORIGINAL AUDIO
        self.label_24.setText(time.strftime('%M:%S', time.localtime(self.player.position() / 1000)))
        self.label_25.setText(time.strftime('%M:%S', time.localtime(self.player.duration() / 1000)))

        self.label_27.setText(time.strftime('%M:%S', time.localtime(self.player_IIR.position() / 1000)))
        self.label_28.setText(time.strftime('%M:%S', time.localtime(self.player_IIR.duration() / 1000)))

    # 第一个dial按钮
    def changeVoice(self):
        self.player_IIR.setVolume(self.dial_3.value())
        self.dial_3.setValue(self.dial_3.value())

    # 第二个dial按钮
    def changeVoice0(self):
        self.player.setVolume(self.dial_2.value())
        self.dial_2.setValue(self.dial_2.value())

    # IIR参数传入
    def designIIR(self):
        ###获取到输入参数：滤波器四个指标
        self.progressBar_2.setValue(0)
        try:
            self.An_wp = self.lineEdit_3.text()
            self.An_wst = self.lineEdit_2.text()
            self.Rp = self.lineEdit.text()
            self.As = self.lineEdit_4.text()

            self.fs = self.lineEdit_5.text()

            self.filterType = self.comboBox_2.currentText()
            self.iirType = self.comboBox_3.currentText()
            self.progressBar_2.setValue(10)
            self.Process.IIR_Designer(self)
        except Exception as e:
            print(e)
        self.progressBar_2.setValue(100)

    # IIR进行滤波
    def applyIIR(self):
        self.progressBar_2.setValue(0)
        try:
            self.process_flag = 0
            self.player_IIR.pause()  # 暂停当前播放
            self.player_IIR.setMedia(QMediaContent(QUrl(self.path)))  # 避免占用
            self.progressBar_2.setValue(20)
            self.SampleFreq = float(self.lineEdit_5.text())
            self.StartTime = self.lineEdit_6.text()
            self.DurationTime = self.lineEdit_7.text()
            self.Process.apply_IIR(self)
            if self.process_flag:  # 如果处理好了
                try:
                    self.isPlay = 0  # 不再播放
                    self.isPlay_IIR = 0
                    self.player.pause()  # 暂停另外的播放器
                    self.horizontalSlider_4.setMinimum(0)  # 设置最小值
                    self.horizontalSlider_4.setMaximum(self.player_IIR.duration())  # 设置最大值，毫秒
                    self.horizontalSlider_4.setValue(self.horizontalSlider_4.value() + 1000)
                    self.horizontalSlider_4.setSliderPosition(0)  # 进度条清零
                    self.label_26.setText("滤波后音频文件: " + os.path.basename(self.path))

                    self.player_IIR.setMedia(QMediaContent(QUrl(self.saveDatepath_IIR)))  # 选中需要播放的音乐
                    # self.player_IIR.setMedia(QMediaContent(), buf)  # 从缓存里面读出来的

                # self.player_IIR.setMedia(QMediaContent(QUrl(self.path)))
                except Exception as e:
                    print(e)
            else:  # 没有处理好，也就是没有进行滤波操作
                self.textBrowser_3.setText("请先进行滤波器设计")
        except Exception as e:
            print(e)
        self.progressBar_2.setValue(100)

    # 播放滤波处理后的音频
    def playIIRaudio(self):
        try:
            self.isPlay = 0  # 点按任意一个播放器的播放暂停按钮都会停止
            self.player.pause()

            if self.process_flag:  # 如果处理好了
                if not self.isPlay_IIR:
                    self.horizontalSlider_4.setValue(self.player_IIR.position())
                    self.player_IIR.play()
                    self.isPlay_IIR = 1
                    self.textBrowser_3.append("播放")
                else:  # 如果发现播放器正在播放
                    self.player_IIR.pause()
                    self.isPlay_IIR = 0
                    self.textBrowser_3.append("暂停")
            else:  # 没有处理好，也就是没有进行滤波操作
                self.textBrowser_3.setText("请先应用滤波器")
        except Exception as e:
            print(e)

    # FIR滤波器参数传入
    def designFIR(self):
        try:
            self.progressBar.setValue(0)
            self.f1 = float(self.lineEdit_3.text())

            self.f2 = float(self.lineEdit_2.text())

            self.filter_length = float(self.lineEdit_16.text())

            self.fs_FIR = float(self.lineEdit_5.text())

            self.filterType_FIR = self.comboBox_6.currentText()

            self.firType = self.comboBox_9.currentText()

            self.progressBar.setValue(20)
            self.Process.FIR_Designer(self)
            self.progressBar.setValue(100)
        except Exception as e:
            print(e)

    # 应用FIR滤波器
    def applyFIR(self):
        self.progressBar.setValue(0)
        try:
            self.process_flag = 0
            # 处理之前全部关掉播放器对音乐的链接，不然会导致文件写不进去
            self.isPlay = 0
            self.isPlay_IIR = 0
            self.player.pause()  # 暂停另外的播放器
            self.player_IIR.pause()
            self.player_IIR.setMedia(QMediaContent(QUrl(self.path)))  # 先绑定到其他地方去，避免占用
            self.progressBar.setValue(20)
            self.SampleFreq = float(self.lineEdit_5.text())
            self.StartTime = self.lineEdit_6.text()
            self.DurationTime = self.lineEdit_7.text()
            self.Process.apply_FIR(self)

            #print(self.process_flag)
            if self.process_flag:  # 如果处理好了
                self.horizontalSlider_4.setMinimum(0)
                self.horizontalSlider_4.setMaximum(self.player_IIR.duration())
                self.horizontalSlider_4.setValue(self.horizontalSlider_4.value() + 1000)
                self.horizontalSlider_4.setSliderPosition(0)
                self.label_26.setText("滤波器后音频: " + os.path.basename(self.path))

                self.player_IIR.setMedia(QMediaContent(QUrl(self.saveDatepath_FIR)))  # 选中需要播放的音乐

            else:  # 没有进行滤波操作
                self.textBrowser_3.setText("请先进行滤波器设计")
        except Exception as e:
            print(e)
        self.progressBar.setValue(100)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWin = MyMainForm()
    myWin = CandyWindow.createWindow(myWin, 'pink', title='DSP滤波器系统 by:通信3班 王重诺')
    myWin.show()
    sys.exit(app.exec_())
