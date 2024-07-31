"""
    * @file: das.py
    * @version: v1.0.0
    * @author: Zhiyu Zhang
    * @desc: 
    * @date: 2023-07-25 10:09:34
    * @Email: erbiaoger@gmail.com
    * @url: erbiaoger.site

"""

import os
import pathlib
import numpy as np
from scipy import fft

import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO
from sklearn.linear_model import LinearRegression
from scipy.stats import linregress

from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as colors
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['font.size'] = 16


from dasQt.utools.config import loadConfig, saveConfig
from dasQt.utools.readh5 import readh5
from dasQt.utools.readdat import readdat
from dasQt.CarClass.norm_trace1 import norm_trace
from dasQt.Process.bpFilter import bpFilter
from dasQt.Process.fkDip import fkDip
from dasQt.Process.freqattributes import (spectrum, spectrogram, fk_transform)
from dasQt.Cog.cog import ncf_corre_cog
from dasQt.CarClass.curves_class import pickPoints, autoSeparation, deleteSmallClass, showClass, getVelocity, classCar
from dasQt.utools.logPy3 import HandleLog


class DAS():
    def __init__(self) -> None:
        # data 
        self.data            : np.ndarray = None
        self.orig_data       : np.ndarray = None
        
        self.nt              : int        = None
        self.nx              : int        = None
        self.dt              : int        = None
        self.dx              : int        = None
        self.speed           : float       = 0.25 # 1x 2x x 0.5x 0.25x
        self.startX          : int        = 0
        self.fIndex          : int        = 0
        self.display_T       : float       = 30.0   # display time [s]
        self.scale           : float       = 1.0
        self.cc              : np.ndarray = None
        self.CClist : list[str]  = []
        self.colormap        : str        = 'rainbow'

        # Process bool
        self.bool_cut         : bool = False
        self.bool_downSampling: bool = False
        self.bool_filter       : bool = False
        self.bool_columnFK    : bool = False
        self.bool_rowFK       : bool = False
        self.bool_saveCC      : bool = False
        self.bool_saveFig     : bool = False
        self.bool_upcc        : bool = False
        self.bool_downcc      : bool = False
        

        # Car Class
        self.threshold: float  = 0.01
        self.skip_Nch : int   = 2
        self.skip_Nt  : int   = 1000
        self.maxMode  : int   = 10
        self.minCarNum: int   = 15
        self.to       : float  = 0.01
        self.mode     : str   = 'max'

        # Dispersion
        self.indexClick      : int  = 0
        self.dispersion_parse: dict = {}
        self.radon_parse     : dict = {}
        self.logger = HandleLog(os.path.split(__file__)[-1].split(".")[0], path=os.getcwd(), level="info")
        
        # self.model = YOLO('./dasQt/YOLO/best.pt')


    #-----------------------------------------------------------
    #
    # Load Data
    #
    #-----------------------------------------------------------

    def getData(self, fileType, filename):
        if fileType == '.h5':
            data, metadata = readh5(filename)
            self.color_value = 1000.
        elif fileType == '.dat':
            data, metadata = readdat(filename)
            self.color_value = 0.5
        else:
            self.logger.error("File Type Error!")
            raise ValueError("File Type Error!")
        
        return data, metadata


    def readData(self, filename: str='./Data/SR_2023-07-20_09-09-38_UTC.h5') -> None:
        """read data from h5 or dat file"""
        self.fname = filename
        file_path   = pathlib.Path(self.fname)
        fileType    = file_path.suffix

        data, metadata = self.getData(fileType, filename)
        self.data = data
        self.fs, self.dx, self.dt, self.nt, self.nx, self.gauge = metadata['fs'], metadata['dx'], metadata['dt'], metadata['nt'], metadata['nx'], metadata['gauge']

        self.profile_X   = np.arange(self.nx) * self.dx
        self.process_data = self.data.copy()
        filepath         = pathlib.Path(filename)

        self.process()
        self.logger.info(f"\n{filepath.name} read done!\nnt: {self.nt}, nx: {self.nx}, fs: {int(1./self.dt)}, dx: {self.dx}, gauge: {self.gauge}")
        if self.bool_saveFig:
            self.saveFig(colormap=self.colormap)



    def readMoreData(self, filename: str='./Data/SR_2023-07-20_09-09-38_UTC.h5', num: int=3) -> None:
        """read data from h5 or dat file"""
        self.fname       = filename
        id               = self.getFileID(filename)
        if self.fnames:
            self.fIndex += 1
            try:
                self.fname = self.fnames[self.fIndex]
                
            except IndexError:
                self.logger.error("Index Error!")
                raise IndexError("Index Error!")


        self.profile_X   = np.arange(self.nx) * self.dx
        self.process_data   = self.data.copy()
        filepath         = pathlib.Path(filename)

        self.process()
        self.logger.info(f"\n{filepath.name} read done!")
        self.logger.info(f"\nnt: {self.nt}, nx: {self.nx}, dt: {self.dt}, dx: {self.dx}")
        if self.bool_saveFig:
            self.saveFig(colormap=self.colormap)



    def openFolder(self, foldername: str='./das/') -> list[str]:
        self.foldername = foldername
        self.fnames     = sorted(os.listdir(foldername))
        self.logger.info(f"open {foldername} done!")

        return self.fnames


    def getFileID(self, fname: str) -> None:
        """get the file index by file name"""
        for index, value in enumerate(self.fnames):
            if value == fname:
                self.fIndex = index
                break

        self.logger.debug(f"{fname} is the {self.fIndex}s file")


    def process(self) -> None:
        """process data after read"""
        if self.bool_cut:
            self.cutData(self.profile_Xmin, self.profile_Xmax)
        if self.bool_downSampling:
            self.downSampling(self.initNumDownSampling)
        if self.bool_filter:
            self.RawDataBpFilter(self.fmin, self.fmin1, self.fmax, self.fmax1)
        if self.bool_columnFK:
            self.RawDataFKFilterColumn(self.w_column)
        if self.bool_rowFK:
            self.RawDataFKFilterRow(self.w_row)


    def readNextData(self) -> None:
        if self.fnames:
            self.fIndex += 1
            try:
                self.fname = self.fnames[self.fIndex]
                self.readData(os.path.join(self.foldername, self.fname))
                self.logger.debug(f"next file is {self.fname}")
            except IndexError:
                self.logger.error("Index Error!")
                raise IndexError("Index Error!")

    #-----------------------------------------------------------
    #
    # Process Data
    #
    #-----------------------------------------------------------



    def cutData(self, Xmin, Xmax) -> None:
        """Cut the data by Xmin and Xmax"""
        self.profile_Xmin = Xmin; self.profile_Xmax = Xmax
        if self.profile_X is None:
            self.profile_X = np.arange(self.nx) * self.dx
            

        Xmin_Num = np.abs(float(Xmin) - self.profile_X).argmin()
        if Xmax == 'end':
            Xmax_Num = self.nx
        else:
            Xmax_Num = np.abs(float(Xmax) - self.profile_X).argmin()

        self.process_data = self.process_data[:, Xmin_Num:Xmax_Num]
        self.profile_X = self.profile_X[Xmin_Num:Xmax_Num]
        self.nx = self.process_data.shape[1]
        self.logger.info("Cut Data Done!")



    def RawDataBpFilter(self, fmin, fmin1, fmax, fmax1) -> None:
        """"Bandpass filter for raw data"""
        self.fmin = fmin; self.fmin1 = fmin1; self.fmax = fmax; self.fmax1 = fmax1
        self.process_data = bpFilter(self.process_data, self.dt, fmin, fmin1, fmax, fmax1)

    def RawDataFKFilterColumn(self, w=0.05) -> None:
        self.w_column = w
        data = fkDip(self.process_data, w)
        self.process_data = self.process_data - data

    def RawDataFKFilterRow(self, w=0.05) -> None:
        self.w_row = w
        data = fkDip(self.process_data.T, w)
        self.process_data = self.process_data - data.T


    def muteData(d, dx, dt, Xmin, Ymin, Xmax, Ymax, mode='up', slope=0.0):
        # Xmin = 20
        # Ymin = 0.0
        # Xmax = 150
        # Ymax = 0.7

        X = np.arange(0, d.shape[1]*dx, dx)
        Y = np.arange(0, d.shape[0]*dt, dt)

        x2, y2 = np.abs(float(Xmin) - X).argmin(), np.abs(float(Ymin) - Y).argmin()
        x3, y3 = np.abs(float(Xmax) - X).argmin(), np.abs(float(Ymax) - Y).argmin()
        print(x2, y2, x3, y3)
        if slope == 0.0:
            # 计算斜边的斜率
            slope = (y3 - y2) / (x3 - x2)

        if mode == 'up':
            # 逐行赋值
            for x in range(x2, d.shape[1]):
                # 根据斜率和行号计算这一行中斜边对应的列号
                y = int(slope * (x - x2) + y2)
                # 将从起始列到斜边对应列的元素赋值为 0
                d[:y + 1, x] = 0
        elif mode == 'down':
            # 逐行赋值
            for x in range(0, d.shape[1]):
                # 根据斜率和行号计算这一行中斜边对应的列号
                y = int(slope * (x - x2) + y2)
                # 将从起始列到斜边对应列的元素赋值为 0
                d[y + 1:, x] = 0
        
        return d, slope


    def signalTrace(self, trace):
        idx = np.abs(self.profile_X - trace).argmin()
        data = self.process_data[:, idx]

        delta = self.dt
        npts = len(data) 
        t = delta * np.arange(npts)
        xf = fft.fftfreq(npts, delta)
        yf = fft.fft(data)

        return t, data, np.abs(xf), np.abs(yf)





    def caculateCogAll(self, Ndata, nch, win, nwin, overlap, offset, fmin1=2., fmin2=3., fmax1=20., fmax2=21., ylim=None):
        files = self.fnames
        path = pathlib.Path(self.fname).parent
        id = self.fIndex

        ncf_all  = np.zeros((win,nch))
        for file in files[id:id+Ndata]:
            self.readData(path/file)
            data = self.pre_data.copy()
            ncf_shot = ncf_corre_cog(data, nch, win, nwin, overlap, offset)
            ncf_all = ncf_all + ncf_shot

        ncf_all = np.fft.fftshift(ncf_all, axes=0)  # Shift along the first axis (similar to MATLAB's dimension 1)
        ncf_all = ncf_all + np.flip(ncf_all, axis=0)  # Flip and add along the first axis

        dt = self.dt
        a1 = bpFilter(ncf_all,dt,fmin1,fmin2,fmax1,fmax2)
        a1 = self.normalize_data(a1)
        a1 = a1[a1.shape[0]//2:,:]

        return a1

    def normalize_data(self, data):
        # find the maximum value of each column
        max_values = np.max(np.abs(data), axis=0)
        data_norm  = np.zeros(data.shape)
        for i in range(0, data.shape[0]):
            data_norm[i, :] = data[i, :] / max_values
        # return the normalized data
        return data_norm










    #-----------------------------------------------------------
    #
    # Imshow Data
    #
    #-----------------------------------------------------------

    def imshowData(self, ax, indexTime=0, colormap='rainbow', downsample=1):
        """"imshow raw data by indexTime and scale"""
        dt         = self.dt
        display_nt = self.display_T / self.dt
        indexTime  = int(indexTime / dt / 100)
        data       = self.process_data[int(indexTime): int(indexTime+display_nt)]
        self.logger.debug(f"dt: {dt}")
        self.colormap = colormap

        ax.imshow(data[::int(downsample)], 
                aspect = 'auto',
                origin = 'lower',
                cmap = colormap,
                extent = [self.profile_X[0], self.profile_X[-1], indexTime*dt, (indexTime+display_nt)*dt],
                vmin   = -self.color_value/self.scale,
                vmax   = self.color_value/self.scale)

        fname = os.path.basename(self.fname).split('.')[0]
        ax.set_xlabel('distance (m)')
        ax.set_ylabel('time (s)')
        ax.set_title(fname)

        return ax

    def saveFig(self, colormap='rainbow') -> None:
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        ax.imshow(self.process_data, 
                aspect = 'auto',
                # origin = 'lower',
                cmap = colormap,
                vmin   = -self.color_value/self.scale,
                vmax   = self.color_value/self.scale)

        fname = pathlib.Path(self.fname).stem
        save_path = pathlib.Path('./figs/')
        save_path.mkdir(parents=True, exist_ok=True)
        figname = save_path / f'{fname}.png'
        self.yolo_figname = figname
        ax.axis("off")
        ax.margins(0)
        plt.savefig(figname, dpi=300)
        self.logger.info(f"Save {figname} Done!")
        plt.close()
        
        

    def imshowDataAll(self, ax):
        """"imshow all raw data in one figure"""
        data       = self.process_data
        dt         = self.dt
        nt         = self.nt
        self.logger.debug(f"dt: {dt}")

        ax.imshow(data, 
                aspect = 'auto',
                cmap   = 'rainbow',
                origin = 'lower',
                extent = [self.profile_X[0], self.profile_X[-1], 0, nt*dt],
                vmin   = -self.color_value/self.scale,
                vmax   = self.color_value/self.scale)

        fname = os.path.basename(self.fname).split('.')[0]
        ax.set_xlabel('distance (m)')
        ax.set_ylabel('time (s)')
        ax.set_title(fname)

        return ax


    def imshowCarClass(self, ax, scale=1, 
                       skip_Nch=2, skip_Nt=1000, threshold=0.1, mode='min',
                       maxMode=10, minCarNum=15, to=0.01,line=0.5):
        dt    = self.dt
        dx    = self.dx
        Nt    = self.nt
        Nch   = self.profile_X.shape[0]
        data  = self.process_data

        # x = np.linspace(0, Nch * dx, Nch)        # x-axis
        x = self.profile_X
        print(Nch, x.shape)
        t = np.linspace(0, Nt * dt, Nt)          # t-axis

        # pick points

        curves = pickPoints(data, Nch, Nt, skip_Nch=skip_Nch, skip_Nt=skip_Nt, threshold=threshold, model=mode)
        if curves is None:
            return 0

        # auto separation
        # to = 0.01, threshold of distance
        # maxMode = 10, 10 classes
        # minCarNum = 15, min car number in one class

        curves_km = autoSeparation(curves, to=to, maxMode=maxMode)
        id_list   = np.unique(curves_km[...,-1])
        class_num = len(id_list)

        curves_km = deleteSmallClass(curves_km, class_num, minCarNum=minCarNum)

        id_list   = np.unique(curves_km[...,-1])
        class_num = len(id_list)

        # get velocity
        velocities, id_list_a = getVelocity(curves_km, x, t, id_list)

        # class car
        id_list_b = classCar(curves_km, id_list, scale=line)
        self.logger.info(f"velocities: {velocities}")

        scale = self.color_value / self.scale
        ax = showClass(data, curves_km, id_list_b, t, x, ax, 
                       s='b)', title="Only Car", model='vel', velocities=velocities, 
                       vmin=-scale, vmax=scale)
        self.logger.info("ImShow CarClass Done!")

        return ax

    def imshowCog(self, ax, Ndata, win, nwin, overlap, offset, fmin1=2., fmin2=3., fmax1=20., fmax2=21., ylim=None, vmin=None, vmax=None):
        nch = self.nx
        if ylim is None:
            cog = self.caculateCogAll(Ndata, nch, win, nwin, overlap, offset, fmin1, fmin2, fmax1, fmax2, ylim)
            self.cog = cog
            ax.imshow(cog, cmap='Greys', aspect='auto', extent=[0, self.nx*self.dx, cog.shape[0]*self.dt, 0])
        else:
            cog = self.cog
            ax.imshow(cog, cmap='Greys', aspect='auto', extent=[0, self.nx*self.dx, cog.shape[0]*self.dt, 0], vmin=vmin, vmax=vmax)
        print(cog.shape, self.dt)
        
        
        ax.set_title(f'cog with offset={offset}')
        ax.set_xlabel('Distance (m)')
        ax.set_ylabel('Time (s)')
        # ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        if ylim is not None:
            ax.set_ylim(ylim)

        return ax



    #@nb.jit(nopython=False)
    def wigb(self, ax, indexTime=0):

        dt = self.dt
        dx = self.dx
        display_nt = self.display_T / self.dt

        data = self.process_data[int(indexTime): int(indexTime+display_nt)]

        # data = data / np.nanmax(data, axis=0)
        # nan_max = np.nanmax(data, axis=0)
        # nan_mean = np.nanmean(data, axis=0)
        # data = data - nan_mean
        nan_max = np.nanmax(np.abs(data), axis=0)
        data = data / nan_max

        nt, nx = data.shape

        x = np.arange(0, nx) * dx
        t = np.arange(int(indexTime), int(indexTime+display_nt)) * dt
        # scale = self.color_value/self.scale

        for i in range(nx):
            ax.plot(t, 0.8*data[:, i]+i, 'k', lw=0.5)

        ax.set_xlabel('time (s)')
        ax.set_ylabel('channel')
        fname = os.path.basename(self.fname).split('.')[0]
        ax.set_title(fname)
        return ax




    def getSlope(self, tt, xx, mode='sklearn'):
        tt = tt.reshape(-1, 1)
        if mode == 'sklearn':
            model = LinearRegression()
            model.fit(tt, xx)
            slope = model.coef_[0]
            # intercept = model.intercept_
        elif mode == 'scipy':
            slope = linregress(tt.flatten(), xx.flatten()).slope
        elif mode == 'numpy':
            slope, intercept = np.polyfit(tt.flatten(), xx.flatten(), 1)
            # print('斜率: ', slope)
            # print('截距: ', intercept)

        return slope


    def getMaskXY(self, result, mode='sklearn'):
        T = np.linspace(0, 60, 512)
        X = np.linspace(0, 150, 512)
        vels = []
        if result.masks is None:
            return vels
        for mask in result.masks:
            mask = mask.data.cpu().numpy()
            _, x, y = np.where(mask == 1)
            tt = T[x]
            xx = X[y]
            slope = self.getSlope(tt, xx, mode=mode)

            vel = slope*3.6
            vels.append(vel)

        return vels

    
    def imshowYolo(self, fig):
        ax1 = fig.add_subplot(121)
        img = Image.open(self.yolo_figname)
        img = np.array(img)
        ax1.imshow(img, aspect='auto', extent=[self.profile_X[0], self.profile_X[-1], self.nt*self.dt, 0])
        ax1.set_xlabel('Distance (m)')
        ax1.set_ylabel('Time (s)')
        ax1.set_title('Velocity')
        
        result = self.model(self.yolo_figname, nms=True, iou=0.75, conf=0.65)[0]
        vels = self.getMaskXY(result)

        masks = result.masks.data.cpu().numpy()
        masks_len = len(masks)
        bb = np.zeros_like(masks[0])
        for ii in range(masks_len):
            masks[ii][masks[ii] > 0] = vels[ii]
            bb[masks[ii] > 0] = vels[ii]

        # TODO:
        # aa = np.sum(masks, axis=0)
        aa = bb
        # bb = np.zeros_like(masks[0])
        # for ii in range(masks_len):
        #     cc = bb.copy()
        #     dd = masks[ii][masks[ii] > 0]

        matrix_filtered = np.where(aa > 0, aa, np.nan)

        ax2 = fig.add_subplot(122)
        # 设置归一化的范围为20到100
        norm = colors.Normalize(vmin=30, vmax=55)
        # 选择一个色彩映射
        cmap = plt.cm.gnuplot2

        im = ax2.imshow(matrix_filtered, cmap=cmap, norm=norm, aspect='auto', extent=[self.profile_X[0], self.profile_X[-1], self.nt*self.dt, 0])
        ax2.set_xlabel('Distance (m)')
        ax2.set_ylabel('Time (s)')
        ax2.set_title('Velocity')

        # 对色阶条独立设置位置和大小
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cb = plt.colorbar(im, cax=cax)
        cb.set_label('Velocity (km/h)')

        fig.tight_layout()

        return fig

