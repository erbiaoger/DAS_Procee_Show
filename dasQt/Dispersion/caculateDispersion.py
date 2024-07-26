import numpy as np
from tqdm import tqdm
from . import DAS_module
from .dispersion import get_dispersion

class CaculateDispersion():
    def __init__(self, MyProgram):
        self.MyProgram = MyProgram
        self.__dict__.update(self.MyProgram.__dict__)
        # self.dt = self.MyProgram.dt
        # self.dx = self.MyProgram.dx
        # self.nt = self.MyProgram.nt
        # self.profile_X = self.MyProgram.profile_X
        
        
        pass
    
    def readNextData(self):
        self.MyProgram.readNextData()
        self.__dict__.update(self.MyProgram.__dict__)

    def caculateCC(self, dispersion_parse=None):
        """dasQt/das.py made by Zhiyu Zhang JiLin University in 2024-01-05 17h.
        Parameters
        ----------
        sps : float, (1/self.dt)
            current sampling rate
        samp_freq : float, (1/self.dt)
            targeted sampling rate
        freqmin : float, (0.1)
            pre filtering frequency bandwidth预滤波频率带宽
        freqmax : float, (10)
            note this cannot exceed Nquist freq
        freq_norm : str, ('rma')
            'no' for no whitening, or 'rma' for running-mean average, 'phase_only' for sign-bit normalization in freq domain.
        time_norm : str, ('one_bit')
            'no' for no normalization, or 'rma', 'one_bit' for normalization in time domain
        cc_method : str, ('xcorr')
            'xcorr' for pure cross correlation, 'deconv' for deconvolution;
        smooth_N : int, (5)
            moving window length for time domain normalization if selected (points)
        smoothspect_N : int, (5)
            moving window length to smooth spectrum amplitude (points)
        maxlag : float, (0.5)   
            lags of cross-correlation to save (sec)
        max_over_std : float, (10**9)   
            threahold to remove window of bad signals: set it to 10*9 if prefer not to remove them
        cc_len : float, (5)
            correlate length in second(sec)
        cha1 : int, (31)
            start channel index for the sub-array
        cha2 : int, (70)
            end channel index for the sub-array

        Returns
        -------
        data_liner : ndarray
            DESCRIPTION.
        """

        # ---------------------------input parameters---------------------------------#
        # FOR "COHERENCY" PLEASE set freq_norm to "rma", time_norm to "no" and cc_method to "xcorr"
        dt                   = self.dt
        sps                  = 1/dt                             # current sampling rate
        samp_freq            = 1/dt                             # targeted sampling rate
        cha1                 = np.abs(float(dispersion_parse['cha1']) - self.profile_X).argmin()
        cha2                 = np.abs(float(dispersion_parse['cha2']) - self.profile_X).argmin()
        self.dispersion_cha1 = cha1
        self.dispersion_cha2 = cha2
        # self.logger.debug(f"cha1: {cha1}, cha2: {cha2}")

        cc_len               = dispersion_parse['cc_len']       # correlate length in second(sec)
        maxlag               = dispersion_parse['maxlag']       # lags of cross-correlation to save (sec)
        data                 = self.data[:, cha1:cha2].copy()
        cha_list             = np.array(range(cha1, cha2))
        nsta                 = len(cha_list)
        n_pair               = int(nsta*(data.shape[0]*dt//cc_len))
        n_lag                = int(maxlag * samp_freq * 2 + 1)

        corr_full            = np.zeros([n_lag, n_pair], dtype=np.float32)
        stack_full           = np.zeros([1, n_pair], dtype=np.int32)

        prepro_para     = { 
            'sps'          : 1/dt,                              # current sampling rate
            'npts_chunk'   : cc_len * sps,                      # correlate length in second(sec)
            'nsta'         : nsta,                              # number of stations
            'cha_list'     : cha_list,                          # channel list
            'samp_freq'    : 1/dt,                              # targeted sampling rate
            'freqmin'      : dispersion_parse['freqmin'],       # pre filtering frequency bandwidth
            'freqmax'      : dispersion_parse['freqmax'],       # note this cannot exceed Nquist freq
            'freq_norm'    : dispersion_parse['freq_norm'],     # 'no' for no whitening, or 'rma' for running-mean average, 'phase_only' for sign-bit normalization in freq domain.
            'time_norm'    : dispersion_parse['time_norm'],     # 'no' for no normalization, or 'rma','one_bit' for normalization in time domain
            'cc_method'    : dispersion_parse['cc_method'],     # 'xcorr' for pure cross correlation, 'deconv' for deconvolution;
            'smooth_N'     : dispersion_parse['smooth_N'],      # moving window length for time domain normalization if selected (points)
            'smoothspect_N': dispersion_parse['smoothspect_N'], # moving window length to smooth spectrum amplitude (points)
            'maxlag'       : dispersion_parse['maxlag'],        # lags of cross-correlation to save (sec)
            'max_over_std' : dispersion_parse['max_over_std']   # threahold to remove window of bad signals: set it to 10*9 if prefer not to remove them
        }

        # -------------------------------------------------开始计算---------------------------------------------------
        mm1  = data.shape[0]
        mm   = int(mm1*dt//cc_len)
        mm10 = int(1/dt*cc_len)
        for imin in tqdm(range(1,mm+1), desc="Processing", ncols=100):
            tdata=data[((imin-1)*mm10):imin*mm10, :]

            # preprocess the raw data
            trace_stdS, dataS = DAS_module.preprocess_raw_make_stat(tdata ,prepro_para)
            # do normalization if needed
            white_spect = DAS_module.noise_processing(dataS, prepro_para)
            Nfft        = white_spect.shape[1]
            Nfft2       = Nfft                                             // 2
            data_white  = white_spect[:, :Nfft2]
            del dataS, white_spect
            # find the good data
            ind = np.where((trace_stdS < prepro_para['max_over_std']) &
                        (trace_stdS > 0) &
                        (np.isnan(trace_stdS) == 0))[0]
            if not len(ind):
                raise ValueError('the max_over_std criteria is too high which results in no data')
            sta         = cha_list[ind]
            white_spect = data_white[ind]

            iiS = int(dispersion_parse['iShot'])

            # smooth the source spectrum
            sfft1 = DAS_module.smooth_source_spect(white_spect[iiS], prepro_para)
            # correlate one source with all receivers
            corr, tindx = DAS_module.correlate(sfft1, white_spect, prepro_para, Nfft)

            # update the receiver list
            # tsta         = sta[iiS:]
            # receiver_lst = tsta[tindx]
            # iS           = int((cha2 * 2 - cha1 - sta[iiS] + 1) * (sta[iiS] - cha1) / 2)

            # stacking one minute
            corr_full[:, (imin-1)*nsta:imin*nsta] += corr.T

        data_liner = np.zeros((n_lag, nsta))
        for iiN in range(0, mm):
            data_liner = data_liner + corr_full[:, iiN * nsta:(iiN + 1) * nsta]
        data_liner = data_liner / mm
        data_liner = self.normalize_data(data_liner)
        
        # return allstacks1, data_liner
        return data_liner

    def selfCaculateCC(self, dispersion_parse):
        scale = 0.5
        data_liner = self.caculateCC(dispersion_parse=dispersion_parse)
        
        self.cc = data_liner
        if self.indexClick == 0:
            self.pre_dispersion_data  = data_liner
            self.indexClick          += 1
            self.CClist.append(data_liner)
        else:
            if self.pre_dispersion_data.shape != data_liner.shape:
                self.pre_dispersion_data = data_liner
                self.indexClick          = 0
                self.CClist = []
                # self.logger.debug(f"dp para changed, last index is {self.indexClick}, now index is 0")
            else:
                
                # self.pre_dispersion_data  = (self.pre_dispersion_data*self.indexClick + data_liner) / (self.indexClick+1)
                self.CClist.append(data_liner)
                
                self.indexClick          += 1
                # self.logger.debug(f"now index is {self.indexClick}")

    def caculateCCAll(self) -> None:
        """caculate dispersion for all files with stack"""

        dt = self.dt
        
        data = np.array(self.CClist)
        print(data.shape)
        n, nt, nx = data.shape
        data = data.reshape(n, -1)

        stack_para = {
            'samp_freq': 1/dt,
            'npts_chunk': nt,
            'nsta': nx,
            'stack_method': 'linear'
        }

        allstacks1 = DAS_module.stacking(data,stack_para)
        nt, nx = allstacks1.shape
        self.ccAll = allstacks1

    def normalize_data(self, data):
        # find the maximum value of each column
        max_values = np.max(np.abs(data), axis=0)
        data_norm  = np.zeros(data.shape)
        for i in range(0, data.shape[0]):
            data_norm[i, :] = data[i, :] / max_values
        # return the normalized data
        return data_norm

    def saveCC(self, filename) -> None:
        """save CC data"""
        np.savez(filename, 
                    data=self.ccAll, 
                    x=self.profile_X[self.dispersion_cha1:self.dispersion_cha2],
                    t=np.linspace(-self.ccAll.shape[0]*self.dt/2, self.ccAll.shape[0]*self.dt/2, self.ccAll.shape[0]))

        # self.logger.info(f"Save CC Done! {filename}")

    def saveDispersion(self, filename) -> None:
        """save dispersion data"""
        np.savez(filename, 
                    data=self.dispersionAll, 
                    f=self.f,
                    c=self.c)

        # self.logger.info(f"Save Dispersion Done! {filename}")

    # TODO: CC Data get Up
    def getUpCC(self, up) -> None:
        """get up cross-correlation"""
        self.bool_upcc = up

    def getDownCC(self, down) -> None:
        """get down cross-correlation"""
        self.bool_downcc = down



    def selfGetDispersion(self, cmin=10., cmax=1000.0, dc=5., fmin=4.0, fmax=18.0, df=0.1, bool_all=False):
        if bool_all:
            selfcc = self.ccAll
        else:
            selfcc = self.cc
        
        if self.bool_downcc:
            nt, nx = selfcc.shape
            cc = selfcc[:nt//2, :]
        elif self.bool_upcc:
            nt, nx = selfcc.shape
            cc = selfcc[nt//2:, :]
        else:
            cc = selfcc

        f, c, img, U, t = get_dispersion(cc, self.dx, self.dt, 
                                             cmin, cmax, dc, fmin, fmax, df)
        
        for i in range(img.shape[1]):
            img[:,i] /= np.max(img[:,i])
        
        return f, c, img**2, U, t



    def imshowCC(self, ax, bool_all=False):
        if bool_all:
            cc = self.ccAll
        else:
            cc = self.cc

        cha1 = self.profile_X[self.dispersion_cha1]
        cha2 = self.profile_X[self.dispersion_cha2]
        # nt   = self.pre_dispersion_data.shape[0]
        nt = self.cc.shape[0]

        ax0 = ax.imshow(cc, cmap='RdBu_r', aspect='auto', interpolation='none',
                  origin='lower', extent=[cha1, cha2, -nt//2*self.dt, nt//2*self.dt])

        ax.set_xlabel('distance (m)')
        ax.set_ylabel('time (s)')
        # ax.set_title('Cross-correlation')
        # fig.colorbar(ax0, ax=ax)
        # self.logger.info("ImShow Cross-correlation Done!")
        return ax

    def imshowDispersion(self, ax, cmin, cmax, dc, fmin, fmax, df, bool_all=False):
        f, c, img, U, t = self.selfGetDispersion(cmin, cmax, dc, fmin, fmax, df, bool_all)
        if bool_all:
            self.dispersionAll = img
            self.f = f
            self.c = c
        
        bar2 = ax.imshow(img,aspect='auto',origin='lower',extent=(f[0],f[-1],c[0],c[-1]), 
                cmap='jet', interpolation='quadric')

        ax.grid(linestyle='--',linewidth=2)
        ax.set_xlabel('Frequency (Hz)', fontsize=14)
        ax.set_ylabel('Phase velocity (m/s)', fontsize=14)
        # ax.tick_params(axis = 'both', which = 'major', labelsize = 14)
        # ax.tick_params(axis = 'both', which = 'minor', labelsize = 14)