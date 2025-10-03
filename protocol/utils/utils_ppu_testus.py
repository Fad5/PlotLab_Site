    


# from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import pandas as pd
from scipy.signal import find_peaks


def alignYaxes(axes, align_values=None):
    '''Align the ticks of multiple y axes

    Args:
        axes (list): list of axes objects whose yaxis ticks are to be aligned.
    Keyword Args:
        align_values (None or list/tuple): if not None, should be a list/tuple
            of floats with same length as <axes>. Values in <align_values>
            define where the corresponding axes should be aligned up. E.g.
            [0, 100, -22.5] means the 0 in axes[0], 100 in axes[1] and -22.5
            in axes[2] would be aligned up. If None, align (approximately)
            the lowest ticks in all axes.
    Returns:
        new_ticks (list): a list of new ticks for each axis in <axes>.

        A new sets of ticks are computed for each axis in <axes> but with equal
        length.
    '''
    from matplotlib.pyplot import MaxNLocator

    nax=len(axes)
    ticks=[aii.get_yticks() for aii in axes]
    if align_values is None:
        aligns=[ticks[ii][0] for ii in range(nax)]
    else:
        if len(align_values) != nax:
            raise Exception("Length of <axes> doesn't equal that of <align_values>.")
        aligns=align_values

    bounds=[aii.get_ylim() for aii in axes]

    # align at some points
    ticks_align=[ticks[ii]-aligns[ii] for ii in range(nax)]

    # scale the range to 1-100read_ecofizika
    ranges=[tii[-1]-tii[0] for tii in ticks]
    lgs=[-np.log10(rii)+2. for rii in ranges]
    igs=[np.floor(ii) for ii in lgs]
    log_ticks=[ticks_align[ii]*(10.**igs[ii]) for ii in range(nax)]

    # put all axes ticks into a single array, then compute new ticks for all
    comb_ticks=np.concatenate(log_ticks)
    comb_ticks.sort()
    locator=MaxNLocator(nbins='auto', steps=[1, 2, 2.5, 3, 4, 5, 8, 10])
    new_ticks=locator.tick_values(comb_ticks[0], comb_ticks[-1])
    new_ticks=[new_ticks/10.**igs[ii] for ii in range(nax)]
    new_ticks=[new_ticks[ii]+aligns[ii] for ii in range(nax)]

    # find the lower bound
    idx_l=0
    for i in range(len(new_ticks[0])):
        if any([new_ticks[jj][i] > bounds[jj][0] for jj in range(nax)]):
            idx_l=i-1
            break

    # find the upper bound
    idx_r=0
    for i in range(len(new_ticks[0])):
        if all([new_ticks[jj][i] > bounds[jj][1] for jj in range(nax)]):
            idx_r=i
            break

    # trim tick lists by bounds
    new_ticks=[tii[idx_l:idx_r+1] for tii in new_ticks]

    # set ticks for each axis
    for axii, tii in zip(axes, new_ticks):
        axii.set_yticks(tii)

    return new_ticks

def read_ecofizika(file, axes):
    """Reads data from Ecofizika (Octava)"""
    vibration = pd.read_csv(file, sep='\t', encoding='mbcs', header=None, names=axes,
                          dtype=np.float32,
                          skiprows=4, usecols=range(1,len(axes)+1)).reset_index(drop=True)
    inf = pd.read_csv(file, sep=' ', encoding='mbcs', header=None, names=None,
                     skiprows=2, nrows=1).reset_index(drop=True)
    fs = int(inf.iloc[0, -1])
    return vibration, fs

def find_res_width2(array, freqs, peak_pos):
    """ finds resonance width according to GOST R 56801-2015 """
    n = 2**0.5

    # left border
    for idx in range(peak_pos,0,-1):
        if array[idx] <= array[peak_pos]/n:
            break
    f1 = freqs[idx] + (array[peak_pos]/n - array[idx])/(array[idx+1] - array[idx])*(freqs[idx+1] - freqs[idx])
    
    # right border
    for idx in range(peak_pos, len(array)-1):
        if array[idx] <= array[peak_pos]/n:
            break
    f2 = freqs[idx-1] + (array[peak_pos]/n - array[idx-1])/(array[idx] - array[idx-1])*(freqs[idx] - freqs[idx-1])
            
    return f1, f2


def vibraTableOne(name, files, a, b, h, heights, loads, axes=['2','1'], NFFT=2048, overlap=256, limits=(0, 200), left_lim=5):
    images = {}
    datas = {}
    results = {}

    S = a*b *1e-6  # sampe area (m2)
    
    fig1, axs1 = plt.subplots(2, 1, figsize=(10, 8), tight_layout=True)
    axs1[0].set_title('Передаточная функция')
    axs1[0].set_ylabel('Модуль передаточной функции')
    axs1[0].set_xlabel('Частота, Гц')
    axs1[0].grid(visible='True', which='both', axis='both', ls='--')

    axs1[1].set_title('Эффективность виброизоляции')
    axs1[1].set_ylabel('Эффективность, дБ')
    axs1[1].set_xlabel('Частота, Гц')
    axs1[1].grid(visible='True', which='both', axis='both', ls='--')


    
    
    
    # Fpeaks, Eds, damps = {}, {}, {}
    # all_freqs, all_TR1mean, all_Lmean = {}, {}, {}
    
    
    for M in loads:
        

        project_name = name + '_' + str(M) + 'кг'   # used for saving, e.g. '44.1_5кг'
        
        _h = heights[loads.index(M)] *1e-3   # sample height (m)
        
        vibration_list, fs = read_ecofizika(files[loads.index(M)], axes)
        rms1 = np.sqrt(np.mean(np.square(vibration_list['1'])))
        rms2 = np.sqrt(np.mean(np.square(vibration_list['2'])))
        
        # if rms1 < rms2 :
        #     vibration_list.columns = ['2','1']
        
        # make fft for Dagestan PPU
        Pxx = {}
        freqs_ = {}
        # w = np.hamming(NFFT)
        with plt.ioff():
            for ax in axes:

                fig2, axs = plt.subplots(figsize=(10, 5), tight_layout=True)

                y = vibration_list[ax].iloc[0*fs:]

                # spectrogram
                Pxx[ax], freqs_[ax], _, _ = axs.specgram(y, NFFT=NFFT, Fs=fs, 
                                                            mode='magnitude',
                                                            noverlap=overlap, cmap='viridis')

            
        last_index = int(limits[1] / freqs_['1'][1])
        freqs = freqs_['1'][1:last_index]
        left_lim_idx = np.argmax(freqs>left_lim)   # for finding peak located at frequency greater then 3 Hz

        TR1 = np.mean(Pxx['2'][1:last_index] / Pxx['1'][1:last_index], axis=1)
        TR = np.mean(Pxx['1'][1:last_index] / Pxx['2'][1:last_index], axis=1)
        TR1mean = pd.Series(TR1).rolling(10, min_periods=1, center=True).mean()
        TRmean = pd.Series(TR).rolling(10, min_periods=1, center=True).mean()

        L = 20*np.log10(TR)
        Lmean = 20*np.log10(TRmean)
        
        datas[M] = (freqs, TR1mean, Lmean)
        
        axs1[0].plot(freqs, TR1mean, label=f'{M} кг')
        axs1[1].plot(freqs, Lmean, label=f'{M} кг')

        
        fig2, axs = plt.subplots(2, 1, figsize=(10, 8), tight_layout=True)
        
        axs[0].plot(freqs, TR1)
        axs[0].plot(freqs, TR1mean)
        axs[0].set_title('Передаточная функция')
        axs[0].set_ylabel('Модуль передаточной функции')
        axs[0].set_xlabel('Частота, Гц')
        axs[0].grid(visible='True', which='both', axis='both', ls='--')
#             axs[0].set_xlim([0, 200])


        axs[1].plot(freqs, L)
        axs[1].plot(freqs, Lmean)
        
        axs[1].set_title('Эффективность виброизоляции')
        axs[1].set_ylabel('Эффективность, дБ')
        axs[1].set_xlabel('Частота, Гц')
        axs[1].grid(visible='True', which='both', axis='both', ls='--')
#             axs[1].set_xlim([0, 200])
        plt.close()
        
        try:
            max1 = TR1mean[left_lim_idx:].max()
            f_peaks = find_peaks(TR1mean[left_lim_idx:], distance=100, prominence=0.1*max1)
#             f_height = -1*f_peaks[1]['peak_heights']
            f_peak_pos = f_peaks[0][0]+left_lim_idx
            Fpeak = freqs[f_peak_pos]
            
            axs[0].plot(Fpeak, TR1mean[f_peak_pos], "o", mfc='none', color = "r", linewidth=3, )

            f1, f2 = find_res_width2(TR1mean, freqs, f_peak_pos)
            if f1 < 0:
                raise ValueError('f1 < 0')
            damp = (f2 - f1) / Fpeak
            Ed = 4*np.pi**2 * Fpeak**2 *float(M) *_h / S *1e-6  # dynamic modulus of elasticity
            pressure = (float(M)/a/b * 9.81 *1e3)
            results[M] = (pressure ,Fpeak, Ed, damp)
            
            
            # axs[0].axhline(y=TR1mean[f_peak_pos].values/2**0.5, color="purple", linestyle="--", linewidth=0.5)
            axs[0].axvline(x=f1, color="black", linestyle="--", linewidth=0.5)
            axs[0].axvline(x=f2, color="black", linestyle="--", linewidth=0.5)
            axs[0].annotate(f"Dynamic modulus of elasticity = {Ed:.5f} MPa\n"\
                            f"Damping = {damp:.5f}\n"\
                            f"Frequency = {Fpeak:.2f} Hz",
                            xy=(Fpeak, TR1mean[f_peak_pos]),
                            xytext=(10, 0), textcoords="offset points",
                            horizontalalignment="left",
                            verticalalignment="center"
                            )
            
            axs[1].plot(Fpeak, Lmean[f_peak_pos], "o", mfc='none', color = "r", linewidth=3, )
            
        except IndexError:
            import warnings
            warnings.warn(f'Не найден пик при нагрузке {M} кг, или другая проблема с индексацией при этой нагрузке', UserWarning)
        except ValueError as err:
            print(err)
        
        images[f'{project_name}.png'] = fig2
        
        """
            ### Add data from Sylomer SR11 to pictures
        
            SR11_list, SRfs = read_ecofizika(SR11[loads.index(M)], axes=['1','2'])
        
            SR_h = heights.loc['SR11', :].iloc[loads.index(M)] *1e-3   # sample height (m)
            SR_a = sizes.loc['SR11'].iloc[0] *1e-3
            SR_b = sizes.loc['SR11'].iloc[1] *1e-3
            SR_S = SR_a*SR_b   # sampe area (m2)
        
            SRPxx = {}
            with plt.ioff():
                for ax in axes:

                    fig2, axs = plt.subplots(figsize=(10, 5), tight_layout=True)

                    y = SR11_list[ax].iloc[20*SRfs:]
                
                    # make fft for Sylomer
                    SRPxx[ax], _, _, _ = axs.specgram(y, NFFT=NFFT, Fs=SRfs, 
                                                      mode='magnitude',
#                                                       window=w,
                                                      noverlap=overlap, cmap='viridis')
        
            SR_TR1mean = np.mean(SRPxx['2'][1:last_index] / SRPxx['1'][1:last_index], axis=1)
            SR_TRmean = np.mean(SRPxx['1'][1:last_index] / SRPxx['2'][1:last_index], axis=1)
        
            SR_Lmean = 20*np.log10(SR_TRmean)
        
            fig3, axs = plt.subplots(2, 1, figsize=(10, 8), tight_layout=True)
        
            axs[0].plot(freqs, SR_TR1mean, label='SR11')
            axs[0].plot(freqs, TR1mean, label=f'ППУ-{name}')
#             axs[0].plot(freqs, TR1mean, label=f'{name}')
            axs[0].set_title('Передаточная функция')
            axs[0].set_ylabel('Модуль передаточной функции')
            axs[0].set_xlabel('Частота, Гц')
            axs[0].legend(loc='upper right', fontsize=10, frameon=True)
            axs[0].grid(visible='True', which='both', axis='both', ls='--')
#             axs[0].set_xlim([0, 200])


            axs[1].plot(freqs, SR_Lmean, label='SR11')
            axs[1].plot(freqs, Lmean, label=f'ППУ-{name}')
        
            axs[1].set_title('Эффективность виброизоляции')
            axs[1].set_ylabel('Эффективность, дБ')
            axs[1].set_xlabel('Частота, Гц')
            axs[1].legend(loc='lower right', fontsize=10, frameon=True)
            axs[1].grid(visible='True', which='both', axis='both', ls='--')
#             axs[1].set_xlim([0, 200])
        
            plt.savefig(f'{project_name}+SR11.png', dpi=300)
        """
        
    axs1[0].legend(loc='upper right', fontsize=10, frameon=True)
    axs1[1].legend(loc='lower right', fontsize=10, frameon=True)
    
    try:
        if len(results) >= 2:
            fig3, axs3 = plt.subplots(1, 1, figsize=(10, 5), tight_layout=True)
            fig3.subplots_adjust(right=0.75)

            twin1 = axs3.twinx()
            twin2 = axs3.twinx()
            twin2.spines.right.set_position(("axes", 1.1))
            
            pressures = [float(load) / S * 9.81 *1e-3 for load in loads]
            all_heights = [h]
            all_heights.extend(heights)

            p1, = axs3.plot(range(len(results)+1), all_heights, "C0", linewidth=2, label="Толщина")
            p2, = twin1.plot(range(1, len(results)+1), [results[load][1] for load in loads], "C1", linewidth=2, label="Динамический модуль упругости")
            p3, = twin2.plot(range(1, len(results)+1), [results[load][2] for load in loads], "C2", linewidth=2, label="Коэффициент демпфирования")

            axs3.set(xlim=(0, len(results)), xlabel="Нагрузка, кг (кПа)", ylabel="Толщина, мм")
            # axs3.set_ylim(bottom=0)
            twin1.set(ylabel="Динамический модуль упругости, МПа")
            # twin1.set_ylim(bottom=0)
            twin2.set(ylabel="Коэффициент демпфирования")
            # twin2.set_ylim(bottom=0)

            axs3.yaxis.label.set_color(p1.get_color())
            twin1.yaxis.label.set_color(p2.get_color())
            twin2.yaxis.label.set_color(p3.get_color())
            
            
            axs3.set_xticks(range(len(results)+1))
            xlabels = [0]
            xlabels.extend([f'{load} ({pressure:.2f})' for load, pressure in zip(loads, pressures)])
            axs3.set_xticklabels(xlabels)

            axs3.tick_params(axis='y', colors=p1.get_color())
            axs3.grid(visible='True', which='both', axis='both', ls='--')
            twin1.tick_params(axis='y', colors=p2.get_color())
            # twin1.grid(visible='True', which='major', axis='y', ls='--', color=p2.get_color())
            twin2.tick_params(axis='y', colors=p3.get_color())
            # twin2.grid(visible='True', which='major', axis='y', ls='--', color=p3.get_color())
            alignYaxes((axs3,twin1,twin2))

            axs3.legend(handles=[p1, p2, p3])
            plt.close()
        
        images[f'{name}.png'] = fig1
        
    except (IndexError,KeyError) as err:
        import warnings
        warnings.warn(f'Не найден пик, сводный график не будет построен', UserWarning)
        print(err)
        
    try:
        images[f'{name}_rez.png'] = fig3
    except UnboundLocalError:
        pass
    
    return (images, datas, results)