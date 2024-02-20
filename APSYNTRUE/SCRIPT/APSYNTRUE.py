#############################################################################
#
#    APSYNTRUE: A pedagogical tool for Radio Interferometry Analysis
#
#    Copyright (C) 2020  Ivan Marti-Vidal (University of Valencia, Spain)
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#############################################################################
###
### DISCLAIMER: This software is intended to be used for educational purposes.
###             It has not been assessed to produce output of scientific level.
#############################################################################


#import Tkinter
#import FileDialog
import tkinter as Tk

import matplotlib as mpl
mpl.use('TkAgg')
import numpy as np
import pylab as pl
import scipy.ndimage.interpolation as spndint
import scipy.optimize as spfit
from matplotlib import cm
import matplotlib.image as plimg

from tkinter import scrolledtext as ScrolledText
from astropy.io import fits as pf
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import time as dt 
import cartopy.crs as ccrs
from PIL import Image


plt = mpl.pyplot




from matplotlib.backend_bases import NavigationToolbar2
from tkinter import filedialog as tkFileDialog
from tkinter.messagebox import showinfo
import os
import time
import sys


__version__ = '2.5b'



__help_text__ = """ 

     APSYNTRUE, A PEDAGOGICAL TOOL FOR RADIO-INTERFEROMETRY IMAGING

                           IVAN MARTI-VIDAL 
                       (UNIVERSITY OF VALENCIA)
                    
You can load "uvfits" files with visibilities from any interferometer. If
data are split in several frequency windows, the program will average all
frequency channels when the data are load. Therefore, only continuum 
imaging is supported and the fractional bandwidth is assumed to be narrow.

The main program window shows the antenna distribution on the Earth, as 
seen from the source, the UV coverage, the PSF and the dirty image. It 
can also show a 2D plot of the data with several axis combinations (e.g., 
UV distance, time, etc.).

The user can select whe pixel size (in units of the Nyquist sampling), 
the "Robustness" parameter of the Briggs visibility weighting, and the 
power index of the visibility weights. 

The program has two independent deconvolution algorithms implemented: 
CLEAN (only minor cycles; Hogbom implementation) and Maximum Entropy
(applied to gridded visibilities; experimental).

Send any comments, suggestions and/or bug reports to:

i.marti-vidal@uv.es 


Enjoy!


"""



__CLEAN_help_text__ = """ 

APSYNTRUE - CLEAN GUI

CLEAN is the most simple implementation of a deconvolution algorithm.
It basically subtracts iteratively a version of the PSF that is shifted
to the peak of the "dirty image" and scaled to a fraction of the 
intensity value at that peak. More information is given in the original
paper from Hogbom (1974, A&AS, 15, 417).


You can add CLEAN masks by clicking and dragging the mouse over the 
RESIDUALS plot. Hitting 'F' (when the mouse cursor is on the RESIDUALS
plot) toggles between the 'mask add' and 'mask remove' modes, which 
are self-explanatory.

The CLEAN gain and number of (minor) iterations can be changed in the 
text boxes. Pressing the "CLEAN" button executes the iterations, 
refreshing all images in real time. You can further click on "CLEAN", 
to continue deconvolving. The box "Thres" is the CLEAN threshold 
(in Jy per beam). Setting it to negative values will allow CLEANing 
negative components.


Pressing RELOAD will undo all the CLEANing and update the images from 
the main window. Therefore, if you change anything in the main program 
window (e.g., robustness parameter), pressing RELOAD will apply these 
changes to the images.

TIP: You can load more than one CLEAN GUI, change anything in the main 
window and press "RELOAD" just in one of the GUIs. This way, you can 
compare directly how the changes you made in the main window will 
affect the CLEANing!

Pressing "+/- Resid" will add (or remove) the residuals from the CLEANed 
image.  By default, the residuals are NOT added (i.e., only the restored
CLEAN components are shown in the CLEAN image).

Pressing "(Un)restore" will restore (or unrestore) the CLEAN model with
the CLEAN beam when plotting. Default status is to apply the restore.

Pressing "Rescale" will rescale the color palette (e.g., to see better
the structure of the residuals).

"""


__MEM_help_text__ = """ 

APSYNTRUE - MAXIMUM ENETROPY METHOD - GUI

The Maximum Entropy Method (MEM) implemented here is an adaptation of the
original algorithm by Cornwell & Evans (1985, A&A 143, 77). Basically, MEM 
is a kind of Regularized Maximum Likelihood (RML) solver, where an image 
is found such that the quadratic difference between its Fourier transform 
and the visibilities is minimum, subject to the condition that the 
"image entropy", i.e. 

S = \sum_i{  I_i*\log{I_i}  }

is maximum. This usually generates rather smooth images, free of the 
PSF sidelobes. The condition of maximum entropy is achieved via the use
of Lagrange multipliers, which can be fine-tunned.

"""



class dataTool(object):

  def quit(self,event=None):
  
    self.tks.destroy()
    sys.exit()

  def __init__(self,antenna_file="",model_file="",tkroot=None):

    self.__version__ = __version__


#############
## World map:
    GREENW = -512; MAXLAT = [-85.,85.]
    MAP = 'WORLD_MAP_SM.png' ; FS = 32

    World = np.copy(np.roll(np.asarray(Image.open(MAP))[::-1,:,0],GREENW,axis=1).astype(float))
    World = np.max(World)-World
    World /= -np.max(World)
    NLAT,NLON = np.shape(World)
    x = np.linspace(0.,360.,NLON)
    y = np.linspace(MAXLAT[0],MAXLAT[1],NLAT)
    xs, ys = np.meshgrid(x,y)
    self.World = {'data':World,'xs':xs,'ys':ys,'Nroll':NLON}
#############



    self.cmaps = ['gist_heat','jet','gray']
    self.cmap = 0

    self.YPLOTS = ['Ampli','Phase','Real','Imag']
    self.XPLOTS = ['UV Dist','Time','U','V']

    self.plotYLabel = ['Amplitude (Jy)', 'Phase (deg.)', 'Real (Jy)', 'Imag (Jy)']
    self.plotXLabel = ['UV Distance (km)', 'UT (h)','U (km)','V (km)']

    self.c = 2.99792458e8
    self.DAlpha = 5

    self.tks = tkroot
    self.tks.protocol("WM_DELETE_WINDOW", self.quit)
    self.deg2rad = np.pi/180.
    self.curzoom = [0,0,0,0]
    self.robust = 0.0
    self.gamma = 0.5
    self.lfac = 1.e6
    self.ulab = r'U (M$\lambda$)'
    self.vlab = r'V (M$\lambda$)'

    pl.set_cmap(self.cmaps[self.cmap])
    self.currcmap = cm.get_cmap() # cm.jet

    self.curzoom = {'beamPlot':[],'dirtyPlot':[],'UVPlot':[],'dataPlot':[]}
    self.pressed = False

    self.flagmode = 0 # 0 -> zoom mode; 1 -> Flag; 2 -> Unflag
    self.fgModes = ['Zoom','Flag','Unflag']    

    self.GUIres = True # Make some parts of the GUI respond to events
    self.antLock = False # Lock antenna-update events

    self.myCLEAN = None  # CLEANer instance (when initialized)
    self.myMEM = None  # MEMer instance (when initialized)


##########################
# Default of defaults!
    Npix = 512   # Image pixel size. Must be a power of 2
#    DefaultData = 'VLBI_simulation.uvfits'
    DefaultData = 'SR1_M87_EHT-2017_101_hi.uvfits'

## Setting to True requires A LOT of RAM for the MEM algorithm, but accelerates
## th iterations quite a bit. If you set it to True, you should work with 
## images with a high Nyquist oversampling (i.e., the sources should look 
## large in the image):
    MemoryHungry = False
##########################



# Overwrite defaults from config file:
    d1 = os.path.dirname(os.path.realpath(sys.argv[0]))
  #  print d1

    try:
      conf = open(os.path.join(d1,'APSYNTRUE.config'))
    except:
      d1 = os.getcwd()
      conf = open(os.path.join(d1,'APSYNTRUE.config'))
      
    for line in conf.readlines():
      temp=line.replace(' ','')
      if len(temp)>2:
         if temp[0:4] == 'Npix':
           Npix = int(temp[5:temp.find('#')])
         if temp[0:11] == 'DefaultData':
           DefaultData = temp[12:temp.find('#')].replace('\'','').replace('\"','')
         if temp[0:12] == 'MemoryHungry':
           MemoryHungry = 'True' in temp[13:temp.find('#')]

    conf.close()

# Set instance configuration values:
    self.Npix = Npix
    self.datadir = os.path.join(d1,'..','DATA')
    self.userdir  = os.path.join(d1,'..','SAVED_IMAGES')
    self.MemoryHungry = MemoryHungry


 # Try to read a default dataset:
    if len(antenna_file)==0:
      try:
        self.data_file = os.path.join(self.datadir,DefaultData)
      except:
        self.showError('Data file cannot be found!')


    self.lock=False

    self.PSF = np.zeros((Npix,Npix))
    self.Dirty = np.zeros((Npix,Npix))
    self.GridUV = np.zeros((Npix,Npix),dtype=np.complex128)
    self.GridUVCov = np.zeros((Npix,Npix),dtype=np.complex128)

    self.GUI()


  def showError(self,message):
    showinfo('ERROR!', message)
    raise Exception(message)


  def _getHelp(self):
    win = Tk.Toplevel(self.tks)
    win.title("Help")
    helptext = ScrolledText.ScrolledText(win)
    helptext.config(state=Tk.NORMAL)
    helptext.insert(Tk.INSERT,__help_text__)
    helptext.config(state=Tk.DISABLED)
    helptext.pack()
    win.grab_set()
    Tk.Button(win, text='OK', command=win.destroy).pack()


  def GUI(self):

    fs = 18

    mpl.rcParams['toolbar'] = 'None'

    self.Nphf = self.Npix/2
    self.robfac = 0.0
    self.figUV = pl.figure(figsize=(9,6))
    self.canvas = FigureCanvasTkAgg(self.figUV, master=self.tks)
    self.canvas.draw()

    menubar = Tk.Menu(self.tks)
    menubar.add_command(label="Load data  ", command=self._loadData)
    menubar.add_command(label="  Help", command=self._getHelp)
    menubar.add_command(label="  Quit", command=self.quit)

    self.tks.config(menu=menubar)
    self.canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

    self.buttons = {}



    butFrame = Tk.Frame(self.tks)
    butFrame.pack(side=Tk.TOP)

    plFrame = Tk.Frame(butFrame)
    plFrame.pack(side=Tk.LEFT)

    menFrame = Tk.Frame(butFrame)
    menFrame.pack(side=Tk.LEFT)
    self.buttons['loadData'] = Tk.Button(menFrame,text="Load Data",command=self._loadData)
    self.buttons['loadData'].pack(side=Tk.TOP)
    self.buttons['getHelp'] = Tk.Button(menFrame,text="Help",command=self._getHelp)
    self.buttons['getHelp'].pack(side=Tk.TOP)
    self.buttons['Quit'] = Tk.Button(menFrame,text="Quit",command=self.quit)
    self.buttons['Quit'].pack(side=Tk.TOP)

    separator = Tk.Frame(butFrame,height=2, bd=5, relief=Tk.SUNKEN)
    separator.pack(fill=Tk.X, padx=5, pady=2,side=Tk.LEFT)

    AxFrame = Tk.LabelFrame(butFrame,text="Plot: ")
    AxFrame.pack(side=Tk.LEFT)

    self.buttons['Y'] = Tk.Listbox(AxFrame,exportselection=False,width=7,height=4)
    self.buttons['Y'].pack(side=Tk.LEFT)
    vs = Tk.Label(AxFrame,text=' vs. ')
    vs.pack(side=Tk.LEFT)
    self.buttons['X'] = Tk.Listbox(AxFrame,exportselection=False,width=7,height=4)
    self.buttons['X'].pack(side=Tk.LEFT)


    separator = Tk.Frame(butFrame,height=2, bd=5, relief=Tk.SUNKEN)
    separator.pack(fill=Tk.X, padx=5, pady=2,side=Tk.LEFT)



    for pli in self.YPLOTS:
      self.buttons['Y'].insert(Tk.END,pli)

    for pli in self.XPLOTS:
      self.buttons['X'].insert(Tk.END,pli)

    self.buttons['Y'].select_set(0)
    self.buttons['X'].select_set(0)

    self.buttons['X'].bind('<<ListboxSelect>>', self._plotData)
    self.buttons['Y'].bind('<<ListboxSelect>>', self._plotData)

    HgFrame = Tk.LabelFrame(butFrame,text="Highlight by: ")
    HgFrame.pack(side=Tk.LEFT)

    self.IsBas = Tk.IntVar()
    self.buttons['BLSel'] = Tk.Checkbutton(HgFrame,text='Baseline ',variable=self.IsBas,command=self._onBas)
    self.buttons['BLSel'].pack(side=Tk.LEFT)

    scrollbar = Tk.Scrollbar(HgFrame, orient=Tk.VERTICAL)
    scrollbar.pack(side=Tk.LEFT,fill=Tk.Y)

    self.buttons['A1'] = Tk.Listbox(HgFrame,exportselection=False,width=5,height=4,yscrollcommand=scrollbar.set)
    scrollbar.config(command=self.buttons['A1'].yview)
    self.buttons['A1'].pack(side=Tk.LEFT)
    vs = Tk.Label(HgFrame,text=' to ')
    vs.pack(side=Tk.LEFT)

    scrollbar2 = Tk.Scrollbar(HgFrame, orient=Tk.VERTICAL)
    scrollbar2.pack(side=Tk.LEFT,fill=Tk.Y)
    self.buttons['A2'] = Tk.Listbox(HgFrame,exportselection=False,width=5,height=4,yscrollcommand=scrollbar2.set)
    scrollbar2.config(command=self.buttons['A2'].yview)    
    self.buttons['A2'].pack(side=Tk.LEFT)
    self.buttons['A2'].insert(Tk.END,'ALL')

    self.buttons['A1'].bind('<<ListboxSelect>>', self._selectBL)
    self.buttons['A2'].bind('<<ListboxSelect>>', self._selectBL)


    ScanAngSel = Tk.Frame(HgFrame)
    ScanAngSel.pack(side=Tk.LEFT,fill=Tk.X)

    UTFram = Tk.Frame(ScanAngSel)
    UTFram.pack(side=Tk.TOP,fill=Tk.X)


    self.buttons['UT0'] = Tk.Scale(UTFram,from_=0,to=24,orient=Tk.HORIZONTAL,length=200,resol=1,command=self._selectUT)
    self.buttons['UT0'].pack(side=Tk.RIGHT)

    self.IsScan = Tk.IntVar(UTFram)
    self.IsBas.set(False)
    self.buttons['ScanSel'] = Tk.Checkbutton(UTFram,text='Obs. time: ',variable=self.IsScan,command=self._onScan)
    self.buttons['ScanSel'].pack(side=Tk.LEFT)



    AngFram = Tk.Frame(ScanAngSel)
    AngFram.pack(side=Tk.TOP,fill=Tk.X)


    self.buttons['Ang'] = Tk.Scale(AngFram,from_=0,to=180,orient=Tk.HORIZONTAL,length=200,resol=self.DAlpha,command=self._selectAng)
    self.buttons['Ang'].pack(side=Tk.RIGHT)

    self.IsAng = Tk.IntVar(AngFram)
    self.IsAng.set(False)
    self.buttons['AngSel'] = Tk.Checkbutton(AngFram,text='UV Angle: ',variable=self.IsAng,command=self._onAng)
    self.buttons['AngSel'].pack(side=Tk.LEFT)





    separator = Tk.Frame(butFrame,height=2, bd=5, relief=Tk.SUNKEN)
    separator.pack(fill=Tk.X, padx=5, pady=2,side=Tk.LEFT)


    ImFrame = Tk.LabelFrame(butFrame,text="Image Reconstruction: ")
    ImFrame.pack(side=Tk.LEFT)


    CLEAN = Tk.Frame(ImFrame)
    CLEAN.pack(side=Tk.LEFT)

    self.buttons['CLEAN'] = Tk.Button(CLEAN,text="CLEAN",command=self._doCLEAN)
    self.buttons['CLEAN'].pack(side=Tk.TOP)
    self.buttons['MEM'] = Tk.Button(CLEAN,text="MEM",command=self._doMEM)
    self.buttons['MEM'].pack(side=Tk.TOP)


    separator = Tk.Frame(ImFrame,height=2, bd=5, relief=Tk.SUNKEN)
    separator.pack(fill=Tk.X, padx=5, pady=2,side=Tk.LEFT)






    ScaleSel = Tk.Frame(ImFrame)
    ScaleSel.pack(side=Tk.LEFT)    


##########
    PixSel = Tk.Frame(ScaleSel)
    PixSel.pack(side=Tk.TOP,fill=Tk.X)


    self.buttons['Pixel'] = Tk.Scale(PixSel,from_=3,to=60,orient=Tk.HORIZONTAL,length=250,resol=1,command=self._setPixel,showvalue=0)
    self.buttons['Pixel'].pack(side=Tk.RIGHT)

    pltxt = Tk.Label(PixSel,text='Pixel (Nyq):')
    pltxt.pack(side=Tk.LEFT)
 

##########
    RobSel = Tk.Frame(ScaleSel)
    RobSel.pack(side=Tk.TOP,fill=Tk.X)


    self.buttons['Robust'] = Tk.Scale(RobSel,from_=-2.0,to=2.0,orient=Tk.HORIZONTAL,length=250,resol=0.1,command=self._reWeight,showvalue=0)
    self.buttons['Robust'].pack(side=Tk.RIGHT)

    pltxt = Tk.Label(RobSel,text='Robustness:')
    pltxt.pack(side=Tk.LEFT)


########## 
    PowSel = Tk.Frame(ScaleSel)
    PowSel.pack(side=Tk.TOP,fill=Tk.X)


    self.buttons['Power'] = Tk.Scale(PowSel,from_=0.0,to=1.0,orient=Tk.HORIZONTAL,length=250,resol=0.1,command=self._rePower,showvalue=0)
    self.buttons['Power'].pack(side=Tk.RIGHT)

    pltxt = Tk.Label(PowSel,text='Wgt. Power:')
    pltxt.pack(side=Tk.LEFT)

    

###########
    LabSel = Tk.Frame(ScaleSel)
    LabSel.pack(side=Tk.TOP,fill=Tk.X)

    self.buttons['ImLab'] = Tk.Label(LabSel,text='Nyquist: %2i; Robust: %2.1f;   Weight: %2.1f'%(3,0.0,1.0))
    self.buttons['ImLab'].pack(side=Tk.LEFT)




    self.UVPlot = self.figUV.add_subplot(232,aspect='equal',facecolor=(0,0,0))
    self.beamPlot = self.figUV.add_subplot(233,aspect='equal')
    self.dataFig = self.figUV.add_axes([0.07,0.07,0.55,0.38])
    self.dataPlot = self.dataFig.plot([],[],'.k',label='Data')[0]
    self.dataPlotHI = self.dataFig.plot([],[],'.r',label='Highlighted')[0]
    self.dataFlagged = self.dataFig.plot([],[],'xm',label='Flagged')[0]
    self.dataFig.legend(numpoints=1,ncol=1,loc='best')

    self.UVCov = [self.UVPlot.plot([],[],'.w',picker=True)[0], self.UVPlot.plot([],[],'.w',picker=True)[0]]
    self.UVCovHI = [self.UVPlot.plot([],[],'.r')[0], self.UVPlot.plot([],[],'.r')[0]]

    self.Box = self.dataFig.plot([0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.],lw=2,color='b')[0]
    self.BoxFg = self.dataFig.plot([0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.],lw=2,color='r')[0]


## PLOT EARTH:


    self.dirtyPlot = self.figUV.add_subplot(236,aspect='equal')


    self.dataTxt = self.dataFig.text(0.02,0.92,'Current Mouse Mode:  %s  (press \'F\' to change)'%self.fgModes[self.flagmode],
                                     transform=self.dataFig.transAxes)

    self.UVInfo = self.UVPlot.text(0.02,0.95,'',
                                     transform=self.UVPlot.transAxes,color = 'w')

    self.dirtyInfo = self.dirtyPlot.text(0.03,0.93,'',
                                     transform=self.dirtyPlot.transAxes,bbox=dict(facecolor='white',alpha=0.7))


    instruct = self.beamPlot.text(0.03,0.05,'\'Z\': zoom in\n\'U\': zoom out',
                                     transform=self.beamPlot.transAxes,bbox=dict(facecolor='white',alpha=0.7))

    instruct = self.UVPlot.text(0.02,0.05,'\'Z\': zoom in\n\'U\': zoom out',
                                     transform=self.UVPlot.transAxes,color = 'w')

    instruct = self.dirtyPlot.text(0.03,0.05,'\'Z\': zoom in\n\'U\': zoom out',
                                     transform=self.dirtyPlot.transAxes,bbox=dict(facecolor='white',alpha=0.7))



    self.figUV.subplots_adjust(left=0.02,right=0.99,top=0.97,bottom=0.07,hspace=0.25)
    self.canvas.mpl_connect('pick_event', self._onPick)
    self.canvas.mpl_connect('button_press_event',self._onButtonPress)
    self.canvas.mpl_connect('button_release_event',self._onButtonRelease)
    self.canvas.mpl_connect('key_press_event', self._onKeyPress)
    self.canvas.mpl_connect('motion_notify_event', self._onDrag)


    self.fmtBas = r'Bas %s $-$ %s  at  UT = %4.2fh'
    self.fmtD = r'% .2e Jy/beam' "\n" r'$\Delta\alpha = $ % 4.2f / $\Delta\delta = $ % 4.2f '
    self.fmtM = r'%.2e Jy/pixel' "\n"  r'$\Delta\alpha = $ % 4.2f / $\Delta\delta = $ % 4.2f'

    self.PSFImage = self.beamPlot.imshow(self.PSF,cmap=self.currcmap)
    self.dirtyImage = self.dirtyPlot.imshow(self.PSF,cmap=self.currcmap)


    self.beamPlot.set_title('PSF',rotation=-90.,x=1.05, y=0.5,fontsize=fs)

    self.dirtyPlot.set_title('DIRTY IMAGE',rotation=-90.,x=1.05, y=0.45,fontsize=fs)
    self.UVPlot.set_title('UV SPACE',rotation=-90.,x=1.05, y=0.45,fontsize=fs)
    self.dataFig.set_title('VISIBILITIES',rotation=0.,x=0.1, y=1.01,fontsize=fs)



    self._loadData(fits_file = self.data_file)
    pl.draw()
    self.canvas.draw()


  def _loadData(self,fits_file=''):

    if len(fits_file)==0:
      fits_file = tkFileDialog.askopenfilename(title='Load UVFITS file...',initialdir=self.datadir)

    if len(fits_file)>0:

      success = False

      if True:
     # try:
        temp = pf.open(fits_file)
        self.snam = temp['PRIMARY'].header['OBJECT']
        if self.snam=='MULTI':
          self.snam = temp['AIPS SU'].data['SOURCE'][0]
          self.ra = temp['AIPS SU'].data['RAEPO'][0]
          self.dec = temp['AIPS SU'].data['DECEPO'][0]
        else:
          self.ra = temp['PRIMARY'].header['OBSRA']
          self.dec = temp['PRIMARY'].header['OBSDEC']

        for key in temp['PRIMARY'].header.keys():
          if temp['PRIMARY'].header[key] == 'FREQ':
            freq = float(temp['PRIMARY'].header['CRVAL'+key[-1]])
            break
        self.date = temp['PRIMARY'].header['DATE-OBS'].split('T')[0]

        self.telesc = ''

        if 'TELESCOP' in temp['PRIMARY'].header.keys():
           self.telesc += temp['PRIMARY'].header['TELESCOP']
        elif 'INSTRUME' in temp['PRIMARY'].header.keys():
           self.telesc += temp['PRIMARY'].header['INSTRUME']

        self.telesc += ' / ' + temp['PRIMARY'].header['OBSERVER']

        ain=-1
        for ti in range(len(temp)):
          if temp[ti].name=='AIPS AN':
            ain=ti
            break

        if ain<0:
          self.showError('BAD UVFITS FILE!')
  
        

        nant = len(temp[ain].data['ANNAME'])

        if self.ra<0.0:
          self.ra += 360.
        RA = self.ra/15.; HH = int(RA); MM = int((RA-HH)*60.); SS = (RA-HH-MM/60.)*3600.
        hh = int(self.dec); mm = int((self.dec-hh)*60.); ss = (self.dec-hh-mm/60.)*3600.
        Ndata = len(temp['PRIMARY'].data['BASELINE'])
        if 'COMMENT' in temp['PRIMARY'].header.keys():
          comments = temp['PRIMARY'].header['COMMENT']
        else:
          comments = 'NONE'

        info =  '\n     METADATA FOR FILE \"%s\": \n\n\n - SOURCE: \n\n     %s located at RA = %02ih%02im%.3fs ; Dec = %02id%02im%.3fs\n\n\n'%(os.path.basename(fits_file),self.snam,RA,MM,SS,hh,mm,ss)
        info += ' - TELESCOPE/OBSERVER: \n\n     %s (%i antennas) oberving at %.2f GHz.\n\n\n - OBSERVATIONS: \n\n     Date of observation: %s ;  There are %i visibilities\n'%(self.telesc,nant,freq/1.e9,self.date,Ndata)
        info += '\n\n - COMMENTS:\n\n'
        for comment in comments:
          info += str(comment)+'\n'
       
        temp.close()
        success = True
      else:
#      except:
        self.showError('BAD UVFITS FILE!')

      if success:

## Draw Earth seen from the correct Declination:
        self.EarthPlot = self.figUV.add_subplot(231,aspect='equal',facecolor=(0,0,0),projection=ccrs.Orthographic(central_longitude=0, central_latitude=self.dec))
        self.EarthMap = self.EarthPlot.pcolormesh(self.World['xs'],self.World['ys'],self.World['data'],transform=ccrs.PlateCarree(),cmap='seismic',vmin=-1.,vmax=1.0) #,animated=True)

        win = Tk.Toplevel(self.tks)
        win.attributes('-topmost', 'true')
        win.title("Current UVFITS File Info")
        infotext = ScrolledText.ScrolledText(win)
        infotext.config(state=Tk.NORMAL)
        infotext.insert(Tk.INSERT,info)
        infotext.config(state=Tk.DISABLED)
        infotext.pack()
        loadProg = Tk.StringVar()
        loadProg.set('Going to load visibilities...')
        progress = Tk.Label(win, textvariable=loadProg)
        progress.config(state=Tk.NORMAL)
        progress.pack()
        win.grab_set()
        self.OKBut = Tk.Button(win, text='OK', command=win.destroy)
        self.OKBut["state"] = Tk.DISABLED
        self.OKBut.pack()
        win.update()
        self.readData(fits_file,win,loadProg)
    else:
      self.showError('NO FILE SELECTED!')

    self.OKBut["state"] = Tk.NORMAL



  def _reWeight(self,value):
     self._gridUV(regrid=False)   
     self.canvas.draw()
     return

  def _rePower(self,value):
     self._gridUV(regrid=False)   
     self.canvas.draw()
     return

  def _setPixel(self,value):
     self._gridUV()   
     self.canvas.draw()
     return



  def _onBas(self):
    self._selectBL(True)
  #  print 'BL: ',self.IsBas.get(), 'Scan: ',self.IsScan.get(), 'Ang: ', self.IsAng.get()
    return


  def _onScan(self):
    self._selectUT(self.buttons['UT0'].get())
  #  print 'BL: ',self.IsBas.get(), 'Scan: ',self.IsScan.get(), 'Ang: ', self.IsAng.get()
    return

  def _onAng(self):
    self._selectAng(self.buttons['Ang'].get())
  #  print 'BL: ',self.IsBas.get(), 'Scan: ',self.IsScan.get(), 'Ang: ', self.IsAng.get()
    return


  def _selectAng(self,value):

    self.UVInfo.set_text('')

    if self.IsAng.get()==1:
      self.IsScan.set(0)
      self.IsBas.set(0)
      self.buttons['ScanSel'].deselect()
      self.buttons['BLSel'].deselect()
    else:
      self.IsAng.set(1)
      self.IsScan.set(0)
      self.IsBas.set(0)
      self.buttons['ScanSel'].deselect()
      self.buttons['BLSel'].deselect()

    self.Mask[:] = np.logical_and(self.Angle >= float(value) -self.DAlpha/2., self.Angle <= float(value) + self.DAlpha)
    self._plotHighlighted()

    return




  def _selectUT(self,value):

     self.UVInfo.set_text('')

     if self.IsScan.get()==1:
       self.IsBas.set(0)
       self.IsAng.set(0)
       self.buttons['BLSel'].deselect()
       self.buttons['AngSel'].deselect()
     else:
       self.IsScan.set(1)
       self.IsBas.set(0)
       self.IsAng.set(0)
       self.buttons['BLSel'].deselect()
       self.buttons['AngSel'].deselect()


     self.Mask[:] = self.UT == self.Utimes[int(value)]
     self._plotEarth(True)

     self._plotHighlighted()

     return





  def _selectBL(self,value):

    self.UVInfo.set_text('')

    if self.IsBas.get()==1:
      self.IsScan.set(0)
      self.IsAng.set(0)
      self.buttons['ScanSel'].deselect()
      self.buttons['AngSel'].deselect()
    else:
      self.IsBas.set(1)
      self.IsAng.set(0)
      self.IsScan.set(0)
      self.buttons['ScanSel'].deselect()
      self.buttons['AngSel'].deselect()

    A1 = int(self.buttons['A1'].curselection()[0])
    A2 = int(self.buttons['A2'].curselection()[0])

    if A2 == self.Nants:
      self.Mask[:] = np.logical_or(self.ant1 == A1+1,self.ant2==A1+1)
    else:  
      Mask1 = np.logical_and(self.ant1 == A2+1,self.ant2 == A1+1)
      Mask2 = np.logical_and(self.ant2 == A2+1,self.ant1 == A1+1)
      self.Mask[:] = np.logical_or(Mask1,Mask2)
      del Mask1, Mask2

    self._plotHighlighted()




  def _plotHighlighted(self):

      self.UVCovHI[0].set_data( self.UV[self.Mask,0], self.UV[self.Mask,1])
      self.UVCovHI[1].set_data(-self.UV[self.Mask,0],-self.UV[self.Mask,1])

      Y = int(self.buttons['Y'].curselection()[0])
      X = int(self.buttons['X'].curselection()[0])

      if Y==0:
        YHi = np.concatenate([self.ampl[self.Mask,i] for i in range(self.NIF)])
      elif Y==1:
        YHi = np.concatenate([self.phase[self.Mask,i] for i in range(self.NIF)])
      elif Y==2:
        YHi = np.concatenate([self.vis.real[self.Mask,i] for i in range(self.NIF)])        
      elif Y==3:
        YHi = np.concatenate([self.vis.imag[self.Mask,i] for i in range(self.NIF)])
      if X==0:
        XHi = np.concatenate([self.Q[self.Mask] for i in range(self.NIF)])
      elif X==1:
        XHi = np.concatenate([self.UT[self.Mask] for i in range(self.NIF)])
      elif X==2:
        XHi = np.concatenate([self.UV[self.Mask,0] for i in range(self.NIF)])        
      elif X==3:
        XHi = np.concatenate([self.UV[self.Mask,1] for i in range(self.NIF)])        

      self.dataPlotHI.set_data(XHi,YHi)
      del XHi, YHi
      pl.draw()
      self.canvas.draw()




  def readData(self,datafile,win=None,loadText=None):

    data = pf.open(datafile)

  #  try:
    if True:

       ain=-1
       for ti in range(len(data)):
         if data[ti].name=='AIPS AN':
           ain=ti
           break

       if ain<0:
         self.showError('BAD UVFITS FILE!')
  
       AN = data[ain]
       FQ = data['AIPS FQ']
       PR = data['PRIMARY']

       self.antnames = list(AN.data['ANNAME'])
       self.antcoords = np.copy(AN.data['STABXYZ'])
       self.antidx = list(AN.data['NOSTA'])
       self.Nants = len(self.antnames)

       self.antLat = 180./np.pi*np.arctan2(self.antcoords[:,2],np.sqrt(self.antcoords[:,0]**2.+self.antcoords[:,1]**2.))
       self.antLon = 180./np.pi*np.arctan2(self.antcoords[:,1],self.antcoords[:,0])

       if 'GSTIA0' in AN.header.keys():
         GST0 = float(AN.header['GSTIA0'])
       else:
         GST0 = 0.0


       if 'FREQ' in AN.header.keys():
         self.Nu0 = float(AN.header['FREQ'])

       if data.__contains__('AIPS SU'):
         isMulti = True
         self.RA = data['AIPS SU'].data['RAEPO'][0]
         self.Dec = data['AIPS SU'].data['DECEPO'][0]
       else:
         isMulti = False


       
       foundNu = False
       foundRA = False
       foundDec = False

       for key in PR.header.keys():

           if foundNu and foundRA and foundDec:
              break

      #     print key, PR.header[key]
           if str(PR.header[key]).startswith('FREQ') and 'SEL' not in PR.header[key]:
               self.Nu0 = float(PR.header['CRVAL'+key[-1]])
#  We rather average over IFs:
               self.freqs = [self.Nu0 + np.average(FQ.data['IF FREQ'])]

# Will rather average over IFs.
               self.NIF = 1
               self.BWTOT = float(self.NIF)*(FQ.data['CH WIDTH'][0])

               foundNu = True

           if not isMulti:
               if str(PR.header[key]).startswith('RA'):
                   self.RA = float(PR.header['CRVAL'+key[-1]])
                   foundRA = True
               if str(PR.header[key]).startswith('DEC'):
                   self.Dec = float(PR.header['CRVAL'+key[-1]])
                   foundDec = True

       NvisTot = len(data['PRIMARY'].data)




## Figure out if there is a sign flip in the antenna coordinates:


       self.UV = []
       self.JDate = []
       self.vis = []
       self.wgt = []
       self.ant1 = []
       self.ant2 = []
       self.Nvis = 0

       DataCols = filter(lambda x: 'PTYPE' in x, data['PRIMARY'].header.keys())
       ids = {'JD':[]}; TIMEREAD = False; PZERO = 0.0
       for dtc in DataCols:
       ##  print dtc, TIMEREAD, PZERO, data['PRIMARY'].header[dtc]
         if data['PRIMARY'].header[dtc].startswith('UU'):
           ids['U'] = int(dtc[5:])-1
         if data['PRIMARY'].header[dtc].startswith('VV'):
           ids['V'] = int(dtc[5:])-1
         if data['PRIMARY'].header[dtc].startswith('DATE'):
           if TIMEREAD:
             PZERO = float(data['PRIMARY'].header['PZERO%i'%(int(dtc[5:]))])
             ids['JD'].append(int(dtc[5:])-1)
           else:
             ids['JD'].append(int(dtc[5:])-1)
             TIMEREAD=True
         if data['PRIMARY'].header[dtc].startswith('BASELINE'):
           ids['B'] = int(dtc[5:])-1

       for i in range(NvisTot):
          temp = data['PRIMARY'].data[i]
          bas = int(temp[ids['B']])
          a1 = bas%256
          a2 = bas/256

         # if i%100==0:
          if win is not None and i%100==0:
            loadText.set('Loading visib %i of %i'%(i+1,NvisTot))
            win.update()

          if a1!=a2:
            visib = np.average(temp[-1][0,0,:],axis=1)
            if np.max(visib[:,0,2])>0:
              self.Nvis += 1
              self.UV.append([temp[ids['U']],temp[ids['V']]])
              tin = np.sum([float(temp[jdi]) for jdi in ids['JD']])+PZERO
              self.JDate.append(tin)
              self.ant1.append(a1)
              self.ant2.append(a2)
              NStk = np.shape(visib)[1]
              if NStk==1:
#                self.vis.append(visib[:,0,0] + 1.j*visib[:,0,1])
#                self.wgt.append(visib[:,0,2])
# Will rather average over IFs:
                self.vis.append([np.average(visib[:,0,0] + 1.j*visib[:,0,1],weights=visib[:,0,2])])
                self.wgt.append([np.sum(visib[:,0,2])])
              else:

#                self.vis.append(0.5*(visib[:,0,0] + 1.j*visib[:,0,1] + visib[:,1,0] + 1.j*visib[:,1,1]))
#                self.wgt.append(visib[:,0,2] + visib[:,1,2])
# Will rather average over IFs:
                self.vis.append([0.5*np.average(visib[:,0,0] + 1.j*visib[:,0,1] + visib[:,1,0] + 1.j*visib[:,1,1],weights=visib[:,0,2] + visib[:,1,2])])
                self.wgt.append([np.sum(visib[:,0,2] + visib[:,1,2])])

       data.close()

  #  except:
    else:    
      self.showError('Problem reading the uvfits file!')


    loadText.set('Arranging data...')
    win.update()



###################
## Figure out if there is a sign flip in the antenna coordinates:
   ## Find first cross-correlation: 
    i0 = 0
    for i in range(self.Nvis):
      if self.ant1[i] != self.ant2[i]: 
        i0 = i
        break

   ## Get UV coordinates of that visibility:
    U0,V0 = self.UV[i0] 

   ## Get Hour Angle (UT corrected by GST) and Baseline:
    H0S = (GST0 + (self.JDate[i0]-int(self.JDate[i0]))*360.-self.RA)*np.pi/180.
    D0S = self.Dec*np.pi/180.
    B0 = (self.antcoords[int(self.ant2[i0]-1),:] - self.antcoords[int(self.ant1[i0]-1),:])/self.c
   # Compute U and V:
    Uc = -(B0[0]*np.sin(H0S) + B0[1]*np.cos(H0S))
    Vc = -B0[0]*np.sin(D0S)*np.cos(H0S)+B0[1]*np.sin(D0S)*np.sin(H0S)+B0[2]*np.cos(D0S)

   ## Correct sign if needed:
    if U0*Uc<0.0 and V0*Vc<0.0:
      print('Applying baseline sign flip!')
      self.antLon *= -1.0
####################



    self.Mask = np.zeros(self.Nvis,dtype=bool)
    self.flags = np.zeros(self.Nvis,dtype=bool)

    self.UV = np.array(self.UV,dtype=np.float64)


    self.UVLam = [self.UV*nui for nui in self.freqs]
    self.pixU = [np.zeros(self.Nvis,dtype=np.int32) for i in range(self.NIF)]
    self.pixV = [np.zeros(self.Nvis,dtype=np.int32) for i in range(self.NIF)]

    self.UV *= self.c*1.e-3 # in Km.
   # print np.shape(self.UV)

 
    self.Q = np.sqrt(self.UV[:,0]**2.+self.UV[:,1]**2.)
    self.Angle = np.arctan2(self.UV[:,0],self.UV[:,1])*180./np.pi+180.
    self.Angle[self.Angle>180.] -= 180.
    self.JDate = np.array(self.JDate,dtype=np.float64)
    self.ant1 = np.array(self.ant1,dtype=np.int32)
    self.ant2 = np.array(self.ant2,dtype=np.int32)
    self.vis = np.array(self.vis,dtype=np.complex128)
    self.wgt = np.array(self.wgt,dtype=np.float64)
    self.ampl = np.abs(self.vis)
    self.MaxAmp = np.max(self.ampl)
    self.phase = 180./np.pi*np.angle(self.vis)
    self.wgt[self.wgt<0.0] = 0.0

    T0 = np.floor(np.min(self.JDate))
    self.UT = (self.JDate - T0)*24.0
    self.UT0 = int(np.floor(np.min(self.UT))) 
    self.UT1 = int(np.floor(np.max(self.UT))+1.0)
    self.Qmax = np.max(self.Q)

# Return GMST from UT time (from NRAO webpage):
    UJDate = np.unique(self.JDate)
    t = (UJDate -2451544.0)/36525.
    Hh = (UJDate - np.floor(UJDate))
    GMsec = 24110.54841 + 8640184.812866*t + 0.093104*t*t - 0.0000062*t*t*t
    self.GMST = (GMsec/86400. + Hh)*360.

    self.buttons['UT0'].config(from_=self.UT0,to=self.UT1)
    self.buttons['UT0'].set(self.UT0)

    NoldAnts = self.buttons['A1'].size()
    self.buttons['A2'].delete(NoldAnts)

    for i in range(NoldAnts-1,-1,-1):
        self.buttons['A1'].delete(i)
        self.buttons['A2'].delete(i)

    for i in range(self.Nants):
      self.buttons['A1'].insert(Tk.END,self.antnames[i])
      self.buttons['A2'].insert(Tk.END,self.antnames[i])
    self.buttons['A2'].insert(Tk.END,'ALL')

    self.buttons['A1'].select_set(0)
    self.buttons['A2'].select_set(0)

    self.Utimes = np.unique(self.UT)
    self.Ntimes = len(self.Utimes)

    self.buttons['UT0'].config(from_=0,to=self.Ntimes-1)
    self.buttons['UT0'].set(0)

    self.STATIONS = self.EarthPlot.plot(self.antLon,self.antLat,'or',transform=ccrs.PlateCarree())
    self.STATION_NAMES = [self.EarthPlot.text(self.antLon[i],self.antLat[i],self.antnames[i],transform=ccrs.PlateCarree(),color='r') for i in range(len(self.antLon))]

    self._plotEarth(True)
    self._plotUV(True)

    self._gridUV()

    self.buttons['BLSel'].invoke()
    self.buttons['Robust'].set(0.0)
    self.buttons['Power'].set(1.0)
    self.buttons['Pixel'].set(3)


####################
#### FOR TESTING:
#    self.vis[:] = 0.0
#    Nvis = len(self.vis[:,0])
#    np.random.seed(42)
#    self.vis[:,0] = 0.5*(np.exp(-2.*np.pi*1.j/self.Qmax*(self.UV[:,0] + self.UV[:,1])) + np.exp(2.*np.pi*1.j/self.Qmax*(self.UV[:,0] + self.UV[:,1])))
#    self.vis[:,0] += np.random.normal(0.,0.001,Nvis)+1.j*np.random.normal(0.,0.001,Nvis) 
####################


    self._plotData(True)

    loadText.set('DONE!')
    win.update()


    return


  def _plotEarth(self,event):



    selLat = self.Dec
    selLon = (self.GMST[self.buttons['UT0'].get()]-self.RA)%360.

## All this try/except rubish comes from the different versions of cartopy (geocollections).
## I haven't found a good way to force the same version for different platforms.
    try:
      self.EarthMap.set_array(np.roll(self.World['data'],int(selLon*self.World['Nroll']/360),axis=1)[:,:])
    except:
      try:
        self.EarthMap.set_array(np.roll(self.World['data'],int(selLon*self.World['Nroll']/360),axis=1)[:-1,:-1])
      except:
        try:
          self.EarthMap.set_array(np.roll(self.World['data'],int(selLon*self.World['Nroll']/360),axis=1)[:,:].ravel())
        except:
          self.EarthMap.set_array(np.roll(self.World['data'],int(selLon*self.World['Nroll']/360),axis=1)[:-1,:-1].ravel())


    self.STATIONS[0].set_data(self.antLon+selLon+180.,self.antLat)
    for st in range(self.Nants):
      print('TEXT FOR %s'%self.antnames[st])
      self.STATION_NAMES[st].set_position((self.antLon[st]+selLon+180.,self.antLat[st]))
      print('DONE')

    H = self.Utimes[self.buttons['UT0'].get()]
    M = (H-int(H))*60.
    S = (M - int(M))*60.
    self.EarthPlot.text(0.05,0.94,'%02i:%02i.%02i UT'%(int(H),int(M),int(S)),transform=self.EarthPlot.transAxes,color='w')


  def _gridUV(self,regrid=True):

    robustness = float(self.buttons['Robust'].get())
    nyqFac = int(self.buttons['Pixel'].get())
    wgtPower = float(self.buttons['Power'].get())
    self.buttons['ImLab'].config(text='Pix. Size: %i (Nyq);   Robust: %2.1f;   Wgt. Power: %2.1f'%(nyqFac,robustness,wgtPower))

    self.PSF[:] = 0.0
    self.Dirty[:] = 0.0
    self.GridUV[:] = 0.0
    self.GridUVCov[:] = 0.0

    Nyquist = self.Npix/((self.Qmax*1.e3)/(self.c/np.max(self.freqs)))
    imsize = Nyquist/float(nyqFac)
    UVpixsize = 1./imsize
    Nph = int(self.Npix/2)

    if regrid:
      for i in range(self.NIF):
        self.pixU[i][:] = np.rint(self.UVLam[i][:,0]/UVpixsize).astype(np.int32)
        self.pixV[i][:] = np.rint(self.UVLam[i][:,1]/UVpixsize).astype(np.int32)

    WgtTot = 0.0
    for i in range(self.NIF):
      wgtPow = (np.logical_not(self.flags))*np.power(self.wgt[:,i],wgtPower)
      toadd = wgtPow*self.vis[:,i]
      np.add.at(self.GridUV,(-self.pixV[i]+Nph,-self.pixU[i]+Nph),toadd)
      np.add.at(self.GridUV,(self.pixV[i]+Nph,self.pixU[i]+Nph),np.conjugate(toadd))
      np.add.at(self.GridUVCov,(-self.pixV[i]+Nph,-self.pixU[i]+Nph), wgtPow)
      np.add.at(self.GridUVCov,(self.pixV[i]+Nph,self.pixU[i]+Nph), wgtPow)
      del toadd
      WgtTot += np.sum(wgtPow)

    WgtTot /= 0.5

# Weight sum for uniform weighting:
    robfac = (5.*10.**(2.0))**2.*(2.*WgtTot)/np.sum(np.conjugate(self.GridUVCov)*self.GridUVCov)
    self.WgtScale = np.sum(self.GridUVCov.real/(1. + robfac*self.GridUVCov.real)).real

# Actual weights:
    robfac = (5.*10.**(-robustness))**2.*(2.*WgtTot)/np.sum(np.conjugate(self.GridUVCov)*self.GridUVCov)
    robust = 1./(1. + robfac*self.GridUVCov)

    self.GridUVCov *= robust
    self.GridUV *= robust


    self.PSF[:] = np.fft.fftshift((np.fft.fft2(np.fft.fftshift(self.GridUVCov))).real)
    self.Dirty[:] = np.fft.fftshift((np.fft.fft2(np.fft.fftshift(self.GridUV))).real)

    PSFMax = np.max(self.PSF)
    self.PSF /= PSFMax
    self.Dirty /= PSFMax

    del robust

    self.PSFImage.set_array(self.PSF)
    self.dirtyImage.set_array(self.Dirty)

    imsec = imsize*180./np.pi*3600.
    self.xfac = 1.0
    if imsec<1.e-4:
      self.unit = r'($\mu$as)'
      self.xfac = 1.e6
      imval = imsec*self.xfac
    elif imsec<1.e-2:
      self.unit = '(mas)'
      self.xfac = 1.e3
      imval = imsec*self.xfac
    else:
      self.unit = '(as)'
      imval = imsec

    pl.setp(self.PSFImage, extent=(imval*0.5*(1.+1./self.Npix),-imval*0.5*(1.-1./self.Npix),-imval*0.5*(1.-1./self.Npix),imval*0.5*(1.+1./self.Npix)))
    self.beamPlot.set_xlabel('Relative RA '+self.unit)
    self.beamPlot.set_ylabel('Relative Dec '+self.unit)
    self.PSFImage.norm.vmin = np.min(self.PSF)
    self.PSFImage.norm.vmax = 1.0

    pl.setp(self.dirtyImage, extent=(imval*0.5*(1.+1./self.Npix),-imval*0.5*(1.-1./self.Npix),-imval*0.5*(1.-1./self.Npix),imval*0.5*(1.+1./self.Npix)))
    self.dirtyPlot.set_xlabel('Relative RA '+self.unit)
    self.dirtyPlot.set_ylabel('Relative Dec '+self.unit)
    self.dirtyImage.norm.vmin = np.min(self.Dirty)
    self.dirtyImage.norm.vmax = np.max(self.Dirty)

    self.curzoom['beamPlot'] = [imval/4.,-imval/4.,-imval/4.,imval/4.]
    self.curzoom['dirtyPlot'] = [imval/4.,-imval/4.,-imval/4.,imval/4.]
    self.Xaxmax = imval/4.

    self.dirtyPlot.set_xlim((self.Xaxmax,-self.Xaxmax))
    self.dirtyPlot.set_ylim((-self.Xaxmax,self.Xaxmax))
    self.beamPlot.set_xlim((self.Xaxmax,-self.Xaxmax))
    self.beamPlot.set_ylim((-self.Xaxmax,self.Xaxmax))



    Peak = np.max(self.Dirty)
    if Peak<0.01:
      self.dirtyInfo.set_text('Peak: %.3f mJy/beam'%(Peak*1.e3))
    else:
      self.dirtyInfo.set_text('Peak: %.3f Jy/beam'%(Peak))
 

    return




  def _plotData(self,event):

      Y = int(self.buttons['Y'].curselection()[0])
      X = int(self.buttons['X'].curselection()[0])

      if Y==0:
        Yplot = np.concatenate([self.ampl[:,i] for i in range(self.NIF)])
        Ymin = 0.0 ; Ymax = self.MaxAmp*1.1
      elif Y==1:
        Yplot = np.concatenate([self.phase[:,i] for i in range(self.NIF)])
        Ymin = -179.9 ; Ymax = 179.9
      elif Y==2:
        Yplot = np.concatenate([self.vis.real[:,i] for i in range(self.NIF)])
        Ymin = -self.MaxAmp ; Ymax = self.MaxAmp
      elif Y==3:
        Yplot = np.concatenate([self.vis.imag[:,i] for i in range(self.NIF)])
        Ymin = -self.MaxAmp ; Ymax = self.MaxAmp
      if X==0:
        Xplot = np.concatenate([self.Q for i in range(self.NIF)])
        Xmin = -self.Qmax*0.1 ; Xmax = self.Qmax*1.1
      elif X==1:
        Xplot = np.concatenate([self.UT for i in range(self.NIF)])
        Dt = (self.UT1-self.UT0)
        Xmin = self.UT0 - Dt*0.1 ; Xmax = self.UT1 + Dt*0.1
      elif X==2:
        Xplot = np.concatenate([self.UV[:,0] for i in range(self.NIF)])
        Xmin = -self.Qmax*1.1 ; Xmax = self.Qmax*1.1
      elif X==3:
        Xplot = np.concatenate([self.UV[:,1] for i in range(self.NIF)])
        Xmin = -self.Qmax*1.1 ; Xmax = self.Qmax*1.1

      self.dataPlot.set_data(Xplot,Yplot)
      self.dataFlagged.set_data(Xplot[self.flags],Yplot[self.flags])

      del Xplot, Yplot

      self.dataFig.set_xlabel(self.plotXLabel[X])
      self.dataFig.set_ylabel(self.plotYLabel[Y])
      self.dataFig.set_ylim((Ymin,Ymax))
      self.dataFig.set_xlim((Xmin,Xmax))

      if self.IsBas.get()==1:
        self._selectBL(True)
      elif self.IsScan.get()==1:
        self._selectUT(self.buttons['UT0'].get())
      elif self.IsAng.get()==1:
        self._selectAng(self.buttons['Ang'].get())
      else:
        self.canvas.draw()
        pl.draw()

      self.curzoom['dataPlot'] = [Xmin,Xmax,Ymin,Ymax]




  def _plotUV(self,event):
    self.UVCov[0].set_data( self.UV[:,0], self.UV[:,1])
    self.UVCov[1].set_data(-self.UV[:,0],-self.UV[:,1])


    self.UVPlot.set_xlim((self.Qmax*1.1,-self.Qmax*1.1))
    self.UVPlot.set_ylim((-self.Qmax*1.1,self.Qmax*1.1))
    self.UVPlot.set_xlabel('U (km)')
    self.UVPlot.set_ylabel('V (km)')
    self.curzoom['UVPlot'] = [self.Qmax*1.1,-self.Qmax*1.1,-self.Qmax*1.1,self.Qmax*1.1]
    return
      





# UVPlot and dataPlot:
  def _onPick(self,event):

       SelVis = event.ind[0]


       ANT1 = self.antnames[self.antidx.index(self.ant1[SelVis])]
       ANT2 = self.antnames[self.antidx.index(self.ant2[SelVis])]
       H = self.UT[SelVis]
       M = (H-int(H))*60.0
       self.UVInfo.set_text('Clicked on %s-%s at UT %2i:%2.1f'%(ANT1,ANT2,int(H),M))
       self.Mask[:] = False
       self.Mask[SelVis] = True
       self._plotHighlighted()
       
       return


  def _onDrag(self,event):
 
    if self.pressed and event.inaxes == self.dataFig:
      self.moved = True
## PLOT RECTANGLE!
      x1 = event.xdata ; y1 = event.ydata
      x0,y0 = self.xy0

      if self.flagmode==0:
        self.Box.set_data([x0,x0,x1,x1,x0],[y0,y1,y1,y0,y0])
      else:
        self.BoxFg.set_data([x0,x0,x1,x1,x0],[y0,y1,y1,y0,y0])

    else:
      self.pressed = False
      self.moved = False
## UNPLOT RECTANGLE!
      self.Box.set_data([0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.])
      self.BoxFg.set_data([0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.])

    if event.inaxes == self.dataFig:
      pl.draw()
      self.canvas.draw()

    return


  def _onButtonPress(self,event):

    if event.inaxes == self.dataFig:
      self.pressed = True
      self.xy0 = [event.xdata, event.ydata]
    else:
      self.pressed = False
    return


  def _onButtonRelease(self,event):

    self.Box.set_data([0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.])
    self.BoxFg.set_data([0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.])

    if event.inaxes != self.dataFig:
      self.moved = False
      self.pressed = False

    elif self.pressed:
 
      self.pressed = False
      x1 = event.xdata ; y1 = event.ydata
      x0,y0 = self.xy0

      if x0 != x1 or y1 != y0:

        if x0<x1:
          xx0,xx1 = [x0,x1]
        else:
          xx1,xx0 = [x0,x1]

        if y0<y1:
          yy0,yy1 = [y0,y1]
        else:
          yy1,yy0 = [y0,y1]


        if self.flagmode > 0:

          X,Y = self.dataPlot.get_data()
          newFlags = np.logical_and(np.logical_and(X>=xx0,X<=xx1),np.logical_and(Y>=yy0,Y<=yy1))
        
          if self.flagmode==1:
            self.flags[newFlags] = True
          else:
            self.flags[newFlags] = False

          self.dataFlagged.set_data(X[self.flags],Y[self.flags])
          del X, Y, newFlags
      
          self._gridUV(regrid=False)


        else:
          self.curzoom['dataPlot'] = [xx0,xx1,yy0,yy1]
          event.inaxes.set_xlim((xx0,xx1))
          event.inaxes.set_ylim((yy0,yy1))

        pl.draw()
        self.canvas.draw()
        
    self.canvas.get_tk_widget().focus_force()
    return




  def _onPress(self,event):

   
    if event.axname in ['beamPlot','dirtyPlot','UVPlot']:
      inv = True
      if event.axname == 'UVPlot':
        MaxX = [-self.Qmax*1.1, self.Qmax*1.1]
        MaxY = MaxX
      else:
        MaxX = [-self.Xaxmax, self.Xaxmax]
        MaxY = MaxX
    else:
      inv = False
      MaxX = (-2.e5,1.3e7)
      MaxY = (-1.31e5,1.3e7)

    if len(event.axname)>0:
# ZOOM IN:

      if event.key == 'z':

         RA = event.xdata
         Dec = event.ydata
         xL = np.abs(self.curzoom[event.axname][1]-self.curzoom[event.axname][0])/4.
         yL = np.abs(self.curzoom[event.axname][3]-self.curzoom[event.axname][2])/4.
         x0 = RA-xL
         x1 = RA+xL
         y0 = Dec-xL
         y1 = Dec+xL

         if x0 < MaxX[0]:
           x0 = MaxX[0]
           x1 = x0 + 2.*xL
         if x1 > MaxX[1]:
           x1 = MaxX[1]
           x0 = x1 - 2.*xL
         if y0 < MaxY[0]:
           y0 = MaxY[0]
           y1 = y0 + 2.*xL
         if y1 > MaxY[1]:
           y1 = MaxY[1]
           y0 = y1 - 2.*xL



# ZOOM OUT:
      if event.key == 'u':


         RA = event.xdata
         Dec = event.ydata
         xL = np.abs(self.curzoom[event.axname][1]-self.curzoom[event.axname][0])
         yL = np.abs(self.curzoom[event.axname][3]-self.curzoom[event.axname][2])
         if xL > MaxX[1]:
           xL = MaxX[1]
         if yL > MaxY[1]:
           yL = MaxY[1]

         x0 = RA-xL
         x1 = RA+xL
         y0 = Dec-xL
         y1 = Dec+xL

         if x0 < MaxX[0]:
           x0 = MaxX[0]
           x1 = x0 + 2.*xL
         if x1 > MaxX[1]:
           x1 = MaxX[1]
           x0 = x1 - 2.*xL
         if y0 < MaxY[0]:
           y0 = MaxY[0]
           y1 = y0 + 2.*xL
         if y1 > MaxY[1]:
           y1 = MaxY[1]
           y0 = y1 - 2.*xL


      if event.key in ['z','u']:
        if inv:
          event.inaxes.set_xlim((x1,x0))
          event.inaxes.set_ylim((y0,y1))
          self.curzoom[event.axname] = [x1,x0,y0,y1]
        else:
          event.inaxes.set_xlim((x0,x1))
          event.inaxes.set_ylim((y0,y1))
          self.curzoom[event.axname] = [x0,x1,y0,y1]

        pl.draw()
        self.canvas.draw()

    return






  def _onKeyPress(self,event):

    event.axname = ''

    if event.key in ['c','C']:
      print('Change color')
      self.cmap += 1
      if self.cmap == len(self.cmaps):
        self.cmap = 0

      pl.set_cmap(self.cmaps[self.cmap])
      self.currcmap = cm.get_cmap() # cm.jet
      self.PSFImage.set_cmap(self.currcmap)
      self.dirtyImage.set_cmap(self.currcmap)

      if self.myCLEAN is not None:
        self.myCLEAN.ResidPlotPlot.set_cmap(self.currcmap)
        self.myCLEAN.CLEANPlotPlot.set_cmap(self.currcmap)

      if self.myMEM is not None:
        self.myMEM.ResidPlotPlot.set_cmap(self.currcmap)
        self.myMEM.MEMPlotPlot.set_cmap(self.currcmap)

      self.canvas.draw()
      pl.draw()
      self.canvas.flush_events()
      return
 

    if event.inaxes == self.beamPlot:
      event.axname = 'beamPlot'
    elif event.inaxes == self.dirtyPlot:
      event.axname = 'dirtyPlot'
    elif event.inaxes == self.UVPlot:
      event.axname = 'UVPlot'
    elif event.inaxes == self.dataFig:
      event.axname = 'dataPlot'
      if event.key in ['f', 'F']:
        self.flagmode += 1
        if self.flagmode == 3:
          self.flagmode = 0
      self.dataTxt.set_text('Current Mouse Mode:  %s  (press \'F\' to change)'%self.fgModes[self.flagmode])   
      self.canvas.draw()
      pl.draw()
    self._onPress(event)

    return





  def _doCLEAN(self):

    if self.tks is not None:
      self.myCLEAN = CLEANer(self)

    return

  def _doMEM(self):

    if self.tks is not None:
      self.myMEM = MEMer(self)

    return



  def _fillHeader(self, head, dx,bunit,beam={}):
      Now = dt.strftime('%Y-%m-%dT%H:%M:%S.000000',dt.gmtime())
      head['BSCALE'] = 1.0
      head['BZERO'] = 0.0
      for key in beam:
        head[key] = beam[key]
      head['BTYPE'] = 'Intensity'
      head['BUNIT'] = bunit
      head['OBJECT'] = self.snam
      head['EQUINOX'] = 2000.0
      head['CTYPE1'] = 'RA---SIN'
      head['CRVAL1'] = float(self.ra)
      head['CDELT1'] = -float(dx)
      head['CRPIX1'] = self.Npix/4.
      head['CUNIT1'] = 'deg'
      head['CTYPE2'] = 'DEC--SIN'
      head['CRVAL2'] = float(self.dec)
      head['CDELT2'] = float(dx)
      head['CRPIX2'] = self.Npix/4.
      head['CUNIT2'] = 'deg'
      head['CTYPE3'] = 'FREQ'
      head['CRVAL3'] = self.Nu0
      head['CDELT3'] = np.average(self.BWTOT)
      head['CRPIX3'] = 1
      head['CUNIT3'] = 'Hz'
      head['CTYPE4'] = 'STOKES'
      head['CRVAL4'] = 1.
      head['CDELT4'] = 1.
      head['CRPIX4'] = 1.
      head['CUNIT4'] = '  '
      head['RESTFRQ'] = self.Nu0
      head['SPECSYS'] = 'LSRK'
      head['VELREF'] = 257
      head['TELESCOP'] = self.telesc
      head['OBSERVER'] = 'APST '+str(__version__)
      head['DATE-OBS'] = self.date
      head['TIMESYS'] = 'UTC'
      head['OBSRA']   = self.ra                                      
      head['OBSDEC']  =  self.dec                                                  
      head['OBSGEO-X'] = 0.0                                                  
      head['OBSGEO-Y'] = 0.0                                                  
      head['OBSGEO-Z'] = 0.0                            
      head['DATE']     = Now              
      head['ORIGIN']  = 'APSYNTRUE'       








class CLEANer(object):

  def _getHelp(self):
    win = Tk.Toplevel(self.me)
    win.title("Help")
    helptext = ScrolledText.ScrolledText(win)
    helptext.config(state=Tk.NORMAL)
    helptext.insert(Tk.INSERT,__CLEAN_help_text__)
    helptext.config(state=Tk.DISABLED)
    helptext.pack()
    Tk.Button(win, text='OK', command=win.destroy).pack()



  def quit(self):

    self.parent.myCLEAN = None
    self.me.destroy()


  def __init__(self,parent):

    self.parent = parent
    self.me = Tk.Toplevel(parent.tks)

## Needed for MacOS to work:
    try:
      root.state('zoomed')
    except:
      root.attributes('-zoomed',True)

    menubar = Tk.Menu(self.me)
    menubar.add_command(label="Help", command=self._getHelp)
    menubar.add_command(label="Quit", command=self.quit)

    self.me.config(menu=menubar)
    self.me.protocol("WM_DELETE_WINDOW", self.quit)
    self.Np4 = self.parent.Npix/4

    self.figCL1 = pl.figure(figsize=(12,6))    

    self.residuals = np.zeros(np.shape(self.parent.Dirty))
    self.cleanmod = np.copy(self.residuals)
    self.cleanmodd = np.copy(self.residuals)
    self.mask = np.zeros(np.shape(self.parent.Dirty),dtype=bool)
    self.bmask = np.zeros(np.shape(self.parent.Dirty),dtype=bool)
    self.PSF = np.copy(self.residuals)

    self.ResidPlot = self.figCL1.add_subplot(121,aspect='equal')
    self.CLEANPlot = self.figCL1.add_subplot(122,aspect='equal',sharex=self.ResidPlot,sharey=self.ResidPlot)
    self.ResidPlot.set_adjustable('box')
    self.CLEANPlot.set_adjustable('box')

    self.frames = {}
    self.frames['FigFr'] = Tk.Frame(self.me)
    self.frames['GFr'] = Tk.Frame(self.me)

    self.canvas1 = FigureCanvasTkAgg(self.figCL1, master=self.frames['FigFr'])

    self.canvas1.draw()

    self.frames['FigFr'].pack(side=Tk.TOP)

    self.frames['CLOpt'] = Tk.Frame(self.frames['FigFr'])

    self.frames['Gain'] = Tk.Frame(self.frames['CLOpt'])
    self.frames['Niter'] = Tk.Frame(self.frames['CLOpt'])
    self.frames['Thres'] = Tk.Frame(self.frames['CLOpt'])

    Gtext = Tk.Label(self.frames['Gain'],text="Gain:  ")
    Ntext = Tk.Label(self.frames['Niter'],text="# iter:")
    Ttext = Tk.Label(self.frames['Thres'],text="Thres (Jy/b):")

    self.entries = {}
    self.entries['Gain'] = Tk.Entry(self.frames['Gain'])
    self.entries['Gain'].insert(0,"0.1")
    self.entries['Gain'].config(width=5)

    self.entries['Niter'] = Tk.Entry(self.frames['Niter'])
    self.entries['Niter'].insert(0,"100")
    self.entries['Niter'].config(width=5)

    self.entries['Thres'] = Tk.Entry(self.frames['Thres'])
    self.entries['Thres'].insert(0,"0.0")
    self.entries['Thres'].config(width=5)

    Gtext.pack(side=Tk.LEFT)
    self.entries['Gain'].pack(side=Tk.RIGHT)

    Ntext.pack(side=Tk.LEFT)
    self.entries['Niter'].pack(side=Tk.RIGHT)

    Ttext.pack(side=Tk.LEFT)
    self.entries['Thres'].pack(side=Tk.RIGHT)


    self.frames['CLOpt'].pack(side=Tk.LEFT)
    self.canvas1.get_tk_widget().pack(side=Tk.LEFT) 

    self.buttons = {}

    self.buttons['clean'] = Tk.Button(self.frames['CLOpt'],text="CLEAN",command=self._CLEAN)
    self.buttons['reset'] = Tk.Button(self.frames['CLOpt'],text="RELOAD",command=self._reset)
    self.buttons['addres'] = Tk.Button(self.frames['CLOpt'],text="+/- Resid",command=self._AddRes)
    self.buttons['dorestore'] = Tk.Button(self.frames['CLOpt'],text="(Un)restore",command=self._doRestore)
    self.buttons['dorescale'] = Tk.Button(self.frames['CLOpt'],text="Rescale",command=self._doRescale)



    self.frames['Gain'].pack(side=Tk.TOP)
    self.frames['Niter'].pack(side=Tk.TOP)
    self.frames['Thres'].pack(side=Tk.TOP)

    self.buttons['clean'].pack(side=Tk.TOP)
    self.buttons['reset'].pack(side=Tk.TOP)
    self.buttons['addres'].pack(side=Tk.TOP)
    self.buttons['dorestore'].pack(side=Tk.TOP)
    self.buttons['dorescale'].pack(side=Tk.TOP)


    separator = Tk.Frame(self.frames['CLOpt'],height=4, bd=5, relief=Tk.SUNKEN)
    separator.pack(fill=Tk.X, padx=10, pady=20,side=Tk.TOP)

    self.buttons['save'] = Tk.Button(self.frames['CLOpt'],text="SAVE IMAGE",command=self._SAVE)
    self.buttons['save'].pack(side=Tk.TOP)



    self.canvas1.mpl_connect('motion_notify_event', self._doMask)
    self.canvas1.mpl_connect('button_release_event',self._onRelease)
    self.canvas1.mpl_connect('button_press_event',self._onPress)
    self.canvas1.mpl_connect('key_press_event', self._onKeyPress)


    self.doMask = True
    self.maskMode = {True:'Add Mask',False:'Remove Mask'}
    self.pressedID = -1
    self.xy0 = [0,0]
    self.moved = False
    self.resadd = False
    self.dorestore = True



    self._reset()


  def _reset(self):

    self.ResidPlot.cla()
    self.CLEANPlot.cla()
    self.dorestore = True



    self.residuals[:] = self.parent.Dirty
    self.cleanmod[:] = 0.0 #np.zeros(np.shape(self.parent.Dirty))
    self.cleanmodd[:] = 0.0 #np.zeros(np.shape(self.parent.Dirty))
    self.PSF[:] = self.parent.PSF

    self.fmtD2 = r'Resid. Peak: % .2e Jy/beam ; rms: % .2e Jy/beam'
    self.fmtDC = r'CLEAN Peak: % .3e Jy/beam ; SNR: % 4.2f'



    self.X0 = int(self.parent.Npix/4) ; self.X1 = int(self.X0 + int(self.parent.Npix/2))
    dslice = self.residuals[self.X0:self.X1, self.X0:self.X1]

    self.Xmax = self.parent.Xaxmax
    self.ResidPlotPlot = self.ResidPlot.imshow(dslice,interpolation='nearest',picker=True, cmap=self.parent.currcmap, 
                            extent = (self.Xmax,-self.Xmax,-self.Xmax,self.Xmax))

    self.RMS = np.sqrt(np.var(dslice)+np.average(dslice)**2.)
    self.PEAK = np.max(dslice)
    modflux = dslice[self.X0,self.X0] # self.parent.dirtymap[self.parent.Nphf,self.parent.Nphf]
 
    self.pickcoords = [self.parent.Npix/2,self.parent.Npix/2,0.,0.]
    self.ResidText = self.ResidPlot.text(0.05,0.95,self.fmtD2%(self.PEAK,self.RMS),
         transform=self.ResidPlot.transAxes,bbox=dict(facecolor='white', alpha=0.7))

    self.MaskTxt = self.ResidPlot.text(0.05,0.025,'Current Mouse Mode:  %s  (press \'F\' to change)'%self.maskMode[self.doMask],
         transform=self.ResidPlot.transAxes,bbox=dict(facecolor='white', alpha=0.7))   


    self.Box = self.ResidPlot.plot([0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.],lw=2,color='w')[0]

    self.ResidPlot.set_ylabel('Dec offset '+self.parent.unit)
    self.ResidPlot.set_xlabel('RA offset '+self.parent.unit)
    self.ResidPlot.set_title('RESIDUALS')

    self.MaskPlot = self.ResidPlot.contour(np.linspace(self.Xmax,-self.Xmax,self.X1-self.X0),np.linspace(self.Xmax,-self.Xmax,self.X1-self.X0),self.mask[self.X0:self.X1, self.X0:self.X1],levels=[0.5])

    dslice = self.cleanmod[self.X0:self.X1, self.X0:self.X1]
    self.CLEANPlotPlot = self.CLEANPlot.imshow(dslice,interpolation='nearest',picker=True, cmap=self.parent.currcmap, 
                            extent = (self.Xmax,-self.Xmax,-self.Xmax,self.Xmax))

    self.CLEANPEAK = 0.0
    self.totiter = 0

    self.CLEANPlot.set_ylabel('Dec offset '+self.parent.unit)
    self.CLEANPlot.set_xlabel('RA offset '+self.parent.unit)
    self.CLEANPlot.set_title('CLEAN (0 ITER)')
    self.CLEANText = self.CLEANPlot.text(0.05,0.92,'',transform=self.CLEANPlot.transAxes,bbox=dict(facecolor='white', alpha=0.7))

    # DERIVE THE CLEAN BEAM
    MainLobe = np.where(self.parent.PSF>0.5)
    self.cleanBeam = np.zeros(np.shape(self.residuals))

    self.dx = 4.*self.parent.Xaxmax/self.parent.Npix/self.parent.xfac


    if len(MainLobe[0]) < 5:
      showinfo('ERROR!', 'The main lobe of the PSF is too narrow!\n CLEAN model will not be restored')
      self.cleanBeam[:] = 0.0
      self.cleanBeam[self.parent.Npix/2,self.parent.Npix/2] = 1.0
    else:
      dX = np.array(MainLobe[0]-self.parent.Npix/2.).astype(np.float64) ; dY = np.array(MainLobe[1]-self.parent.Npix/2.).astype(np.float64)
    #  if True:
      try:
        fit = spfit.leastsq(lambda x: (np.exp(-(dX*dX*x[0]+dY*dY*x[1]+dX*dY*x[2]))-self.parent.PSF[MainLobe])/(dX*dX+dY*dY+1.0),[1.,1.,0.])
        print(dX)
        print(dY)
        Pang = 180./np.pi*(np.arctan2(fit[0][2],(fit[0][0]-fit[0][1]))/2.)
        AmB = fit[0][2]/np.sin(2.*np.pi/180.*Pang) ;  ApB = fit[0][0]+fit[0][1]
        A = 2.355*(2./(ApB + AmB))**0.5*self.dx # imsize/self.parent.Npix  
        B = 2.355*(2./(ApB - AmB))**0.5*self.dx # imsize/self.parent.Npix
        if A < B:
          A, B = B, A
          Pang = Pang - 90.
        if Pang < -90.:
          Pang += 180.
        if Pang > 90.:
          Pang -= 180.

        self.beamInfo = [A/3600.,B/3600.,float(Pang)]

        if B > 0.1:
          self.Beamtxt = 'BEAM = %.1f x %.1f as (PA = %.1f deg.)'%(A,B,Pang)
        elif B>0.1e-3:
          self.Beamtxt = 'BEAM = %.1f x %.1f mas (PA = %.1f deg.)'%(1000.*A,1000.*B,Pang)
        else:
          self.Beamtxt = ('BEAM = %.1f x %.1f' + r'$\mu$as' + ' (PA = %.1f deg.)')%(1.e6*A,1.e6*B,Pang)

        self.CLEANText.set_text(self.fmtDC%(0.,0.)+'\n'+self.Beamtxt)
        ddX = np.outer(np.ones(self.parent.Npix),np.arange(-self.parent.Npix/2,self.parent.Npix/2).astype(np.float64))
        ddY = np.outer(np.arange(-self.parent.Npix/2,self.parent.Npix/2).astype(np.float64),np.ones(self.parent.Npix))

        self.cleanBeam[:] = np.exp(-(ddY*ddY*fit[0][0]+ddX*ddX*fit[0][1]+ddY*ddX*fit[0][2]))

        del ddX, ddY
   #   else:
      except:
        showinfo('ERROR!', 'Problems fitting the PSF main lobe!\n CLEAN model will not be restored')
        self.cleanBeam[:] = 0.0
        self.cleanBeam[self.parent.Npix/2,self.parent.Npix/2] = 1.0
        self.beamInfo = [self.dx/3600.,self.dx/3600.,0.0]

    self.resadd = False
    self.dorestore = True
    self.ffti = False

    self.totalClean = 0.0

    self.canvas1.draw()



  def _onPress(self,event):

    self.canvas1._tkcanvas.focus_set()
    if event.inaxes == self.ResidPlot:
      self.pressedID = int(event.button)
      RA = event.xdata
      Dec = event.ydata
      self.xydata = [RA,Dec]
      self.xy0[1] = np.floor((2.*self.Xmax-RA)/(4.*self.Xmax)*self.parent.Npix)
      self.xy0[0] = np.floor((2.*self.Xmax-Dec)/(4.*self.Xmax)*self.parent.Npix)
      self.moved = False 


  def _onRelease(self,event):

    if event.inaxes != self.ResidPlot:
      self.moved=False

    if self.moved:
      RA = event.xdata
      Dec = event.ydata
      y1 = np.floor((2.*self.Xmax-RA)/(4.*self.Xmax)*self.parent.Npix)
      x1 = np.floor((2.*self.Xmax-Dec)/(4.*self.Xmax)*self.parent.Npix)
      xi,xf = map(int,[min(self.xy0[0],x1),max(self.xy0[0],x1)])
      yi,yf = map(int,[min(self.xy0[1],y1),max(self.xy0[1],y1)])
      if self.doMask:
        self.mask[xi:xf,yi:yf] = 1.0
        self.bmask[xi:xf,yi:yf] = True
      else:
        self.mask[xi:xf,yi:yf] = 0.0
        self.bmask[xi:xf,yi:yf] = False

      for coll in self.MaskPlot.collections:
        # self.ResidPlot.collections.remove(coll)
         coll.remove()

      self.MaskPlot = self.ResidPlot.contour(np.linspace(self.Xmax,-self.Xmax,int(self.parent.Npix/2)),np.linspace(self.Xmax,-self.Xmax,int(self.parent.Npix/2)),self.mask[self.X0:self.X1, self.X0:self.X1],levels=[0.5])

      self.canvas1.draw()

      self.Box.set_data([0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.])

    self.moved = False
    self.pressedID = -1
    self.canvas1.draw()



  def _doMask(self,event):
    if self.pressedID>=0 and event.inaxes==self.ResidPlot:
      self.moved = True
      RA = event.xdata
      Dec = event.ydata
      y1 = np.floor((self.Xmax-RA)/(2.*self.Xmax)*self.parent.Npix)
      x1 = np.floor((self.Xmax-Dec)/(2.*self.Xmax)*self.parent.Npix)
      self.Box.set_data([self.xydata[0],self.xydata[0],RA,RA,self.xydata[0]],[self.xydata[1],Dec,Dec,self.xydata[1],self.xydata[1]])
      self.canvas1.draw()





  def _CLEAN(self):

     if np.sum(self.bmask)==0:
       goods = np.ones(np.shape(self.bmask)).astype(bool)
       tempres = self.residuals
     else:
       goods = self.bmask
       tempres = self.residuals*self.mask

     try:
       gain = float(self.entries['Gain'].get())
       niter = int(self.entries['Niter'].get())
       thrs = float(self.entries['Thres'].get())
     except:
       showinfo('ERROR!','Please, check the content of Gain, # Iter, and Thres!\nShould be numbers!')
       return

     for i in range(niter):
       self.totiter += 1

       if thrs != 0.0:
         tempres[tempres<thrs] = 0.0
         if thrs < 0.0:
           tempres = np.abs(tempres)

         if np.sum(tempres)==0.0:
           showinfo('INFO','Threshold reached in CLEAN masks!')
           break


       rslice = self.residuals[self.X0:self.X1, self.X0:self.X1]
       peakpos = np.unravel_index(np.argmax(tempres),np.shape(self.residuals))
       peakval = self.residuals[peakpos[0],peakpos[1]]
       self.residuals -= gain*peakval*np.roll(np.roll(self.PSF,int(peakpos[0]-self.parent.Npix/2),axis=0), int(peakpos[1]-self.parent.Npix/2),axis=1)
       tempres[goods] = self.residuals[goods]
       # MODIFY CLEAN MODEL!!
       self.cleanmodd[int(peakpos[0]),int(peakpos[1])] += gain*peakval
       self.cleanmod += gain*peakval*np.roll(np.roll(self.cleanBeam,int(peakpos[0]-self.parent.Npix/2),axis=0), int(peakpos[1]-self.parent.Npix/2),axis=1)
       self.ResidPlotPlot.set_array(rslice)

       self.CLEANPEAK = np.max(self.cleanmod)
       self.totalClean += gain*peakval
       self.CLEANPlot.set_title('CLEAN (%i ITER). TOTAL = %.2e Jy'%(self.totiter,self.totalClean))

       xi,yi,RA,Dec = self.pickcoords

       if self.dorestore:
        if self.resadd:
         toadd = (self.cleanmod + self.residuals)
        else:
         toadd = self.cleanmod
       else:
         toadd = self.cleanmodd

       clFlux = toadd[int(xi),int(yi)]

       self.CLEANPlotPlot.set_array(toadd[self.X0:self.X1, self.X0:self.X1])
       self.CLEANPlotPlot.norm.vmin = np.min(toadd[self.X0:self.X1, self.X0:self.X1])
       self.CLEANPlotPlot.norm.vmax = np.max(toadd[self.X0:self.X1, self.X0:self.X1])

       self.RMS = np.sqrt(np.var(rslice)+np.average(rslice)**2.)
       self.PEAK = np.max(rslice)
       self.ResidText.set_text(self.fmtD2%(self.PEAK,self.RMS))
       self.CLEANText.set_text(self.fmtDC%(self.CLEANPEAK,self.CLEANPEAK/self.RMS)+'\n'+self.Beamtxt)

       self.canvas1.draw()
       self.canvas1.flush_events()

# Re-draw if threshold reached:
     self.canvas1.draw()
     del tempres, goods
     try:
       del toadd
     except:
       pass



  def _doRescale(self):

    clarr = self.CLEANPlotPlot.get_array()
    self.CLEANPlotPlot.norm.vmin = np.min(clarr)
    self.CLEANPlotPlot.norm.vmax = np.max(clarr)
    self.CLEANPlotPlot.set_array(clarr)
    rsarr = self.ResidPlotPlot.get_array()
    self.ResidPlotPlot.norm.vmin = np.min(rsarr)
    self.ResidPlotPlot.norm.vmax = np.max(rsarr)
    self.ResidPlotPlot.set_array(rsarr)

    del clarr, rsarr
    self.canvas1.draw()



  def _doRestore(self):

   if self.dorestore:
    self.dorestore = False
    toadd = self.cleanmodd[self.X0:self.X1, self.X0:self.X1]

   else:
    self.dorestore = True
    if self.resadd:
     toadd = (self.cleanmod + self.residuals)[self.X0:self.X1, self.X0:self.X1]
    else:
     toadd = self.cleanmod[self.X0:self.X1, self.X0:self.X1]


   self.CLEANPlotPlot.set_array(toadd)
   self.CLEANPlotPlot.norm.vmin = np.min(toadd)
   self.CLEANPlotPlot.norm.vmax = np.max(toadd)
   self.canvas1.draw()
   del toadd




  def _AddRes(self):

    if not self.dorestore:
      showinfo('ERROR','Cannot add residual to the (unrestored) CLEAN model!\nRestore first!')

    if self.resadd:
      self.resadd = False
      toadd = self.cleanmod[self.X0:self.X1, self.X0:self.X1]
    else:
      self.resadd = True
      toadd = (self.cleanmod + self.residuals)[self.X0:self.X1, self.X0:self.X1]

    self.CLEANPlotPlot.set_array(toadd)
    self.CLEANPlotPlot.norm.vmin = np.min(toadd)
    self.CLEANPlotPlot.norm.vmax = np.max(toadd)

    self.canvas1.draw()
    del toadd




  def _onKeyPress(self,event):

    if event.key in ['f', 'F']:
      self.doMask = not self.doMask
      self.MaskTxt.set_text('Current Mouse Mode:  %s  (press \'F\' to change)'%self.maskMode[self.doMask])   
      self.canvas1.draw()
      pl.draw()

    return



  def _SAVE(self):


    filename = os.path.join(self.parent.userdir,'%s_CLEAN_%i.fits'%(self.parent.snam, int(dt.time())))
    if not os.path.exists(self.parent.userdir):
      os.makedirs(self.parent.userdir)

    if self.dorestore:
      if self.resadd:
        tosave = (self.cleanmod + self.residuals)[self.X0:self.X1, self.X0:self.X1]
      else:
        tosave = self.cleanmod[self.X0:self.X1, self.X0:self.X1]
    else:
      tosave = self.cleanmodd[self.X0:self.X1, self.X0:self.X1]



    hdu = pf.PrimaryHDU(np.copy(tosave[::-1,:,np.newaxis,np.newaxis]).transpose(2,3,0,1))
    hdulist = pf.HDUList([hdu])
    head = hdulist[0].header
    self.parent._fillHeader(head, self.dx/3600., 'Jy/beam',beam={'BMAJ':self.beamInfo[0],'BMIN':self.beamInfo[1],'BPA':self.beamInfo[2]})
    if os.path.exists(filename):
      os.remove(filename)
    hdulist.writeto(filename)

    showinfo('INFO','\n\nFITS CLEAN image saved succesfully!\n\nName: %s\n'%os.path.basename(filename))





    return






class MEMer(object):

  def _getHelp(self):
    win = Tk.Toplevel(self.me)
    win.title("Help")
    helptext = ScrolledText.ScrolledText(win)
    helptext.config(state=Tk.NORMAL)
    helptext.insert(Tk.INSERT,__MEM_help_text__)
    helptext.config(state=Tk.DISABLED)
    helptext.pack()
    Tk.Button(win, text='OK', command=win.destroy).pack()


  def quit(self):
    self.parent.myMEM = None
    self.me.destroy()


  def __init__(self,parent):

    self.parent = parent
    self.me = Tk.Toplevel(parent.tks)

## Needed for MacOS to work:
    try:
      root.state('zoomed')
    except:
      root.attributes('-zoomed',True)



    menubar = Tk.Menu(self.me)
    menubar.add_command(label="Help", command=self._getHelp)
    menubar.add_command(label="Quit", command=self.quit)

    self.me.config(menu=menubar)
    self.me.protocol("WM_DELETE_WINDOW", self.quit)

    self.figMM1 = pl.figure(figsize=(12,6))    


    NPIX = self.parent.Npix
    self.X0 = int(self.parent.Npix/4) ; self.X1 = self.X0 + int(self.parent.Npix/2)


    self.UVCov = np.zeros((NPIX,NPIX),dtype=np.complex128)
    self.UVData = np.copy(self.UVCov)
    self.MEM = np.zeros((NPIX,NPIX))
    self.Dirty = np.copy(self.parent.Dirty)
    self.residuals = np.copy(self.Dirty)
    self.Model = np.zeros((NPIX,NPIX))
    self.MODFFT = np.zeros((NPIX,NPIX),dtype=np.complex128)


    self.Xpix = np.outer(np.ones(NPIX),np.arange(NPIX))
    self.Ypix = np.outer(np.arange(NPIX),np.ones(NPIX))


    self.Chi2Grad = np.zeros((NPIX,NPIX))
    self.EntrGrad = np.zeros((NPIX,NPIX))
    self.Metric = np.zeros((NPIX,NPIX))
    self.Grad = np.zeros((NPIX,NPIX))

    self.ResidPlot = self.figMM1.add_subplot(121,aspect='equal')
    self.MEMPlot = self.figMM1.add_subplot(122,aspect='equal',sharex=self.ResidPlot,sharey=self.ResidPlot)
    self.ResidPlot.set_adjustable('box')
    self.MEMPlot.set_adjustable('box')



    self.frames = {}
    self.frames['FigFr'] = Tk.Frame(self.me)
    self.frames['GFr'] = Tk.Frame(self.me)

    self.canvas1 = FigureCanvasTkAgg(self.figMM1, master=self.frames['FigFr'])

    self.canvas1.draw()

    self.frames['FigFr'].pack(side=Tk.TOP)

    self.frames['MMOpt'] = Tk.Frame(self.frames['FigFr'])

    self.frames['Hyper'] = Tk.Frame(self.frames['MMOpt'])
    self.frames['Niter'] = Tk.Frame(self.frames['MMOpt'])
    self.frames['RMS'] = Tk.Frame(self.frames['MMOpt'])
    self.frames['Flux'] = Tk.Frame(self.frames['MMOpt'])

    Htext = Tk.Label(self.frames['Hyper'],text="Damping:  ")
    Ntext = Tk.Label(self.frames['Niter'],text="# iters.:")
    Rtext = Tk.Label(self.frames['RMS'],text="RMS (Jy/b):")
    Stext = Tk.Label(self.frames['Flux'],text="Flux (Jy):")

    self.entries = {}
    self.entries['Hyper'] = Tk.Entry(self.frames['Hyper'])
    self.entries['Hyper'].insert(0,"0.01")
    self.entries['Hyper'].config(width=5)

    self.entries['Niter'] = Tk.Entry(self.frames['Niter'])
    self.entries['Niter'].insert(0,"100")
    self.entries['Niter'].config(width=5)

    self.entries['RMS'] = Tk.Entry(self.frames['RMS'])
    self.entries['RMS'].insert(0,"0.001")
    self.entries['RMS'].config(width=5)

    self.entries['Flux'] = Tk.Entry(self.frames['Flux'])
    self.entries['Flux'].insert(0,"1.0")
    self.entries['Flux'].config(width=5)


    Htext.pack(side=Tk.LEFT)
    self.entries['Hyper'].pack(side=Tk.RIGHT)

    Ntext.pack(side=Tk.LEFT)
    self.entries['Niter'].pack(side=Tk.RIGHT)

    Rtext.pack(side=Tk.LEFT)
    self.entries['RMS'].pack(side=Tk.RIGHT)

    Stext.pack(side=Tk.LEFT)
    self.entries['Flux'].pack(side=Tk.RIGHT)


    self.frames['MMOpt'].pack(side=Tk.LEFT)
    self.canvas1.get_tk_widget().pack(side=Tk.LEFT) #, fill=Tk.BOTH, expand=1)

    self.buttons = {}

    self.buttons['MEM'] = Tk.Button(self.frames['MMOpt'],text="MEM!",command=self._MEM)
    self.buttons['reset'] = Tk.Button(self.frames['MMOpt'],text="RELOAD",command=self._reset)


    self.frames['Hyper'].pack(side=Tk.TOP)
    self.frames['Niter'].pack(side=Tk.TOP)
    self.frames['RMS'].pack(side=Tk.TOP)
    self.frames['Flux'].pack(side=Tk.TOP)

    self.buttons['MEM'].pack(side=Tk.TOP)
    self.buttons['reset'].pack(side=Tk.TOP)


    separator = Tk.Frame(self.frames['MMOpt'],height=4, bd=5, relief=Tk.SUNKEN)
    separator.pack(fill=Tk.X, padx=10, pady=20,side=Tk.TOP)

    self.buttons['save'] = Tk.Button(self.frames['MMOpt'],text="SAVE IMAGE",command=self._SAVE)
    self.buttons['save'].pack(side=Tk.TOP)

    self._reset()




  def _reset(self):
   
    self.ResidPlot.cla()
    self.MEMPlot.cla()

    try:
      FLUX = float(self.entries['Flux'].get())
    except:
      showinfo('ERROR!','Please, check the content of all quantities!\nShould be numbers!')
      return      


    NPIX = self.parent.Npix
    self.UVCov[:] = np.fft.fftshift(self.parent.GridUVCov)
    self.UVData[:] = np.fft.fftshift(self.parent.GridUV)
    self.MEM[:] = FLUX/float(NPIX*NPIX/4)
    
    self.Dirty[:] = self.parent.Dirty
    self.residuals[:] = np.copy(self.Dirty)

    self.POS = np.where(np.abs(self.UVCov)>0.0)
    self.Resid = np.zeros(len(self.POS[0]),dtype=np.complex128)
    self.Wgt = np.zeros(len(self.POS[0]))
    self.Wgt[:] = np.abs(self.UVCov[self.POS])


    WgtFac = self.parent.WgtScale/np.sum(self.Wgt)
 
    self.Wgt *= WgtFac
    self.UVData *= WgtFac
    self.UVCov *= WgtFac


  #  self.ScaleFactor = np.sum(self.Wgt)/np.sum(np.fft.ifft2(np.fft.fftshift(self.parent.PSF)).real)


    if self.parent.MemoryHungry:
      ANGLE = 2.*np.pi*(self.POS[0][:,np.newaxis,np.newaxis]*self.Xpix[np.newaxis,:,:]+self.POS[1][:,np.newaxis,np.newaxis]*self.Ypix[np.newaxis,:,:])/NPIX
      self.COS = np.cos(ANGLE)
      self.SIN = np.sin(ANGLE)
      del ANGLE


    self.q = np.sum(self.Wgt)

    self.dx = 4.*self.parent.Xaxmax/self.parent.Npix/self.parent.xfac

    self.fmtD2 = r'Resid. Peak: % .2e Jy/beam' + '\n' + r'rms: % .2e Jy/beam'
    self.fmtDC = r'MEM Peak: % .3e Jy/pixel ' + '\n' + r'Flux density (above noise): %.3e Jy'

    dslice = self.residuals[self.X0:self.X1, self.X0:self.X1]

    self.Xmax = self.parent.Xaxmax
    self.ResidPlotPlot = self.ResidPlot.imshow(dslice,interpolation='nearest',picker=True, cmap=self.parent.currcmap, 
                            extent = (self.Xmax,-self.Xmax,-self.Xmax,self.Xmax))

    self.RMS = np.sqrt(np.var(dslice)+np.average(dslice)**2.)
    self.PEAK = np.max(dslice)
    modflux = dslice[self.X0,self.X0]
 
    self.pickcoords = [NPIX/2, NPIX/2,0.,0.]
    self.ResidText = self.ResidPlot.text(0.05,0.92,self.fmtD2%(self.PEAK,self.RMS),
         transform=self.ResidPlot.transAxes,bbox=dict(facecolor='white', alpha=0.7))

    self.MEMText = self.MEMPlot.text(0.05,0.92,'',transform=self.MEMPlot.transAxes,bbox=dict(facecolor='white', alpha=0.7))


    self.ResidPlot.set_ylabel('Dec offset '+self.parent.unit)
    self.ResidPlot.set_xlabel('RA offset '+self.parent.unit)
    self.ResidPlot.set_title('RESIDUALS')


    dslice = self.MEM  
    self.MEMPlotPlot = self.MEMPlot.imshow(dslice,interpolation='nearest',picker=True, cmap=self.parent.currcmap, 
                            extent = (self.Xmax,-self.Xmax,-self.Xmax,self.Xmax))

    self.MEMPEAK = 0.0
    self.totiter = 0

    self.MEMPlot.set_ylabel('Dec offset '+self.parent.unit)
    self.MEMPlot.set_xlabel('RA offset '+self.parent.unit)
    self.MEMPlot.set_title('MEM MODEL (0 ITER)')
  #  self.MEMText = self.MEMPlot.text(0.05,0.92,'',transform=self.MEMPlot.transAxes,bbox=dict(facecolor='white', alpha=0.7))


    self.totiter = 0
    self.alpha = 0.0
    self.canvas1.draw()
    pl.draw()

    return






  def _SAVE(self):

    filename = os.path.join(self.parent.userdir,'%s_MEM_%i.fits'%(self.parent.snam, int(dt.time())))
    if not os.path.exists(self.parent.userdir):
      os.makedirs(self.parent.userdir)

    tosave = self.Model[self.X0:self.X1, self.X0:self.X1]

    hdu = pf.PrimaryHDU(np.copy(tosave[::-1,:,np.newaxis,np.newaxis]).transpose(2,3,0,1))
    hdulist = pf.HDUList([hdu])
    head = hdulist[0].header
    self.parent._fillHeader(head, self.dx/3600., 'Jy/pixel')
    if os.path.exists(filename):
      os.remove(filename)
    hdulist.writeto(filename)

    showinfo('INFO','\n\nFITS MEM image saved succesfully!\n\nName: %s\n'%os.path.basename(filename))



    return


  def _MEM(self):

#### MAXIMUM ENTROPY DECONVOLVER


## READ THE DECONVOLVER PARAMETERS:
    try:
      Damping = float(self.entries['Hyper'].get())
      niter = int(self.entries['Niter'].get())
      RMS = float(self.entries['RMS'].get())
      TOTALFLUX = float(self.entries['Flux'].get())
    except:
      showinfo('ERROR!','Please, check the content of all quantities!\nShould be numbers!')
      return


# SOME CONSTANTS:
  # Normalization factors:
    QNorm = self.q*self.q 
    NpixSq = float(self.parent.Npix**2) 
    AvgWgt = np.average(self.Wgt)
    NPIX = self.parent.Npix


# The RMS will try to be kept to the expected value within this window:
    CHI_DISP = 0.10

# LOWEST VALUE IN THE IMAGE:
    CUTOFF = RMS/(NPIX*NPIX/4)

# INITIAL PIXEL VALUES (I.E: ALL PIXELS WILL HAVE THE SAME INTENSITY):
    INIFAC = TOTALFLUX/(NPIX*NPIX/4.)

# Array to store the temporary FFT of the residuals:
    FFTResids = np.zeros((NPIX,NPIX),dtype=np.complex128)


#########
## First iteration only:
    if self.totiter == 0:
      self.MEM[:] = CUTOFF 
## NOTICE THAT ONLY THE INNER HALF OF THE IMAGE IS USED IN THE
## DECONVOLUTION (TO AVOID ALIASING AT THE EDGES), JUST AS IN CLEAN:
      self.MEM[self.X0:self.X1,self.X0:self.X1] = INIFAC

## FIRST RESIDUALS ARE THE DATA:
      self.Resid[:] = self.UVData[self.POS]
      Chi2_0 = np.sum(self.Resid*np.conjugate(self.Resid)).real/QNorm
    #  print 'FIRST RMS0: ', np.sqrt(Chi2_0), np.sum(self.Wgt), np.max(self.Wgt),np.min(self.Wgt), AvgWgt
########

# FFT OF THE CURRENT MODEL:
    self.MODFFT[:] = np.fft.ifft2(np.fft.fftshift(self.MEM))

# GRIDDED RESIDUAL VISIBILITIES (ONLY NON-EMPTY UV PIXELS ARE USED):
    self.Resid[:] = (self.UVData[self.POS] - self.Wgt*self.MODFFT[self.POS])

    Chi2 = np.sum(self.Resid*np.conjugate(self.Resid)).real/QNorm

    if not self.parent.MemoryHungry:
      ANGLE = np.zeros(np.shape(self.Xpix))
      dPOS = [2.*np.pi*np.array(self.POS[0],dtype=np.float64),2.*np.pi*np.array(self.POS[1],dtype=np.float64)]
      TempChi = np.zeros(np.shape(self.Xpix))


    for k in range(niter):

## NEW FLUX IS THE PIXEL SUM WITHIN THE INNER HALF OF THE IMAGE:
      FLUXNew = np.sum(self.MEM[self.X0:self.X1,self.X0:self.X1])

## NORMALIZE NEW RESIDUALS:
      self.Resid[:] /=  self.Wgt

# The robust weighting is not working for MEM (WHY???). Will FORCE Uniform to compute the ChiSq gradient:
      if self.parent.MemoryHungry:
        self.Chi2Grad[:] = -np.transpose(np.fft.fftshift(2.*(np.sum((self.Resid[:,np.newaxis,np.newaxis].real*self.COS + self.Resid[:,np.newaxis,np.newaxis].imag*self.SIN), axis=0))))*self.q
      else:
        TempChi[:] = 0.0
        for psi in range(len(dPOS[0])):
          ANGLE[:] = (dPOS[0][psi]*self.Xpix[:,:]+dPOS[1][psi]*self.Ypix[:,:])/NPIX
          TempChi[:] += self.Resid[psi].real*np.cos(ANGLE)+self.Resid[psi].imag*np.sin(ANGLE)
        self.Chi2Grad[:] = -2.*np.transpose(np.fft.fftshift(TempChi))*self.q




#      self.Chi2Grad[:] = np.transpose(np.fft.fftshift(2.*(np.sum((self.Resid[:,np.newaxis,np.newaxis].real*self.COS + self.Resid[:,np.newaxis,np.newaxis].imag*self.SIN)*self.Wgt[:,np.newaxis,np.newaxis]/AvgWgt, axis=0))))*self.q


## ENTROPY GRADIENT:
      self.EntrGrad[:] = (INIFAC + np.log(self.MEM/INIFAC))

   #   print len(self.POS[0]), self.q, np.max(np.abs(self.Chi2Grad))


###############
### FIRST ITERATION ONLY.
### WILL FIND A SELF-CONSISTENT ALPHA AND METRIC TO START:
      if self.totiter==0:
        self.Metric[:] = 1.
        for kk in range(100):
          self.alpha = np.sqrt(4.*np.sum(self.EntrGrad*self.EntrGrad*self.Metric)/np.sum(self.Chi2Grad*self.Chi2Grad*self.Metric))
          self.Metric[:] = 1./(INIFAC/self.MEM + 2.*self.alpha*self.q)
##############


##############
#### LAGRANGE MULTIPLIERS:

### SET THE METRIC FROM THE CURRENT MEM IMAGE AND ALPHA VALUE:
      self.Metric[:] = 1./(INIFAC/self.MEM + 2.*self.alpha*self.q)
### SET BETA TO FORCE THE TOTAL FLUX:
      beta = -((TOTALFLUX*NpixSq-FLUXNew) + np.sum((-self.EntrGrad*Damping +self.alpha*self.Chi2Grad)*self.Metric))/np.sum(self.Metric)
     # print 'Alpha, beta:',self.alpha, beta

##############


#############
# JUST FOR TESTING:      
#      alpha = 1.0
#      Damping = 0.0
#      self.Metric[:] = 1./(2.*alpha*self.q)
#############


      Db = (-self.alpha*self.Chi2Grad + self.EntrGrad*Damping - beta)*self.Metric
    #  print 'Change in beta: ', np.sum(Db), (TOTALFLUX*NpixSq-FLUXNew)
      self.MEM += Db 

      if True:
### SET OUTER HALF OF IMAGE TO "ZERO" (A.K.A., THE MINIMUM POSSIBLE VALUE):
        self.MEM[:self.X0,:] = CUTOFF
        self.MEM[self.X1+1:,:] = CUTOFF
        self.MEM[:,:self.X0] = CUTOFF
        self.MEM[:,self.X1+1:] = CUTOFF

### IF ANY NEGATIVE OR NULL PIXEL IS FOUND, SET IT TO THE MINIMUM ALLOWED VALUE:
        isOut = self.MEM<=CUTOFF
        self.MEM[isOut] = CUTOFF

### ONLY GOOD PIXELS CONTRIBUTE TO THE FLUX:
        isSource = np.logical_not(isOut)
        FLUXNew = np.sum(self.MEM[isSource])
        FLUXNewTot = np.sum(self.MEM)

     #   print 'FLUXSCALE: ',TOTALFLUX, FLUXNew, TOTALFLUX/FLUXNew, NpixSq, len(self.POS[0])

        del isOut, isSource


### GRIDDED MODEL VISIBILITIES AND RESIDUALS:
      self.MODFFT[:] = np.fft.ifft2(np.fft.fftshift(self.MEM))
      self.Resid[:] = (self.UVData[self.POS] - self.Wgt*self.MODFFT[self.POS])
      Chi2New = np.sum(self.Resid*np.conjugate(self.Resid)).real/QNorm

### CHI2 IS RELATED TO THE RMS (PARSEVAL THEOREM):
      RelChi = (Chi2New-RMS**2.)/RMS**2. 

### UPDATE THE ALPHA VALUE, DEPENDING ON THE CHI2 (AS COMPARED TO ITS EXPECTED VALUE):
      alphaNew = self.alpha
      if RelChi > CHI_DISP:
         alphaNew = self.alpha + np.abs(float(Chi2-RMS**2.))/float(np.sum(self.Chi2Grad*self.Chi2Grad*self.Metric))
      elif RelChi < -CHI_DISP:
         alphaNew = self.alpha - np.abs(float(Chi2-RMS**2.))/float(np.sum(self.Chi2Grad*self.Chi2Grad*self.Metric))
      Change = np.abs((alphaNew-self.alpha)*np.sum(-self.Chi2Grad*self.Metric))
    #  print 'alpha New/Old: ',alphaNew, self.alpha, Change

      if Change < 0.05*RMS and np.abs(alphaNew)<1.e6:
    #    print 'alpha New/Old: ',alphaNew, self.alpha
        self.alpha = alphaNew

      Chi2 = Chi2New


### UPDATE THE MODEL IMAGE IN THE GUI:
      if True: # self.totiter%2==0 or k==niter-1:

        self.Model[:] = self.MEM/NpixSq

        FFTResids[:] = 0.0
        FFTResids[self.POS] = self.Resid
        self.residuals[:] = np.fft.fftshift(np.fft.fft2(FFTResids).real)/self.q
        self.PEAK = np.max(self.residuals[self.X0:self.X1,self.X0:self.X1])
        self.RMS = np.sqrt(np.std(self.residuals[self.X0:self.X1,self.X0:self.X1])**2.+ np.average(self.residuals[self.X0:self.X1,self.X0:self.X1])**2.)
        self.ResidText.set_text(self.fmtD2%(self.PEAK,self.RMS))

        self.ResidPlotPlot.set_array(self.residuals[self.X0:self.X1,self.X0:self.X1])
        self.ResidPlotPlot.norm.vmin = np.min(self.residuals[self.X0:self.X1,self.X0:self.X1])
        self.ResidPlotPlot.norm.vmax = np.max(self.residuals[self.X0:self.X1,self.X0:self.X1])


        self.MEMPlotPlot.set_array(self.Model[self.X0:self.X1,self.X0:self.X1])
        self.MEMPlotPlot.norm.vmin = np.min(self.Model[self.X0:self.X1, self.X0:self.X1])
        self.MEMPlotPlot.norm.vmax = np.max(self.Model[self.X0:self.X1, self.X0:self.X1])
        self.MEMPlot.set_title('MEM MODEL (%i ITER)'%(self.totiter))

        self.MEMText.set_text(self.fmtDC%(np.max(self.Model[self.X0:self.X1, self.X0:self.X1]),np.sum(self.Model[self.X0:self.X1, self.X0:self.X1]) ))
        self.canvas1.draw()
        self.canvas1.flush_events()

        FLUX = np.sum(self.Model[self.X0:self.X1,self.X0:self.X1])

 
    #    print 'RMS it %i: %.3e ; TOTAL FLUX: %.3e'%(k,np.sqrt(Chi2),FLUX)


      self.totiter += 1
      

    return














if __name__ == "__main__":

  root = Tk.Tk()
  TITLE = 'Aperture Synthesis Analysis Tool (I. Marti-Vidal, University of Valencia) - version  %s'%__version__
  root.wm_title(TITLE)


## Needed for MacOS to work:
  try:
    root.state('zoomed')
  except:
    root.attributes('-zoomed',True)


  m = list(root.maxsize())
  m[1]-=100
  root.geometry('{}x{}+0+0'.format(*m))

  myint = dataTool(tkroot=root)
  Tk.mainloop()
