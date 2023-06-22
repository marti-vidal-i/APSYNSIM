# -*- mode: python -*-

# SPEC FILE FOR PYINSTALLER:

a = Analysis(['z:\\APSYNSIM\\APSYNTRUE\\SCRIPT\\APSYNTRUE.py'],
             pathex=['Z:\\APSINSYM_WIN'],
             hiddenimports=['scipy.special._ufuncs_cxx','mpl_toolkits','mpl_toolkits.basemap'],
             hookspath=None,
             runtime_hooks=None)
pyz = PYZ(a.pure)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='APSYNSIM.exe',
          debug=False,
          strip=None,
          upx=True,
          console=True , icon='z:\\APSYNSIM\\COMPILE\\APSYNTRUE.ico')

import os
import mpl_toolkits.basemap

coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas, 
               strip=None,
               upx=True,
               name='APSYNSIM')

