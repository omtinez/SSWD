
# coding: utf-8

# In[12]:

get_ipython().magic('matplotlib inline')

import re
import os
import datetime

import tqdm
import PIL.Image
import PIL.ExifTags

import numpy as np
import pandas as pd
import seaborn as sns


# In[20]:

import subprocess

class VideoEncoder:
    
    def __init__(self, ffmpeg_path='ffmpeg'):
        self.ffmpeg = ffmpeg_path
        self.proc = None
        
    def start(self, outfile, framerate=1, vcodec='mjpeg'):
        if self.proc:
            raise RuntimeError('Calling "start()" before process pipe was closed.')

        cmd = (self.ffmpeg,
                '-y',
                '-r', str(framerate),
                '-f','image2pipe',
                '-vcodec', vcodec,
                '-i', 'pipe:', 
                '-vcodec', 'libxvid',
                re.sub(r'[^\d\w\.]', '_', outfile))
        
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        
    def encode(self, framelist):
        ''' `framelist` must be an iterable of raw bytes '''
        for frame in framelist:
            self.proc.stdin.write(frame)
            
    def close(self):
        self.proc.stdin.close()
        self.proc = None


# In[3]:

# Build a map of filename -> timestamp, camera, resolution
TS_FORMAT = '%Y:%m:%d %H:%M:%S'
ts_map = {}
for fname in [f for f in os.listdir('session') if os.path.isfile(os.path.join('session', f))]:
    img = PIL.Image.open(os.path.join('session', fname))
    exif = {
        PIL.ExifTags.TAGS[k]: v
        for k, v in img._getexif().items()
        if k in PIL.ExifTags.TAGS
    }
    timestamp = datetime.datetime.strptime(exif.get('DateTime'), TS_FORMAT)
    ts_map[fname] = (
        timestamp,
        exif.get('Model').replace('\x00', ''),
        '%dx%d' % (exif.get('ExifImageHeight'), exif.get('ExifImageWidth')))


# In[4]:

# Separate data based on camera type
df = pd.DataFrame.from_dict(ts_map, orient='index').reset_index()    .rename(columns={'index': 'filename', 0: 'timestamp', 1: 'camera', 2: 'resolution'}).sort_values('timestamp')
cameras = df['camera'].unique().tolist()

df_ = {}
for camera in cameras:
    df_[camera] = df[df['camera'] == camera]
    
print('Cameras:', cameras)
df_[cameras[0]].head()


# In[ ]:

enc = VideoEncoder()
for camera in cameras:
    df = df_[camera]
    
    # Compute the difference between frames
    df['timediff'] = df['timestamp'].diff().astype('timedelta64[s]')
    
    # Assign values to contiguous frames
    tmp = df['timediff'] > 120
    df['framegroup'] = tmp.astype(int).cumsum()
    
    # Filter out frames with bogus resolution
    df['filter'] = 1
    df['area'] = df['resolution'].apply(lambda x: int(x.split('x')[0]) * int(x.split('x')[1]))
    for group in df['framegroup'].unique():
        ix = df['framegroup'] == group
        median = df.loc[ix, 'area'].median()
        df.loc[ix, 'filter'] = (df.loc[ix, 'area'] == median).astype(int)
        
        ix = (df['framegroup'] == group) & df['filter']
        frames = df.loc[ix]
        if len(frames) < 10: continue

        mintime = frames['timestamp'].min()
        maxtime = frames['timestamp'].max()

        avediff = frames['timediff'].mean()
        framerate = 8.0 * 60.0 / avediff

        print('Time span: %s - %s (%s)' % (mintime, maxtime, (maxtime - mintime)))
        print('Framerate: %f (~%f)' % (framerate, avediff))
        
        frames = df[df['filter'] & (df['framegroup'] == group)]
        if len(frames) < 10: continue
        framerate = 8.0 * 60.0 / frames['timediff'].mean()
        enc.start('%s_%s_seastars.mp4' % (camera, mintime), framerate=framerate)
        for fname in tqdm.tqdm(frames['filename'][:]):
            img = open(os.path.join('session', fname), 'rb')
            enc.encode([img.read()])
        enc.close()

df[cameras[0]].head()

