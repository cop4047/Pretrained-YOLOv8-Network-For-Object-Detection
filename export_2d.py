import subprocess
import sys

script = r'C:\Users\alros\AppData\Roaming\Python\Python311\site-packages\nuscenes\scripts\export_2d_annotations_as_json.py'

subprocess.run([
    sys.executable, script,
    '--dataroot', 'D:/Downloads/mini',
    '--version',  'v1.0-mini',
    '--filename', 'image_annotations.json'
])
