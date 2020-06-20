import cv2
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np

m = 0
wm = 0
im = 0

for i in range(853):

    tree = ET.parse(f'annotations/maksssksksss{i}.xml')

    root = tree.getroot()
    folder = root.find('folder').text
    filename = root.find('filename').text
    size = root.find('size')
    width, height, depth = [int(i.text) for i in size]
    objects = root.findall('object')

    img = cv2.imread(f'{folder}\{filename}', 1).astype(np.uint8)

    for obj in objects:

        name = obj.find('name').text
        pose = obj.find('pose').text
        truncated = int(obj.find('truncated').text)
        occluded = int(obj.find('occluded').text)
        difficult = int(obj.find('difficult').text)
        x_min, y_min, x_max, y_max = [int(i.text) for i in obj.find('bndbox')]

        face_img = img[y_min:y_max, x_min:x_max]
        
        face_img = cv2.cvtColor(face_img,cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(face_img)
        pil_im = pil_im.resize((224,224), Image.LANCZOS)

        if name=='with_mask':
            pil_im.save(f'dataset\with_mask\image{m}.png')
            m += 1
        elif name=='without_mask':
            pil_im.save(f'dataset\without_mask\image{wm}.png')
            wm += 1
        else:
            pil_im.save(f'dataset\incorrect_mask\image{im}.png')
            im += 1

print(m, wm, im)
