import cv2
import xml.etree.ElementTree as ET

m = 0
wm = 0

for i in range(853):
    
    tree = ET.parse(f'annotations/maksssksksss{i}.xml')
    
    root = tree.getroot()
    folder = root.find('folder').text
    filename = root.find('filename').text
    size = root.find('size')
    width, height, depth = [int(i.text) for i in size]
    objects = root.findall('object')
    
    img = cv2.imread(f'{folder}\{filename}', -1)
    
    for obj in objects:
        
        name = obj.find('name').text
        pose = obj.find('pose').text
        truncated = int(obj.find('truncated').text)
        occluded = int(obj.find('occluded').text)
        difficult = int(obj.find('difficult').text)
        x_min, y_min, x_max, y_max = [int(i.text) for i in obj.find('bndbox')]
        
        face_img = img[y_min:y_max, x_min:x_max]
        
        if name=='with_mask':
            cv2.imwrite(f'with_mask\image{m}.png', face_img)
            m += 1
        elif name=='without_mask':
            cv2.imwrite(f'without_mask\image{m}.png', face_img)
            wm += 1
        