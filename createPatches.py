import numpy as np
from xml.dom import minidom
import matplotlib.path as mplPath
import matplotlib.pyplot as plt
import matplotlib.patches as patches

xmldoc = minidom.parse('/home/ghost/Desktop/annotations/tumor_078.xml')
itemlist = xmldoc.getElementsByTagName('Annotation')
from openslide import * 

def getTumor(tumorName) : 
 
    waypoints=[]

    count = 0
    for s in itemlist:
        if(tumorName == s.attributes['Name'].value) :    #JUST TUMOR 1
            points = s.getElementsByTagName('Coordinate')
            for t in points :   #83
                
                a = float(t.attributes['X'].value)
                b = float(t.attributes['Y'].value)

                waypoints.append([a,b])

    return mplPath.Path(np.asarray(waypoints))

img = OpenSlide('/home/ghost/Desktop/tumor_078.tif')

def getRegion(name, patchIndex, polygon) : 

    region = img.read_region(location=(int(polygon.vertices[patchIndex][0]-100),int(polygon.vertices[patchIndex][1]-100)), level= 0, size=(200,200))
    region = np.asarray(region.convert('RGB'))

    matrix = np.zeros((200,200))

    for i in range(200) :
        for j in range(200) :

            x = polygon.vertices[patchIndex][0]+(i-100)
            y = polygon.vertices[patchIndex][1]+(j-100)

            if polygon.contains_point((x, y)) : 
                matrix[j][i] = 255

    im = Image.fromarray(matrix).convert('RGB')
    im2 = Image.fromarray(region).convert('RGB')
    im.save(f'./dataset/{name}_gt.png')
    im2.save(f'./dataset/{name}_im.png')    

count = 0
for j in range(200) :     
    name = f'Tumor{j+1}'
    polygon = getTumor(name)        
    for i in range(len(polygon.vertices)) : 
        getRegion(count, i, polygon)
        count+=1
        print(count)



