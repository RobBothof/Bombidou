""" this script parses the ascii and stokefiles from the iam dataset, and creates a pickle datafile with matching stroke and asciitext lines """

import os,sys
import pickle
import numpy as np
import xml.etree.ElementTree as ET
import random
import matplotlib.pyplot as plt

data_filename = "training_data.pkl"

min_stroke_points = 512 ##timesteps
min_ascii_length = 10

### we expect about 25 points per character

print ("Parsing dataset...")
print ()

### not all asciifiles have corresponding strokefiles, so we use the stroke xmlfiles and match ascci files to those
xmlfiles= []

for dirName, subdirList, fileList in os.walk("./lineStrokes"):
    for fname in fileList:
        xmlfiles.append(dirName+"/"+fname)             

print(str(len(xmlfiles)) + " xml strokefiles found.")   

### match strokefiles and ascifiles, build 2 matching lists
stroke_data = []
ascii_data = []

for i in range(len(xmlfiles)):
    if (xmlfiles[i][-3:] == 'xml'):
        xml_root = ET.parse(xmlfiles[i]).getroot()

        ###first get all strokes for the xml stroke file
        strokes = []
        x_offset = 1e20
        y_offset = 1e20
        y_height = 0
        for j in range(1, 4):
            x_offset = min(x_offset, float(xml_root[0][j].attrib['x']))
            y_offset = min(y_offset, float(xml_root[0][j].attrib['y']))
            y_height = max(y_height, float(xml_root[0][j].attrib['y']))
        y_height -= y_offset
        x_offset -= 100
        y_offset -= 100

        for s in xml_root[1].findall('Stroke'):
            points = []
            for p in s.findall('Point'):
                points.append([float(p.attrib['x'])-x_offset,float(p.attrib['y'])-y_offset])
            strokes.append(points)

        ###convert stokes into a 2d numpy int16 array
        n_point = 0
        for p in range(len(strokes)):
            n_point += len(strokes[p])

        strokes_array = np.zeros((n_point, 3), dtype=np.int16)

        prev_x = 0
        prev_y = 0
        counter = 0

        for j in range(len(strokes)):
            for k in range(len(strokes[j])):
                strokes_array[counter, 0] = int(strokes[j][k][0]) - prev_x
                strokes_array[counter, 1] = int(strokes[j][k][1]) - prev_y
                prev_x = int(strokes[j][k][0])
                prev_y = int(strokes[j][k][1])
                strokes_array[counter, 2] = 0
                if (k == (len(strokes[j])-1)): # end of stroke
                    strokes_array[counter, 2] = 1
                counter += 1


        ###get the correcsponding line of asciifile and line of text that matches the stroke_array 
        ascii_file = xmlfiles[i].replace("lineStrokes","ascii")[:-7] + ".txt"
        line_number = xmlfiles[i][-6:-4]
        line_number = int(line_number) - 1
        ascii = ""

        with open(ascii_file, "r") as f:
            s = f.read()

        s = s[s.find("CSR"):]

        if len(s.split("\n")) > line_number+2:
            ascii = s.split("\n")[line_number+2]

        ### if the line of text is long enough
        if len(ascii) > min_ascii_length:
            ###if we have enough stroke points
            if len(strokes_array) > min_stroke_points+2:

                ### store this the stroke array and ascii-line
                stroke_data.append(strokes_array)
                ascii_data.append(ascii)
                sys.stdout.write('.')
                sys.stdout.flush()
            else:
                print()
                print ("======>>>> Not enough Stroke Points Line was: " + ascii)

        else:
            print()
            print ("======>>>> Line length was too short. Line was: " + ascii)



assert(len(stroke_data)==len(ascii_data)), "There should be a 1:1 correspondence between stroke data and ascii labels."




f = open(data_filename,"wb")
pickle.dump([stroke_data,ascii_data], f, protocol=2)
f.close()
print()
print ("Finished parsing dataset. Saved {} lines".format(len(stroke_data)))

# except:
    # print("could not find or load data")