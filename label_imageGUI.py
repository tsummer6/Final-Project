# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import numpy as np
import cv2
import csv
import tensorflow.compat.v1 as tf
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QGridLayout
from PyQt5 import QtGui
from PIL.ImageQt import ImageQt
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image

tf.disable_eager_execution()


def load_graph(model_file):
    '''
    This function reads in a .bff file and returns the
    information on the board to be constructed.

    **Parameters**
        model_file: *str*
            name of the file to be read with its extension

    **Returns**

        graph: *list of str*
            a list of strings describing the grid to be constructed.
            Each character is a cell in the board.
    '''

    graph = tf.Graph()
    graph_def = tf.GraphDef()
    # Open the model file in the binary protobuf format
    with open(model_file, "rb") as f:
        # Read data into a buffer
        # Convert to GraphDef object from data
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        # Convert to Graph object from GraphDef object
        tf.import_graph_def(graph_def)

    return graph


def read_tensor_from_image_file(file_name, input_height=299, input_width=299, input_mean=0, input_std=255):
    '''
    The below code just opens an image file in the bmp,
    gif, png or jpg format and reads the file.

    **Parameters**
        file_name: *str*
            name of the file to be read with its extension
        input_height : *str*
            name of the file to be read with its extension
        input_width : *str*
            name of the file to be read with its extension
        input_mean: *str*
            name of the file to be read with its extension
        input_std : *str*
            name of the file to be read with its extension

    **Returns**

        result: *numpy ndarray*
    '''
    input_name = "file_reader"
    # output_name = "normalized"
    # Operation to open and read the image file to the buffer
    file_reader = tf.read_file(file_name, input_name)
    # Decode the data as png, gif, bmp or jpeg based on the file extension
    if file_name.endswith(".png"):
        image_reader = tf.image.decode_png(
            file_reader, channels=3, name='png_reader')
    elif file_name.endswith(".gif"):
        image_reader = tf.squeeze(
            tf.image.decode_gif(file_reader, name='gif_reader'))
    elif file_name.endswith(".bmp"):
        image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
    else:
        image_reader = tf.image.decode_jpeg(
            file_reader, channels=3, name='jpeg_reader')
    # Cast the uint8 data to float32 to prepare for the range conversion later
    float_caster = tf.cast(image_reader, tf.float32)
    # Add one more dimension in axis 0. For example,
    # if the original image has a dimension 1024x800x3,
    # it will become 1x1024x800x3
    dims_expander = tf.expand_dims(float_caster, 0)
    # Resize image 1x299x299x3 to using bilinear interpolation.
    resized = tf.image.resize_bilinear(
        dims_expander, [input_height, input_width])
    # Normalize data. By default, shift the value from 0–255 to 0–1
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    # Run the session to actually read the data and return in a numpy ndarray.
    sess = tf.Session()
    result = sess.run(normalized)

    return result


def load_labels(label_file):
    '''
    This function reads in a .bff file and returns the
    information on the board to be constructed.

    **Parameters**
        label_file: *str*
            name of the file to be read with its extension

    **Returns**

        label: *list of str*
            a list of strings describing the grid to be constructed.
            Each character is a cell in the board.
    '''
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label


def find_match(file_name, model_file, label_file):
        '''
        This code will load the retrained model and the reading in
        the image file to give that image its label

        **Parameters**
        file_name : *str*
            name of the image file be labeled with the model
        model_file : *str*
            name of the file containing the retrained model
        label_file: *str*
            name of the file containing the labels of each pokemon

        **Returns**

        labels[idx]: *str*
            The pokemon name are model decided was the most likely match
        results[idx] : *float*
            This is a percent of confidence that the pokemon is
            the correct one
        '''
        input_height = 224
        input_width = 224
        input_mean = 128
        input_std = 128
        input_layer = "input"
        output_layer = "final_result"

        graph = load_graph(model_file)
        t = read_tensor_from_image_file(file_name,
                                        input_height=input_height,
                                        input_width=input_width,
                                        input_mean=input_mean,
                                        input_std=input_std)

        input_name = "import/" + input_layer
        output_name = "import/" + output_layer
        input_operation = graph.get_operation_by_name(input_name)
        output_operation = graph.get_operation_by_name(output_name)

        with tf.Session(graph=graph) as sess:
            # start = time.time()
            results = sess.run(
                output_operation.outputs[0], {input_operation.outputs[0]: t})
            # end = time.time()
        results = np.squeeze(results)
        labels = load_labels(label_file)
        val, idx = max((val, idx) for (idx, val) in enumerate(results))

        return labels[idx], results[idx]


def get_data(pic_name):
    try:
        # We can open image the image file for the maze
        img = Image.open(pic_name)
        new_img = img.resize((400, 400))
        with open('PokemonData.csv', 'r') as f:
            reader = csv.reader(f)
            PmkData = list(reader)
        name, percent = find_match(
            pic_name, "training/retrained_graph2.pb",
            "training/retrained_labels2.txt")

        # If an image does not have at least an 80% confidence
        # it will be marked as invalid
        if(percent < 0.80):
            Name = "\nName:\t\t\tINVALID" + "\n\n"
            Type = "\nType:\t\t\tINVALID" + "\n\n"
            Total = "\nTotal:\t\t\tINVALID" + "\n\n"
            HP = "\nHP:\t\t\tINVALID" + "\n\n"
            Att = "\nAttack:\t\t\tINVALID" + "\n\n"
            Def = "\nDefense\t\t\tINVALID" + "\n\n"
            SpAtt = "\nSp. Attack:\t\tINVALID" + "\n\n"
            SpDef = "\nSp. Defense:\t\tINVALID" + "\n\n"
            Spd = "\nSpeed:\t\t\tINVALID" + "\n\n"
        # Otherwise the name will be used to extract information
        # from th csv file
        else:
            for i in range(0, len(PmkData)):
                if(name.capitalize() == PmkData[i][0]):
                    Name = "\nName:\t\t\t" + PmkData[i][0] + "\n\n"
                    Type = "\nType:\t\t\t" + PmkData[i][1] + "\n\n"
                    Total = "\nTotal:\t\t\t" + PmkData[i][2] + "\n\n"
                    HP = "\nHP:\t\t\t" + PmkData[i][3] + "\n\n"
                    Att = "\nAttack:\t\t\t" + PmkData[i][4] + "\n\n"
                    Def = "\nDefense\t\t\t" + PmkData[i][5] + "\n\n"
                    SpAtt = "\nSp. Attack:\t\t" + PmkData[i][6] + "\n\n"
                    SpDef = "\nSp. Defense:\t\t" + PmkData[i][7] + "\n\n"
                    Spd = "\nSpeed:\t\t\t" + PmkData[i][8] + "\n\n"
        # The information is stored in this label
        DataLabel = Name + Type + Total
        DataLabel = DataLabel + HP + Att + Def + SpAtt
        DataLabel = DataLabel + SpDef + Spd
    except IOError:
        # Print this error message if the file is not an image file
        messagebox.showinfo("Error", "File is not an image file.")

    return new_img, DataLabel


class Pokedex(QWidget):
        '''
        Pokedex class contains the  method to read in an image of a Pokemon and
        test it accordingly to see that Pokemon's information

        Methods included in the class are:
            - OpenWebcam: Opens a webcam and allows the user to take a
                picture with it
            - UploadImage: Allow the user to load a picture from their machine
        '''

        def __init__(self):
                super().__init__()
                self.title = 'Pokedex'
                self.width = 500
                self.height = 250
                self.initUI()

        def initUI(self):
                '''
                This function will set the initial GUI before the user inputs
                any data

                **Parameters**

                    None
                **Returns**

                    None
                '''
                # Set the title of the GUI
                self.setWindowTitle(self.title)
                # Resize the GUI
                self.resize(self.width, self.height)

                self.layout = QGridLayout()
                self.setLayout(self.layout)

                # Set the two Buttons
                # and the calls for when the user clicks on them
                self.btn1 = QPushButton('Webcam Upload')
                self.btn1.clicked.connect(self.OpenWebcam)
                self.btn2 = QPushButton('Manual Upload')
                self.btn2.clicked.connect(self.UploadImage)
                self.layout.addWidget(self.btn1, 0, 0)
                self.layout.addWidget(self.btn2, 0, 1)
                # Adding the label for the picture of the
                # Pokemon
                self.img = QLabel(self)
                self.layout.addWidget(self.img, 1, 1)
                # Setting the Label for the Pokemon Infomartion
                self.label = QLabel(self)
                self.label.setText(
                    "\nName:\nType:\nTotal:\nHP:\nAttack:\nDefense" /
                    "\nSp. Attack:\nSp. Defense:\nSpeed:")
                self.layout.addWidget(self.label, 1, 0)

        def OpenWebcam(self):
                '''
                Open a webcam on the users machine if it
                exists and read in an image using that image
                it will update the GUI with the Pokemon's information

                **Parameters**

                    None
                **Returns**

                    None
                '''
                cam = cv2.VideoCapture(0)
                if not cam.isOpened():
                    # Print this error message if there is no webcam
                    messagebox.showinfo("Error", "No camers is detected")
                else:
                    while True:
                        ret, frame = cam.read()
                        # Display the resulting frame
                        cv2.imshow("frame", frame)
                        key = cv2.waitKey(1)
                        # Exit when ESC key is pressed
                        if key == 27:
                            break
                        # Take a Picture when Space Bar is pressed
                        elif key == 32:
                            cv2.imwrite(filename='saved_img.jpg', img=frame)
                            new_img, DataLabel = get_data('saved_img.jpg')
                            self.label.setText(DataLabel)

                            qImage = ImageQt(new_img)
                            pixmap = QtGui.QPixmap.fromImage(qImage)
                            self.img.setPixmap(pixmap)

                # When everything done, release the capture
                cam.release()
                cv2.destroyAllWindows()

        def UploadImage(self):
                '''
                This function will open a File Explorer
                on the users machine and allow them to open their image
                **Parameters**

                    None
                **Returns**

                    None
                '''
                root = tk.Tk()
                root.withdraw()
                # Stores the file path from the machine in a
                # string
                file_path = filedialog.askopenfilename()
                # Reads the image and gets the label and
                # confidence from that Picture
                new_img, DataLabel = get_data(file_path)
                self.label.setText(DataLabel)

                qImage = ImageQt(new_img)
                pixmap = QtGui.QPixmap.fromImage(qImage)
                self.img.setPixmap(pixmap)


if __name__ == "__main__":

        app = QApplication(sys.argv)
        dlg = Pokedex()
        dlg.show()
        sys.exit(app.exec_())
