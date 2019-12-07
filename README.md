## PokeDex 

Pokedex is a device that contains all the esstenial stats for each Pokemon. In the games or anime each time a character would encounter one 
of these creatures that information would be displayed to the character. This project has reconstructed a real world application Graphical
User Interface (GUI) to read in an image and display the Pokemon's information. Tensorflow is essential to this project because we will be
using ransfer learning, which means we are starting with a model that has been already trained on another problem. We will then retrain it
on a similar problem. 

## Table of Content
* [Technologies](#technologies)
* [Installing](#installing)
* [Setup](#setup)
* [Sources](#sources)
* [File List](#file-list)
* [Sample Code](#sample-code)

## Technologies
Project is created with:
* Python version: 3.7.3
* Sublime Text version: 3.2.1
Project uses:
* Git Bash

Installing

To properly train your image classifier follow these instructions:
Step 1:  Installing Tensorflow use Git Bash
$ pip install tensorflow

Step 2: Clone Git repository
git clone https://github.com/tensorflow/tensorflow.git

Step 3: Define variables:
IMAGE_SIZE=224
ARCHITECTURE="mobilenet_0.50_${IMAGE_SIZE}"

Step 4: Train the classifier
python -m scripts.retrain \
  --bottleneck_dir=tf_files/bottlenecks \
  --how_many_training_steps=500 \
  --model_dir=tf_files/models/ \
  --summaries_dir=tf_files/training_summaries/"${ARCHITECTURE}" \
  --output_graph=tf_files/retrained_graph.pb \
  --output_labels=tf_files/retrained_labels.txt \
  --architecture="${ARCHITECTURE}" \
  --image_dir=tf_files/XXX_photos

where XXX_photos is the data set containing the directory of images you want to classifly 
For instance, having two seperate folders of 100 images of Dogs and Cats

This will produce two files: 
  retrained_graph.pb 
  retrained_labels.txt 
in tf_files folder in the tensor

** Running the tests

python -m scripts.label_image \
    --graph=tf_files/retrained_graph.pb  \
    --image=tf_files/XXX.jpg
where XXX.jpg is an image you want to classify 

## Setup
To run this project, use a Text Editor such as Sumblime Text to open the files:
* label_imageGUI.py 

make sure to take the retrained retrained_graph.pb and retrained_labels.txt files
You need to use them with this python file to produce the correct results

Then use the Anaconda Prompt to run the files in Python 3.

## Sources
EN.540.635 - Software Carpentry by Henry C. Herbol
* https://www.thecomputationalist.com/Weekly_Challenges/
Syed Sadat Nazrul 
* https://github.com/snazrul1/PyRevolution
Googlecodelabs
* https://github.com/tensorflow/tensorflow

## File List
Project contains the following files in the repository:
* label_imageGUI.py - Located in the master branch
* webCrawler.py - Located in the master branch
* retrained_graph.pb and retrained_labels.txt - Trained on all 151 Pokemon
* retrained_graph2.pb and retrained_labels2.txt files - Trained on only 16 Pokemon

Citations:


Author Tyler Summers
