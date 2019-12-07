from icrawler.builtin import GoogleImageCrawler
import os
import pickle
import pandas as pd
import csv

DATADIR = "C:/Users/name/Desktop/Pokedex/dataset"
DATA = "C:/Users/name/Desktop/Pokedex/newDataSet/"


def read_file(filename):
    '''
    This function will read in the pickle file and assess
    the data types within the pickle file. Then, parse out
    the list into three individual lists for the various data types.

    **Parameters**

        filename: *str*
            The pickle file to be read in.

    **Returns**

        List: *list, list*
            A list of list, each list holds the pokemon names and
            other information
        Dict: *list, dictionary*
            A list of dictionaries, each dictionary
            holds the element's information including
            weight and vdw_r
        Tuple: *list, tuple*
            A list of tuples, each tuple holds the book
            title, book ranking and
            author name.
    '''
    # Empty lists to hold the information
    example_dict = []
    Dict = []
    List = []
    Tuple = []
    # Reading in the pickle file and storing the informaion
    # in its own list
    pickle_in = open(filename, "rb")
    example_dict = pickle.load(pickle_in)
    # Using a For Loop to iterate thourgh the list
    # from the pickle file
    for i in range(0, len(example_dict)):
        # Depending on whether it is a list,
        # dicationary or tuple it will be stored
        # in its respective list
        if isinstance(example_dict[i], list):
            List.append(example_dict[i])
        elif isinstance(example_dict[i], dict):
            Dict.append(example_dict[i])
        elif isinstance(example_dict[i], tuple):
            Tuple.append(example_dict[i])
    # Returning each of the three lists
    return List, Dict, Tuple


if __name__ == "__main__":
    # Empty lists for the list of list, dictionaries,
    # and tuples
    dictList = []
    PokemonList = []
    tupleList = []
    # Storing that information from the pickle file in those lists
    PokemonList, dictList, tupleList = read_file("Lab_7_py3_data.pickle")
    # A check to see if there are nine items in each list
    # corresponding to:
    # Total, Health Points, Attack, Defense,
    # Special Attack, Special Defense, Speed
    # if not that list will be removed from the list of lists
    for i in range(0, len(PokemonList) - 1):
        if(len(PokemonList[i]) != 9):
            PokemonList.remove(PokemonList[i])
        os.mkdir(os.path.join(DATA, PokemonList[i][0]))
        # Searcing Google for picture of Pokemon
        # to build a data set
        google_crawler = GoogleImageCrawler(storage={
            'root_dir':
            r"C:/Users/name/Desktop/tensorflow-for-poets-2/" /
            "tf_files/newDataSet/" + PokemonList[i][0]})
        google_crawler.crawl(
            keyword=PokemonList[i][0] + " Pokemon", max_num=100)

    # Storing the Pokemon Information in a CSV file
    myData = pd.DataFrame(PokemonList)
    myData.to_csv('PokemonData.csv', index=False, header=False)
    with open("PokemonData.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(PokemonList)
