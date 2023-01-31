"""
AER E 322 Lab 01
Spring 2023
Matthew Mehrtens, Peter Mikolitis, and Natsuki Oda

This Python script imports our experimental data from the .csv file and then runs a number of post-
processing algorithms on the data.
"""
import pandas as pd


DATA_FILE = "Lab 01 Data.csv"


# Starting point of the script
def main():
    # Import the data
    data = import_data(DATA_FILE)


# Imports the .csv file into a panda DataFrame
def import_data(filename):
    return pd.read_csv(filename, 
                       skiprows=2)


# Calls the main function
if __name__ == "__main__":
    main()
