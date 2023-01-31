"""
AER E 322 Lab 01
Spring 2023
Matthew Mehrtens, Peter Mikolitis, and Natsuki Oda

This Python script imports our experimental data from the .csv file and then runs a number of post-
processing algorithms on the data.
"""
import pandas as pd
import matplotlib.pyplot as plt


DATA_FILE = "Lab 01 Data.csv"


# Starting point of the script
def main():
    # Import the data
    data = import_data(DATA_FILE)
    time = data["Time (s)"].to_numpy()
    displacement_time = time[0::20]
    displacement = data["Displacement (mm)"].to_numpy()[0::20]
    position = {
        "run1": data["Position (m)"].to_numpy(),
        "run2": data["Position (m).1"].to_numpy(),
        "run3": data["Position (m).2"].to_numpy()
    }


# Imports the .csv file into a panda DataFrame
def import_data(filename):
    return pd.read_csv(filename, 
                       skiprows=1,
                       usecols=[0,1,2,7,12])


# Calls the main function
if __name__ == "__main__":
    main()
