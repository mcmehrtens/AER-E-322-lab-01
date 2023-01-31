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
    time = data["Time (s)"]
    displacement_time = time[0::20]
    displacement = data["Displacement (mm)"][0::20]
    position = {
        "run1": data["Position (m)"] * 1000,
        "run2": data["Position (m).1"] * 1000,
        "run3": data["Position (m).2"] * 1000
    }

    # Graph Run 1 (Raw)
    plt.figure()
    plt.plot(time, position["run1"], label="Motion Sensor") 
    plt.plot(displacement_time, displacement, label="Displacement Sensor")
    plt.title("Run 1 (Raw)")
    plt.xlabel("Time [s]")
    plt.ylabel("Displacement [mm]")
    plt.legend(loc="best")
    plt.grid()

    # Graph Run 1 (Smoothed)
    position["run1"] = position["run1"].rolling(window=5, center=True, closed="both").mean()

    plt.figure()
    plt.plot(time, position["run1"], label="Motion Sensor")
    plt.title("Run 1 (Smoothed)")
    plt.xlabel("Time [s]")
    plt.ylabel("Displacement [mm]")
    plt.legend(loc="best")
    plt.grid()

    # Graph Run 2 (Raw)
    plt.figure()
    plt.plot(time, position["run2"], label="Motion Sensor")
    plt.title("Run 2 (Raw)")
    plt.xlabel("Time [s]")
    plt.ylabel("Displacement [mm]")
    plt.legend(loc="best")
    plt.grid()

    # Graph Run 2 (Smoothed)
    plt.figure()
    plt.plot(time, position["run1"].rolling(window=5, center=True, closed="both").mean(), label="Motion Sensor")
    plt.title("Run 2 (Smoothed)")
    plt.xlabel("Time [s]")
    plt.ylabel("Displacement [mm]")
    plt.legend(loc="best")
    plt.grid()

    # Graph Run 3 (Raw)
    plt.figure()
    plt.plot(time, position["run3"], label="Motion Sensor")
    plt.title("Run 3 (Raw)")
    plt.xlabel("Time [s]")
    plt.ylabel("Displacement [mm]")
    plt.legend(loc="best")
    plt.grid()

    # Graph Run 3 (Smoothed)
    plt.figure()
    plt.plot(time, position["run1"].rolling(window=5, center=True, closed="both").mean(), label="Motion Sensor")
    plt.title("Run 3 (Smoothed)")
    plt.xlabel("Time [s]")
    plt.ylabel("Displacement [mm]")
    plt.legend(loc="best")
    plt.grid()

    plt.show()


# Imports the .csv file into a panda DataFrame
def import_data(filename):
    return pd.read_csv(filename, 
                       skiprows=1,
                       usecols=[0,1,2,7,12])


# Calls the main function
if __name__ == "__main__":
    main()
