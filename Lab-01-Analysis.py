"""
AER E 322 Lab 01
Spring 2023
Matthew Mehrtens, Peter Mikolitis, and Natsuki Oda

This Python script imports our experimental data from the .csv file and then runs a number of post-
processing algorithms on the data.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq


DATA_FILE = "Lab 01 Data.csv"


# Starting point of the script
def main():
    # Import the data
    data = import_data(DATA_FILE)
    time = data["Time (s)"]
    time_np = time.to_numpy()
    displacement_time = time[0::20]
    displacement = data["Displacement (mm)"][0::20]
    position = {
        "run1": data["Position (m)"] * 1000,
        "run2": data["Position (m).1"] * 1000,
        "run3": data["Position (m).2"] * 1000
    }
    position_smoothed = {
        "run1": smooth_df(position["run1"], [5, 3]),
        "run2": smooth_df(position["run2"], [5, 3, 3]),
        "run3": smooth_df(position["run3"], [5, 3])
    }
    lobf = {
        "run1": sin_lobf(position["run1"].to_numpy(), time_np),
        "run1-smoothed": sin_lobf(position_smoothed["run1"].to_numpy(), time_np),
        "run2": sin_lobf(position["run2"].to_numpy(), time_np),
        "run2-smoothed": sin_lobf(position_smoothed["run2"].to_numpy(), time_np),
        "run3": sin_lobf(position["run3"].to_numpy(), time_np),
        "run3-smoothed": sin_lobf(position_smoothed["run3"].to_numpy(), time_np),
    }
    lobf_piecewise = {
        "run1": sin_lobf_piecewise(position["run1"].to_numpy(), time_np, 100),
        "run1-smoothed": sin_lobf_piecewise(position_smoothed["run1"].to_numpy(), time_np, 100),
        "run2": sin_lobf_piecewise(position["run2"].to_numpy(), time_np, 100),
        "run2-smoothed": sin_lobf_piecewise(position_smoothed["run2"].to_numpy(), time_np, 100),
        "run3": sin_lobf_piecewise(position["run3"].to_numpy(), time_np, 100),
        "run3-smoothed": sin_lobf_piecewise(position_smoothed["run3"].to_numpy(), time_np, 100),
    }

    # Graph Run 1 (Raw)
    plt.figure()
    plt.plot(time, position["run1"], label="Motion Sensor") 
    plt.plot(displacement_time, displacement, label="Displacement Sensor")
    plt.plot(time, lobf["run1"], label="Non-Linear LSQ Fit Curve")
    plt.plot(time, lobf_piecewise["run1"], label="Piecewise Non-Linear LSQ Fit Curve")
    plt.title("Run 1 (Raw)")
    plt.xlabel("Time [s]")
    plt.ylabel("Displacement [mm]")
    plt.legend(loc="best")
    plt.grid()

    # Graph Run 1 (Smoothed)
    plt.figure()
    plt.plot(time, position_smoothed["run1"], label="Motion Sensor")
    plt.plot(time, lobf["run1-smoothed"], label="Non-Linear LSQ Fit Curve")
    plt.title("Run 1 (Smoothed)")
    plt.xlabel("Time [s]")
    plt.ylabel("Displacement [mm]")
    plt.legend(loc="best")
    plt.grid()

    # Graph Run 2 (Raw)
    plt.figure()
    plt.plot(time, position_smoothed["run2"], label="Motion Sensor")
    plt.plot(time, lobf["run2"], label="Non-Linear LSQ Fit Curve")
    plt.title("Run 2 (Raw)")
    plt.xlabel("Time [s]")
    plt.ylabel("Displacement [mm]")
    plt.legend(loc="best")
    plt.grid()

    # Graph Run 2 (Smoothed)
    plt.figure()
    plt.plot(time, position_smoothed["run2"], label="Motion Sensor")
    plt.plot(time, lobf["run2-smoothed"], label="Non-Linear LSQ Fit Curve")
    plt.title("Run 2 (Smoothed)")
    plt.xlabel("Time [s]")
    plt.ylabel("Displacement [mm]")
    plt.legend(loc="best")
    plt.grid()

    # Graph Run 3 (Raw)
    plt.figure()
    plt.plot(time, position["run3"], label="Motion Sensor")
    plt.plot(time, lobf["run3"], label="Non-Linear LSQ Fit Curve")
    plt.title("Run 3 (Raw)")
    plt.xlabel("Time [s]")
    plt.ylabel("Displacement [mm]")
    plt.legend(loc="best")
    plt.grid()

    # Graph Run 3 (Smoothed)
    plt.figure()
    plt.plot(time, position["run3"], label="Motion Sensor")
    plt.plot(time, lobf["run3-smoothed"], label="Non-Linear LSQ Fit Curve")
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


# Smooths a DataFrame
# Input Args:
#   df: 1 col DataFrame
#   smooth: list of smoothing windows, e.g., [5, 3, 3]
def smooth_df(df, smooth):
    smoothed_data = df
    # Smooth the bulk of the data
    for win in smooth:
        smoothed_data = smoothed_data.rolling(win, min_periods=1).mean()

    return smoothed_data


# Generate line of best fit for a sine curve
def sin_lobf(data, t):
    guess_mean = np.mean(data)
    guess_phase = 0
    guess_freq = 2*np.pi*10
    guess_amp = -1

    # Define the function to optimize, in this case, we want to minimize the difference
    # between the actual data and our "guessed" parameters
    optimize_func = lambda x: x[0] * np.sin(x[1] * (t - x[2])) + x[3] - data
    opt_amp, opt_freq, opt_phase, opt_mean = leastsq(optimize_func, [guess_amp, guess_freq, guess_phase, guess_mean])[0]

    # recreate the fitted curve using the optimized parameters
    return opt_amp * np.sin(opt_freq * (t - opt_phase)) + opt_mean


# Generates a line of best for pieces of the data
def sin_lobf_piecewise(data, t, n):
    ret = np.zeros(len(data))
    i = 0
    j = n
    while j <= len(data):
        ret[i:j] = sin_lobf(data[i:j], t[i:j])
        i += n
        j += n
    if j != len(data) and len(data) - i > 1:
        print(i)
        print(j)
        print(data)
        print(data[i:len(data)])
        print(data[i:])
        ret[i:len(data)] = sin_lobf(data[i:len(data)], t[i:len(data)])
    return ret



# Calls the main function
if __name__ == "__main__":
    main()
