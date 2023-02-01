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
from scipy.optimize import curve_fit


DATA_FILE = "Lab 01 Data.csv"
USE_PIECEWISE_LOBF = False


# Starting point of the script
def main():
    # Import the data
    data = import_data(DATA_FILE)

    # Parse the data into variables
    time = data["Time (s)"] # DataFrame

    # Parse the displacement sensor data
    displacement_time = time[0::20]
    displacement = data["Displacement (mm)"][0::20]

    # Parse the motion sensor data from three runs
    position = {
        "run1": data["Position (m)"] * 1000,
        "run2": data["Position (m).1"] * 1000,
        "run3": data["Position (m).2"] * 1000
    }

    # Smooth the motion sensor data from the three runs
    position_smoothed = {
        "run1": smooth_df(position["run1"], [5, 3]),
        "run2": smooth_df(position["run2"], [5, 3, 3]),
        "run3": smooth_df(position["run3"], [5, 3])
    }

    # Generate the Line of Best Fit (LOBF) for both the raw data and smoothed data from the three runs
    lobf = {
        "run1": sin_lobf(position["run1"], time, True)["fitfunc"](time),
        "run1-smoothed": sin_lobf(position_smoothed["run1"], time, True)["fitfunc"](time),
        "run2": sin_lobf(position["run2"], time, True)["fitfunc"](time),
        "run2-smoothed": sin_lobf(position_smoothed["run2"], time, True)["fitfunc"](time),
        "run3": sin_lobf(position["run3"], time, True)["fitfunc"](time),
        "run3-smoothed": sin_lobf(position_smoothed["run3"], time, True)["fitfunc"](time),
    }

    # Generated LOBFs based on smaller chunks of data
    lobf_piecewise = {
        "run1": sin_lobf_piecewise(position["run1"], time, 110),
        "run1-smoothed": sin_lobf_piecewise(position_smoothed["run1"], time, 225),
        "run2": sin_lobf_piecewise(position["run2"], time, 100),
        "run2-smoothed": sin_lobf_piecewise(position_smoothed["run2"], time, 100),
        "run3": sin_lobf_piecewise(position["run3"], time, 100),
        "run3-smoothed": sin_lobf_piecewise(position_smoothed["run3"], time, 250),
    }

    # Display Graph of Run 1 (Raw)
    plt.figure()
    plt.plot(time, position["run1"], label="Motion Sensor") 
    plt.plot(displacement_time, displacement, label="Displacement Sensor")
    if USE_PIECEWISE_LOBF:
        plt.plot(time, lobf_piecewise["run1"], label="Piecewise Fit Curve")
    else:
        plt.plot(time, lobf["run1"], label="Fit Curve")
    plt.title("Run 1 (Raw)")
    plt.xlabel("Time [s]")
    plt.ylabel("Displacement [mm]")
    plt.legend(loc="best")
    plt.grid()

    # Display Graph of Run 1 (Smoothed)
    plt.figure()
    plt.plot(time, position_smoothed["run1"], label="Motion Sensor")
    if USE_PIECEWISE_LOBF:
        plt.plot(time, lobf_piecewise["run1-smoothed"], label="Piecewise Fit Curve")
    else:
        plt.plot(time, lobf["run1-smoothed"], label="Fit Curve")
    plt.title("Run 1 (Smoothed)")
    plt.xlabel("Time [s]")
    plt.ylabel("Displacement [mm]")
    plt.legend(loc="best")
    plt.grid()

    # Display Graph of Run 2 (Raw)
    plt.figure()
    plt.plot(time, position["run2"], label="Motion Sensor")
    if USE_PIECEWISE_LOBF:
        plt.plot(time, lobf_piecewise["run2"], label="Piecewise Fit Curve")
    else:
        plt.plot(time, lobf["run2"], label="Fit Curve")
    plt.title("Run 2 (Raw)")
    plt.xlabel("Time [s]")
    plt.ylabel("Displacement [mm]")
    plt.legend(loc="best")
    plt.grid()

    # Display Graph of Run 2 (Smoothed)
    plt.figure()
    plt.plot(time, position_smoothed["run2"], label="Motion Sensor")
    if USE_PIECEWISE_LOBF:
        plt.plot(time, lobf_piecewise["run2-smoothed"], label="Piecewise Fit Curve")
    else:
        plt.plot(time, lobf["run2-smoothed"], label="Fit Curve")
    plt.title("Run 2 (Smoothed)")
    plt.xlabel("Time [s]")
    plt.ylabel("Displacement [mm]")
    plt.legend(loc="best")
    plt.grid()

    # Display Graph of Run 3 (Raw)
    plt.figure()
    plt.plot(time, position["run3"], label="Motion Sensor")
    if USE_PIECEWISE_LOBF:
        plt.plot(time, lobf_piecewise["run3"], label="Piecewise Fit Curve")
    else:
        plt.plot(time, lobf["run3"], label="Fit Curve")
    plt.title("Run 3 (Raw)")
    plt.xlabel("Time [s]")
    plt.ylabel("Displacement [mm]")
    plt.legend(loc="best")
    plt.grid()

    # Display Graph of Run 3 (Smoothed)
    plt.figure()
    plt.plot(time, position_smoothed["run3"], label="Motion Sensor")
    if USE_PIECEWISE_LOBF:
        plt.plot(time, lobf_piecewise["run3-smoothed"], label="Piecewise Fit Curve")
    else:
        plt.plot(time, lobf["run3-smoothed"], label="Fit Curve")
    plt.title("Run 3 (Smoothed)")
    plt.xlabel("Time [s]")
    plt.ylabel("Displacement [mm]")
    plt.legend(loc="best")
    plt.grid()

    # Show all the graphs
    plt.show()


# Imports the .csv file into a pandas DataFrame
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
    for win in smooth:
        # min_periods=1 stops the NaNs
        smoothed_data = smoothed_data.rolling(win, min_periods=1).mean()

    return smoothed_data


# Modified from https://stackoverflow.com/questions/16716302/how-do-i-fit-a-sine-curve-to-my-data-with-pylab-and-numpy
# Uses a Fast Fourier Transform to generate an initial guess then uses a non-linear least squares algorithm to
# generate a matching sine curve
def sin_lobf(data, t, stats=False):
    # Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"
    t = np.array(t)
    data = np.array(data)
    ff = np.fft.fftfreq(len(t), (t[1]-t[0]))   # assume uniform spacing
    Fyy = abs(np.fft.fft(data))
    guess_freq = abs(ff[np.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
    guess_amp = np.std(data) * 2.**0.5
    guess_offset = np.mean(data)
    guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])

    def sinfunc(t, A, w, p, c):  return A * np.sin(w*t + p) + c
    popt, pcov = curve_fit(sinfunc, t, data, p0=guess) # this is the least squares bit
    A, w, p, c = popt
    f = w/(2.*np.pi)
    fitfunc = lambda t: A * np.sin(w*t + p) + c
    dict = {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": fitfunc, "maxcov": np.max(pcov), "rawres": (guess,popt,pcov)}
    
    # Print some stats
    if stats:
        print("Amplitude: %g mm\nFrequency: %g Hz\nLOBF: f(t) = %gsin(%g * t + %g) + %g\n----------------------------------------------------------" % (dict['amp'], dict['freq'], dict['amp'], dict['omega'], dict['phase'], dict['offset']))
    return dict


# This function used the same non-linear least squares algorithm, but it struggled
# because the initial frequency guess was so finnicky... The above function works
# much better for resolving the appropriate frequency

# Generate line of best fit for a sine curve
# def sin_lobf(data, t):
#     guess_mean = np.mean(data)
#     guess_phase = 0
#     guess_freq = 2 * np.pi * 10
#     guess_amp = -1

#     # Define the function to optimize, in this case, we want to minimize the difference
#     # between the actual data and our "guessed" parameters
#     optimize_func = lambda x: x[0] * np.sin(x[1] * (t - x[2])) + x[3] - data
#     opt_amp, opt_freq, opt_phase, opt_mean = leastsq(optimize_func, [guess_amp, guess_freq, guess_phase, guess_mean])[0]

#     # recreate the fitted curve using the optimized parameters
#     return opt_amp * np.sin(opt_freq * (t - opt_phase)) + opt_mean


# Generates a line of best for pieces of the data
def sin_lobf_piecewise(data, t, n):
    pw_lobf = np.zeros(len(data))
    i = 0
    j = n

    # Run the LOBF fit algorithm on consecutive n chunks of data
    while j <= len(data):
        f = sin_lobf(data[i:j], t[i:j])["fitfunc"]
        pw_lobf[i:j] = f(t[i:j])
        i += n
        j += n

    # After the bulk of the data is done, how much is left?
    if j != len(data) and len(data) - i > 1:
        # If more than 1 element is left, run the algorithm per usual
        f = sin_lobf(data[i:len(data)], t[i:len(data)])["fitfunc"]
        pw_lobf[i:len(data)] = f(t[i:len(data)])
    else:
        # If only 1 element is left, just assign the variable :P
        pw_lobf[len(data) - 1] = data[len(data) - 1]
    return pw_lobf


# Calls the main function
if __name__ == "__main__":
    main()
