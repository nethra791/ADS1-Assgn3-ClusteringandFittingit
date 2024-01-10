import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import sklearn.preprocessing as pp
import sklearn.metrics as skmet
from sklearn import cluster
import matplotlib.cm as cm
from scipy.optimize import curve_fit

#Function to read csv files
def read_csv_with_pandas(FileName):
    """
    Reads data from a CSV file using pandas.

    Parameters:
    - FileName (str): The path to the CSV file.

    Returns:
    - pd.DataFrame or None: A DataFrame containing the data from the 
     CSV file if the file is found, otherwise, None is returned.
    """
    try:
        df = pd.read_csv(FileName)
        return df
    except FileNotFoundError:
        print(f"Error: The file '{FileName }' was not found.")
        return None

# Function to calculate silhoutte score
def one_silhoutte(ab, n):
    """
    Calculate the silhouette score for k-means clustering.

    Parameters:
    ab (array): The dataset to be clustered, typically a 2D array.
    n (int): The number of clusters to form.

    Returns:
    float: The silhouette score
    """
    kmeans = cluster.KMeans(n_clusters=n, n_init=20)
    kmeans.fit(ab)
    labels = kmeans.labels_
    score = skmet.silhouette_score(ab, labels)
    return score


# Logistic growth model function
def logistic_growth(x, K, r, t0):
    """
    Logistic growth model function.

    Parameters:
    x (float or array): The input variable, typically time.
    K (float): Carrying capacity of the environment.
    r (float): Growth rate.
    t0 (float): The time of maximum growth rate.

    Returns:
    float or array: The value of the logistic function at each point in x.
    """
    return K / (1 + np.exp(-r * (x - t0)))


# Function to calculate numerical derivatives
def deriv(x, func, parameter, ip):
    """
    Calculate the numerical derivative of a function at a given point.

    Parameters:
    x (float or array): The point(s) at which the derivative is calculated.
    func (function): The function whose derivative is to be calculated.
    parameter (array): Parameters of the function 'func'.
    ip (int): The index of the parameter with respect to which
    the derivative is taken.

    Returns:
    float or array: The derivative of the function 'func' with respect to 
                         parameter at index 'ip' evaluated at 'x'.
    """
    scale = 1e-6
    delta = np.zeros_like(parameter, dtype=np.float64)
    val = scale * np.abs(parameter[ip])
    delta[ip] = val
    x_float = np.array(x, dtype=np.float64)
    parameter_float = np.array(parameter, dtype=np.float64)
    diff = 0.5 * (func(x_float, *(parameter_float + delta)) - func(x_float, *(parameter_float - delta)))
    return diff / val


# error propagation function
def error_prop(x, func, parameter, covar):
    """
    Calculate the propagated error for a function 
    given the covariance of its parameters.

    Parameters:
    x (float or array): The point(s) at which the error is calculated.
    func (function): The function for which the error is being calculated.
    parameter (array): Parameters of the function 'func'.
    covar (2D array): Covariance matrix of the parameters.

    Returns:
    float or array: The standard deviation (square root of variance)
    of the function 'func' at each point in 'x', due to 
    the uncertainty in its parameters.
    """
    var = np.zeros_like(x, dtype=np.float64)
    for i in range(len(parameter)):
        deriv1 = deriv(x, func, parameter, i)
        for j in range(len(parameter)):
            deriv2 = deriv(x, func, parameter, j)
            var += deriv1 * deriv2 * covar[i, j]
    return np.sqrt(var)

#Code for Clustering of data
data_Tub = read_csv_with_pandas("TubeData.csv")
print(data_Tub.describe())
# use 2000 and 2020 for clustering. Countries with one NaN are removed
data_Tub = data_Tub[(data_Tub["2000"].notna()) & (data_Tub["2022"].notna())]
data_Tub = data_Tub.reset_index(drop=True)
# extract 2000 data
growth = data_Tub[["Country Name", "2000"]].copy()
# calculate the growth over 22 years
growth["Growth"] = 100.0/22.0 * (data_Tub["2022"]-data_Tub["2000"]) / data_Tub["2000"]
warnings.filterwarnings("ignore", category=UserWarning)
print(growth.describe())
print()
print(growth.dtypes)

plt.figure(figsize=(8, 8))
plt.scatter(growth["2000"], growth["Growth"])
plt.xlabel("Incidence of tuberculosis (per 100,000 people),2000")
plt.ylabel("decline/incline per year [%]")
plt.show()

# create a scaler object
scaler = pp.RobustScaler()
# set up the scaler
# extract the columns for clustering
df_ex = growth[["2000", "Growth"]]
scaler.fit(df_ex)
# apply the scaling
norm = scaler.transform(df_ex)
plt.figure(figsize=(8, 8))
plt.scatter(norm[:, 0], norm[:, 1])
plt.xlabel("Incidence of tuberculosis (per 100,000 people),2000")
plt.ylabel("decline/incline per year [%]")
plt.show()

#calculate silhouette score for 2 to 10 clusters
for ic in range(2, 11):
    score = one_silhoutte(norm, ic)
    print(f"The silhouette score for {ic: 3d} is {score: 7.4f}")

# set up the clusterer with the number of expected clusters
kmeans = cluster.KMeans(n_clusters=3, n_init=20)
# Fit the data, results are stored in the kmeans object
kmeans.fit(norm) # fit done on x,y pairs
# extract cluster labels
labels = kmeans.labels_
# extract the estimated cluster centres and convert to original scales
cen = kmeans.cluster_centers_
cen = scaler.inverse_transform(cen)
xkmeans = cen[:, 0]
ykmeans = cen[:, 1]
plt.figure(figsize=(8.0, 8.0))
# plot data with kmeans cluster number
plt.scatter(growth["2000"], growth["Growth"], 10, labels, marker="o", cmap=cm.rainbow)
# show cluster centres
plt.scatter(xkmeans, ykmeans, 45, "k", marker="d")
plt.xlabel("Incidence of tuberculosis (per 100,000 people),2000")
plt.ylabel("decline/incline per year [%]")
plt.show()
print(cen)

#getting subset of the normalized data
growth2 = growth[labels==0].copy()
print(growth2.describe())
df_ex = growth2[["2000", "Growth"]]
scaler.fit(df_ex)

# apply the scaling
norm = scaler.transform(df_ex)
plt.figure(figsize=(8, 8))
plt.scatter(norm[:, 0], norm[:, 1])
plt.xlabel("Incidence of tuberculosis (per 100,000 people),2000")
plt.ylabel("decline/incline per year [%]")
plt.show()

# set up the clusterer with the number of expected clusters
kmeans = cluster.KMeans(n_clusters=3, n_init=20)
# Fit the data, results are stored in the kmeans object
kmeans.fit(norm) # fit done on x,y pairs
# extract cluster labels
labels = kmeans.labels_
# extract the estimated cluster centres and convert to original scales
cen = kmeans.cluster_centers_
cen = scaler.inverse_transform(cen)
xkmeans = cen[:, 0]
ykmeans = cen[:, 1]
plt.figure(figsize=(8.0, 8.0))
# plot data with kmeans cluster number
plt.scatter(growth2["2000"], growth2["Growth"], 10, labels, marker="o", cmap=cm.rainbow)
# show cluster centres
plt.scatter(xkmeans, ykmeans, 45, "k", marker="d")
plt.xlabel("Incidence of tuberculosis (per 100,000 people),2000")
plt.ylabel("decline/incline per year [%]")
plt.show()



#code for fitting the data
# Load data
filename = 'NamData.csv'  
data = read_csv_with_pandas(filename)

# Transpose the data for easier processing
data_transposed = data.transpose()

# Clean the transposed data
data_transposed.columns = data_transposed.iloc[0]
data_transposed = data_transposed.drop(data_transposed.index[0])

# Extracting years and values from the transposed data
years = np.array(data_transposed.index, dtype=np.float64)
values = data_transposed.iloc[:, 0].values.astype(np.float64)

# Fit the logistic growth model
initial_guess = [max(values), 0.01, np.mean(years)]
lpar, logistic_covariance = curve_fit(logistic_growth, years, values, p0=initial_guess, maxfev=10000)

# Make predictions for future years
future_years = np.arange(2023, 2034, dtype=np.float64)
future_predictions = logistic_growth(future_years, *lpar)

# Calculate confidence intervals
confidence_intervals_historical = error_prop(years, logistic_growth, lpar, logistic_covariance)
confidence_intervals = error_prop(future_years, logistic_growth, lpar, logistic_covariance)

# Plotting
plt.figure(figsize=(12, 8))
plt.plot(years, values, 'g-', label='Original Data')
plt.plot(years, logistic_growth(years, *lpar), 'r-', label='Logistic Growth Fit (Historical)')
plt.plot(future_years, future_predictions, 'r--', label='Logistic Growth Fit (Future)')
plt.fill_between(years, logistic_growth(years, *lpar) - confidence_intervals_historical, logistic_growth(years, *lpar) + confidence_intervals_historical, color='red', alpha=0.2, label='Confidence Interval (Historical)')
plt.fill_between(future_years, future_predictions - confidence_intervals, future_predictions + confidence_intervals, color='gray', alpha=0.2, label='Confidence Interval')
plt.xlabel('Year')
plt.ylabel("Incidence of tuberculosis (per 100,000 people)")
plt.title('Logistic Growth Model Fit and Future Prediction')
plt.legend()
plt.show()
