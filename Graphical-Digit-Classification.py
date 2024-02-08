# Author: Lior Ben Moha, 207214503 , Hadar Miller, 207865262

# Standard scientific Python imports
import matplotlib.pyplot as plt
import numpy as np

# Import datasets, classifiers and performance metrics
from sklearn import datasets, metrics, svm, preprocessing, linear_model
from sklearn.model_selection import train_test_split, cross_val_predict
from scipy.stats import skew




################
# Bonus Question
def std_dev_matrix(matrix):
    """
    Calculate the standard deviation of the elements in a matrix.

    Parameters:
    - matrix (numpy.ndarray): The input matrix.

    Returns:
    - float: The standard deviation of the elements in the matrix.
    """
    flat_matrix = np.array(matrix).flatten()
    return np.std(flat_matrix)


def median_matrix(matrix):
    """
    Calculate the median value of the elements in a matrix.

    Parameters:
    - matrix (numpy.ndarray): The input matrix.

    Returns:
    - float: The median value of the elements in the matrix.
    """
    flat_matrix = np.array(matrix).flatten()
    return np.median(flat_matrix)


def entropy_matrix(matrix):
    """
    Calculate the entropy of the elements in a matrix.

    Parameters:
    - matrix (numpy.ndarray): The input matrix.

    Returns:
    - float: The entropy of the elements in the matrix.
    """
    flat_matrix = np.array(matrix).flatten()
    _, counts = np.unique(flat_matrix, return_counts=True)
    probabilities = counts / len(flat_matrix)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy


def skewness_matrix(matrix):
    """
    Calculate the skewness of the elements in a matrix.

    Parameters:
    - matrix (numpy.ndarray): The input matrix.

    Returns:
    - float: The skewness of the elements in the matrix.
    """
    flat_matrix = np.array(matrix).flatten()
    return skew(flat_matrix)

# Bonus Question
################

def sum_of_digit_matrix(digit_matrix):
    """
    Calculate the sum of all values in the given digit matrix.

    Parameters:
    - digit_matrix (numpy.ndarray): The input digit matrix.

    Returns:
    - int: The sum of all values in the matrix.
    """
    return digit_matrix.sum()


def variance_of_sum_rows(digit_matrix):
    """
    Calculate the variance of the sum of rows in the given digit matrix.

    Parameters:
    - digit_matrix (numpy.ndarray): The input digit matrix.

    Returns:
    - float: The variance of the sum of rows in the matrix.
    """
    sum_of_rows = digit_matrix.sum(axis=1)  # Calculate the sum of each row
    return np.var(sum_of_rows)  # Calculate the variance of the sum of rows


def variance_of_sum_columns(digit_matrix):
    """
    Calculate the variance of the sum of columns in the given digit matrix.

    Parameters:
    - digit_matrix (numpy.ndarray): The input digit matrix.

    Returns:
    - float: The variance of the sum of columns in the matrix.
    """
    sum_of_columns = digit_matrix.sum(axis=0)  # Calculate the sum of each column
    return np.var(sum_of_columns)  # Calculate the variance of the sum of columns


def sum_middle_region(matrix):
    """
    Calculate the sum of the middle region of the given matrix.

    Parameters:
    - matrix (numpy.ndarray): The input matrix.

    Returns:
    - int: The sum of the middle region values.
    """
    rows, cols = matrix.shape

    # Define the boundaries of the middle region
    start_row = rows // 4
    end_row = 3 * rows // 4
    start_col = cols // 4
    end_col = 3 * cols // 4

    # Extract the middle region and calculate the sum
    middle_region = matrix[start_row:end_row, start_col:end_col]
    return middle_region.sum()

def avg_sum_of_rows(matrix):
    """
    Calculate the average of the sum of all rows in the given matrix.

    Parameters:
    - matrix (numpy.ndarray): The input matrix.

    Returns:
    - float: The average of the sum of all rows.
    """
    row_sums = matrix.sum(axis=1)  # Calculate the sum of each row
    return np.mean(row_sums)  # Calculate the average of the row sums



def plot_histogram(data, target, title):
    """
    Plots a histogram for the given data.

    Parameters:
        data (numpy.ndarray): The data to be plotted.
        target (numpy.ndarray): The target labels for the data.
        title (str): The title of the plot.

    Returns:
        None
    """
    plt.figure(figsize=(8, 6))
    plt.hist([data[target == 0], data[target == 1]], alpha=0.7, rwidth=0.8, label=['0', '1'])
    plt.title(title)
    plt.legend()
    plt.show()


def plot_scatter(x, y, target, label, title, x_label, y_label):
    """
    Plots a scatter plot for the given data.

    Parameters:
        x (numpy.ndarray): The x-axis data.
        y (numpy.ndarray): The y-axis data.
        target (numpy.ndarray): The target labels for the data.
        label (str): The label for the scatter plot.
        title (str): The title of the plot.
        x_label (str): The label for the x-axis.
        y_label (str): The label for the y-axis.

    Returns:
        None
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(x[target == 0], y[target == 0], label='0', alpha=0.7)
    plt.scatter(x[target == 1], y[target == 1], label='1', alpha=0.7)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()


def plot_scatter3d(x, y, z, target, title, x_label, y_label, z_label):
    """
    Plots a 3D scatter plot for the given data.

    Parameters:
        x (numpy.ndarray): The x-axis data.
        y (numpy.ndarray): The y-axis data.
        z (numpy.ndarray): The z-axis data.
        target (numpy.ndarray): The target labels for the data.
        title (str): The title of the plot.
        x_label (str): The label for the x-axis.
        y_label (str): The label for the y-axis.
        z_label (str): The label for the z-axis.

    Returns:
        None
    """
    fig = plt.figure(figsize=(8, 6))
    fig.suptitle(title, fontsize=14)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x[target == 0], y[target == 0], z[target == 0], label='0', alpha=0.7)
    ax.scatter(x[target == 1], y[target == 1], z[target == 1], label='1', alpha=0.7)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    ax.legend()
    plt.show()

###############################################################################
digits = datasets.load_digits()

# Get the images (matrices) from the digits dataset
digit_matrices = digits.images

# Create an array to hold the sum of values for each matrix
sum_values_array = np.array([sum_of_digit_matrix(digit_matrix) for digit_matrix in digit_matrices])

# Create an array to hold the variance of sum values for each matrix
variance_of_sum_rows_array = np.array([variance_of_sum_rows(digit_matrix) for digit_matrix in digit_matrices])

# Create an array to hold the variance of sum values for each matrix
variance_of_sum_columns_array = np.array([variance_of_sum_columns(digit_matrix) for digit_matrix in digit_matrices])

# Create an array to hold the sum of the middle region values for each matrix the inner 4x4 matrix
sum_middle_region_array = np.array([sum_middle_region(digit_matrix) for digit_matrix in digit_matrices])

# Create an array to hold the average of sum of rows for each matrix
avg_sum_of_rows_array = np.array([avg_sum_of_rows(digit_matrix) for digit_matrix in digit_matrices])

indices_0_1 = np.where(np.logical_and(digits.target >=0 , digits.target <= 1))

std_dev_values_array = np.array([std_dev_matrix(digit_matrix) for digit_matrix in digit_matrices])
median_values_array = np.array([median_matrix(digit_matrix) for digit_matrix in digit_matrices])
entropy_values_array = np.array([entropy_matrix(digit_matrix) for digit_matrix in digit_matrices])
skewness_values_array = np.array([skewness_matrix(digit_matrix) for digit_matrix in digit_matrices])

# Filter vectors based on indices_0_1
sum_values_0_1 = sum_values_array[indices_0_1]
variance_of_sum_rows_0_1 = variance_of_sum_rows_array[indices_0_1]
variance_of_sum_columns_0_1 = variance_of_sum_columns_array[indices_0_1]
sum_middle_region_array_0_1 = sum_middle_region_array[indices_0_1]
avg_sum_of_rows_array_0_1 = avg_sum_of_rows_array[indices_0_1]

plot_histogram(sum_values_0_1, digits.target[indices_0_1], 'Histogram of Sum Values for Digits 0 and 1')
plot_histogram(variance_of_sum_rows_0_1, digits.target[indices_0_1], 'Histogram of Variance of Sum Rows for Digits 0 and 1')
plot_histogram(variance_of_sum_columns_0_1, digits.target[indices_0_1], 'Histogram of Variance of Sum Columns for Digits 0 and 1')
plot_histogram(sum_middle_region_array_0_1, digits.target[indices_0_1], 'Histogram of The Sum Of Middle 4x4 Inner Matrix for Digits 0 and 1')
plot_histogram(avg_sum_of_rows_array_0_1, digits.target[indices_0_1], 'Histogram of The Avarage Sum Of Rows for Digits 0 and 1')

plot_histogram(std_dev_values_array[indices_0_1], digits.target[indices_0_1], "deviation")
plot_histogram(median_values_array[indices_0_1], digits.target[indices_0_1], "median")
plot_histogram(entropy_values_array[indices_0_1], digits.target[indices_0_1], "entropy")
plot_histogram(skewness_values_array[indices_0_1], digits.target[indices_0_1], "skewness")

#########################################################################

# Scatter Plot: Sum Values vs. Variance of Sum Rows for Digits 0 and 1
plot_scatter(sum_values_0_1, variance_of_sum_rows_0_1, digits.target[indices_0_1],
             'Sum Values vs. Variance of Sum Rows', 'Sum Values', 'Variance of Sum Rows', 'Count')

# Scatter Plot: Sum Values vs. Variance of Sum Columns for Digits 0 and 1
plot_scatter(sum_values_0_1, variance_of_sum_columns_0_1, digits.target[indices_0_1],
             'Sum Values vs. Variance of Sum Columns', 'Sum Values', 'Variance of Sum Columns', 'Count')

# Scatter Plot: Sum Values vs. Sum Middle Region Values for Digits 0 and 1
plot_scatter(sum_values_0_1, sum_middle_region_array_0_1, digits.target[indices_0_1],
             'Sum Values vs. Sum Middle Region Values', 'Sum Values', 'Sum Middle Region Values', 'Count')

# Scatter Plot: Sum Values vs. Average Sum of Rows for Digits 0 and 1
plot_scatter(sum_values_0_1, avg_sum_of_rows_array_0_1, digits.target[indices_0_1],
             'Sum Values vs. Average Sum of Rows', 'Sum Values', 'Average Sum of Rows', 'Count')

# Scatter Plot: Variance of Sum Rows vs. Variance of Sum Columns for Digits 0 and 1
plot_scatter(variance_of_sum_rows_0_1, variance_of_sum_columns_0_1, digits.target[indices_0_1],
             'Variance of Sum Rows vs. Variance of Sum Columns', 'Variance of Sum Rows', 'Variance of Sum Columns', 'Count')

# Scatter Plot: Variance of Sum Rows vs. Sum Middle Region Values for Digits 0 and 1
plot_scatter(variance_of_sum_rows_0_1, sum_middle_region_array_0_1, digits.target[indices_0_1],
             'Variance of Sum Rows vs. Sum Middle Region Values', 'Variance of Sum Rows', 'Sum Middle Region Values', 'Count')

# Scatter Plot: Variance of Sum Rows vs. Average Sum of Rows for Digits 0 and 1
plot_scatter(variance_of_sum_rows_0_1, avg_sum_of_rows_array_0_1, digits.target[indices_0_1],
             'Variance of Sum Rows vs. Average Sum of Rows', 'Variance of Sum Rows', 'Average Sum of Rows', 'Count')

# Scatter Plot: Variance of Sum Columns vs. Sum Middle Region Values for Digits 0 and 1
plot_scatter(variance_of_sum_columns_0_1, sum_middle_region_array_0_1, digits.target[indices_0_1],
             'Variance of Sum Columns vs. Sum Middle Region Values', 'Variance of Sum Columns', 'Sum Middle Region Values', 'Count')

# Scatter Plot: Variance of Sum Columns vs. Average Sum of Rows for Digits 0 and 1
plot_scatter(variance_of_sum_columns_0_1, avg_sum_of_rows_array_0_1, digits.target[indices_0_1],
             'Variance of Sum Columns vs. Average Sum of Rows', 'Variance of Sum Columns', 'Average Sum of Rows', 'Count')

# Scatter Plot: Sum Middle Region Values vs. Average Sum of Rows for Digits 0 and 1
plot_scatter(sum_middle_region_array_0_1, avg_sum_of_rows_array_0_1, digits.target[indices_0_1],
             'Sum Middle Region Values vs. Average Sum of Rows', 'Sum Middle Region Values', 'Average Sum of Rows', 'Count')

#########################################################################

# 1
plot_scatter3d(sum_values_0_1, variance_of_sum_rows_0_1, variance_of_sum_columns_0_1, digits.target[indices_0_1],
               'Sum Values & Variance of Sum Rows & Variance of Sum Columns',
               'Sum Values', 'Variance of Sum Rows', 'Variance of Sum Columns')

# 2
plot_scatter3d(sum_values_0_1, variance_of_sum_rows_0_1, sum_middle_region_array_0_1, digits.target[indices_0_1],
               'Sum Values & Variance of Sum Rows & Sum Middle Region Array',
               'Sum Values', 'Variance of Sum Rows', 'Sum Middle Region Array')

# 3
plot_scatter3d(sum_values_0_1, variance_of_sum_rows_0_1, avg_sum_of_rows_array_0_1, digits.target[indices_0_1],
               'Sum Values & Variance of Sum Rows & Avg Sum of Rows Array',
               'Sum Values', 'Variance of Sum Rows', 'Avg Sum of Rows Array')

# 4
plot_scatter3d(sum_values_0_1, variance_of_sum_columns_0_1, sum_middle_region_array_0_1, digits.target[indices_0_1],
               'Sum Values & Variance of Sum Columns & Sum Middle Region Array',
               'Sum Values', 'Variance of Sum Columns', 'Sum Middle Region Array')

# 5
plot_scatter3d(sum_values_0_1, variance_of_sum_columns_0_1, avg_sum_of_rows_array_0_1, digits.target[indices_0_1],
               'Sum Values & Variance of Sum Columns & Avg Sum of Rows Array',
               'Sum Values', 'Variance of Sum Columns', 'Avg Sum of Rows Array')

# 6
plot_scatter3d(sum_values_0_1, sum_middle_region_array_0_1, avg_sum_of_rows_array_0_1, digits.target[indices_0_1],
               'Sum Values & Sum Middle Region Array & Avg Sum of Rows Array',
               'Sum Values', 'Sum Middle Region Array', 'Avg Sum of Rows Array')

# 7
plot_scatter3d(variance_of_sum_rows_0_1, variance_of_sum_columns_0_1, sum_middle_region_array_0_1, digits.target[indices_0_1],
               'Variance of Sum Rows & Variance of Sum Columns & Sum Middle Region Array',
               'Variance of Sum Rows', 'Variance of Sum Columns', 'Sum Middle Region Array')

# 8
plot_scatter3d(variance_of_sum_rows_0_1, variance_of_sum_columns_0_1, avg_sum_of_rows_array_0_1, digits.target[indices_0_1],
               'Variance of Sum Rows & Variance of Sum Columns & Avg Sum of Rows Array',
               'Variance of Sum Rows', 'Variance of Sum Columns', 'Avg Sum of Rows Array')

# 9
plot_scatter3d(variance_of_sum_rows_0_1, sum_middle_region_array_0_1, avg_sum_of_rows_array_0_1, digits.target[indices_0_1],
               'Variance of Sum Rows & Sum Middle Region Array & Avg Sum of Rows Array',
               'Variance of Sum Rows', 'Sum Middle Region Array', 'Avg Sum of Rows Array')

# 10
plot_scatter3d(variance_of_sum_columns_0_1, sum_middle_region_array_0_1, avg_sum_of_rows_array_0_1, digits.target[indices_0_1],
               'Variance of Sum Columns & Sum Middle Region Array & Avg Sum of Rows Array',
               'Variance of Sum Columns', 'Sum Middle Region Array', 'Avg Sum of Rows Array')

###############################################################################

# creating the X (feature) matrix
X = np.column_stack((variance_of_sum_rows_0_1, 
                     variance_of_sum_columns_0_1, avg_sum_of_rows_array_0_1,
                     sum_middle_region_array_0_1))


# scaling the values for better classification performance
X_scaled = preprocessing.scale(X)
 

# the predicted outputs
Y = digits.target[indices_0_1]


# Training Logistic regression
logistic_classifier = linear_model.LogisticRegression(solver='lbfgs')
logistic_classifier.fit(X_scaled, Y)


# show how good is the classifier on the training data
expected = Y
predicted = logistic_classifier.predict(X_scaled)

print("Logistic regression using [variance of sum rows, variance of sum cols, avg sum of rows, sum middle region] features:\n%s\n" % (
metrics.classification_report(expected, predicted)))

print("Confusion matrix:\n%s"
      % metrics.confusion_matrix(expected, predicted))

# estimate the generalization performance using cross validation
predicted2 = cross_val_predict(logistic_classifier, X_scaled, Y, cv=10)

print("Logistic regression using [variance of sum rows, variance of sum cols, avg sum of rows, sum middle region] features cross validation:\n%s\n" 
      % (metrics.classification_report(expected, predicted2)))
print("Confusion matrix:\n%s" 
      % metrics.confusion_matrix(expected, predicted2))

# Classification
# flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Create a classifier: a support vector classifier
clf = svm.SVC(gamma=0.001)

# Split data into 50% train and 50% test subsets
X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.5, shuffle=False
)

# Learn the digits on the train subset
clf.fit(X_train, y_train)

# Predict the value of the digit on the test subset
predicted = clf.predict(X_test)


# Bonus Question
###############################################################################

# creating the X (feature) matrix
X = np.column_stack((std_dev_values_array, avg_sum_of_rows_array, sum_middle_region_array, 
                     entropy_values_array, sum_values_array, skewness_values_array, 
                     variance_of_sum_rows_array, variance_of_sum_columns_array, median_values_array))


# scaling the values for better classification performance
X_scaled = preprocessing.scale(X)
 

# the predicted outputs
Y = digits.target


# Training Logistic regression
logistic_classifier = linear_model.LogisticRegression(solver='lbfgs', max_iter=1000)
logistic_classifier.fit(X_scaled, Y)


# show how good is the classifier on the training data
expected = Y
predicted = logistic_classifier.predict(X_scaled)

print("Logistic regression using [variance of sum rows, variance of sum cols, avg sum of rows, sum middle region] features:\n%s\n" % (
metrics.classification_report(expected, predicted)))

print("Confusion matrix:\n%s"
      % metrics.confusion_matrix(expected, predicted))

# estimate the generalization performance using cross validation
predicted2 = cross_val_predict(logistic_classifier, X_scaled, Y, cv=10)

print("Logistic regression using [variance of sum rows, variance of sum cols, avg sum of rows, sum middle region] features cross validation:\n%s\n" 
      % (metrics.classification_report(expected, predicted2)))
print("Confusion matrix:\n%s" 
      % metrics.confusion_matrix(expected, predicted2))

# Classification
# flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Create a classifier: a support vector classifier
clf = svm.SVC(gamma=0.001)

# Split data into 50% train and 50% test subsets
X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.5, shuffle=False
)

# Learn the digits on the train subset
clf.fit(X_train, y_train)

# Predict the value of the digit on the test subset
predicted = clf.predict(X_test)

