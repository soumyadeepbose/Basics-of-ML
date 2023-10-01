# Getting started with ML

Welcome to this document! Let me guess... You are here because you want to get started with **Machine Learning**, but don't really know where to begin. Like I always say, recapping a bit of the **Math** needed to go ahead with ML is a good idea before actually delving deep into the algorithms. 
I have created this extensive guide document for you to get started with ML, specifically the Python environment needed for ML, some basic numpy, matplotlib and pandas operations and also some of the Math involved in Linear Regression. I have tried to cover everything that I know and things that I feel you need to know to get the gears moving. I hope this helps you out.

First off, let's start with installing and setting-up Jupyter Notebook on your local system, for future use. Then, we'll move on to the Math part.

## 1. Installing Jupyter Notebook

Before going to the installation part, let's talk a bit about what is Jupyter Notebook. Also, what is the problem with other IDEs that we felt the need to use Jupyter Notebook?

### What is Jupyter Notebook and why do we need it?

**Jupyter Notebook** is an open-source **web application** that allows you to create and share documents that contain **live code**, equations, visualizations and narrative text. It is one of the most popular IDEs for Data Science and Machine Learning. In Jupyter Notebook, you have blocks of code that we call *cells*, that can be run without running the whole program, i.e., the notebook. This is very useful when you are working with large datasets and you want to see the output of some particular blocks of code. Besides adding comments like in a typical Python program, you can also add Markdown texts, to better explain what's going on in your notebook using links and images and texts, of course. 

But why Jupyter? Why not PyCharm or VSCode? Well, in Jupyter we use the notebook style of programming which is very helpful when you want to run particular cells. In other scripting platforms, you'll have to type in the program, save it, then run it as a whole. As you'll see later in this document, running the whole program each time to get outputs isn't very practical in the field of ML. So, we use Jupyter Notebook.

### Much talk, now let's install it.

Open your cmd/terminal and type in the following command:

- ```pip install notebook``` : This will install Jupyter Notebook on your system, via pip, the official package manager for Python.
- ```jupyter notebook``` : This will open up Jupyter Notebook, in a new tab in your browser. Here, you can create new notebooks and run them, like you'll see later.

See! It's that simple to install and setup Jupyter Notebook. Now, let's get started with the basics of the Math required, after which we'll see how we can implement the learned Math using various Python libraries like numpy, matplotlib and statistics.

## 2. Basic Statistics for ML using matplotlib

Below are some formulas that you have already learned in high school, but have most likely forgotten. So, let's recap them, shall we?

- **Mean (Average):**
    - ![](https://quicklatex.com/cache3/87/ql_f54d4766c31581b521ea06a95962c087_l3.png)

- **Median:**
  - For an *Odd*-Numbered Dataset:
    - ![](https://quicklatex.com/cache3/45/ql_67150bcf1e257b373de739e3f0ada445_l3.png)
  - For an *Even*-Numbered Dataset:
    - ![](https://quicklatex.com/cache3/09/ql_ac08c343f18bb3aa8ba76bb66d725e09_l3.png)

- **Standard Deviation:**
    - ![](https://quicklatex.com/cache3/f2/ql_ac1615b0b3fa3ae84a93af91b91233f2_l3.png)

- **z-Score (Standard Score):**
  - ![](https://quicklatex.com/cache3/7f/ql_de163e719a835cb8eb4fad023a68cb7f_l3.png)

- **Variance:**
    - ![](https://quicklatex.com/cache3/e7/ql_400a0897c3700f268b6256e7b25e9fe7_l3.png)

Given below is a code snippet to try out the further below mentioned plots using the very famous matplotlib library of Python, that you can install using ```pip install matplotlib``` from your *cmd*.

```python
import numpy as np
import matplotlib.pyplot as plt

# Let's create a list of numbers for x and y axes. 
x = [1, 2, 3, 4, 5]
y = [2, 5, 6, 0, 9, 4]

# Let's also get data for a random normal distribution here.
# The following data has mean = 0 and standard deviation = 1.
normal = np.random.normal(0, 1, 100)

# Insert the code for plotting a graph here
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('Plot Example')
plt.show()
```

| Type of Plot | Example Code (goes in the comment line in the above snippet) | Why are they used? |
|----------------|--------------| ---------------------|
|Continuous Line Plot| ```plt.plot(x, y)``` | Used to get a simple plot of any data, and perhaps compare the output of a function or dataset with that of some other. Also used to clearly visualize the relationships between different data. |
|Discrete Plot| ```plt.stem(x, y)``` | Used to plot data that is not continuous in nature, i.e., discrete. For example, the number of students in a class. |
|Scatter Plot| ```plt.scatter(x, y)``` | Used to visualize the relationship between two variables, containing data, by displaying individual data points as dots on a plane. |
|Histogram| ```plt.hist(normal, bins=20)``` | Used to better understand the underlying distribution of a data. This divides the data into bins and plots the frequency of each bin. |
|Boxplot| ```plt.boxplot(normal, vert=False)``` | Used to better visualize the medians, quartiles and outliers in given data. |
|Violinplot| ```plt.violinplot(normal, showmedians=True, vert=False)``` | 

More types of plots and their examples can be found [here](https://matplotlib.org/stable/plot_types/index.html).

## 3. Basic Matrices for ML using numpy

Now that you are aware of the various basic types of graphs, let's dive into the world of matrices and linear algebra. We start off with some basic **numpy** functions that you need to know. But before, you will need to install numpy on your system using ```pip install numpy```, if you haven't installed it already.

With numpy installed, let's import numpy into the notebook, using the following line:

```python
import numpy as np
```

| numpy function | What does it do? |
|----------------|--------------|
|```np.array(list)```| You give a python list as a parameter and this function gives you a numpy array, that can be used for various linear algebraic operations. The list can be of any dimensions. |
|```np.ones(size)```| Will return a matrix full of 1s and having the same dimensions as the size variable. |
|```np.zeroes(size)```| Given the size of the array as a list for the parameter, it gives you back a null matrix of that size. |
|```np.eye(size)```| You give the size N as the parameter and it returns an identity matrix of size N*N. Here size variable has to be a single whole number. |
|```m.dot(n)```| If we suppose that m and n are two matrices that can be multiplied, then this code will give you the product of m and n. |
|`np.mean(m)`| Given a matrix m of any dimensions, first it will flatten the matrix to 1 dimension, and then calculate and return the mean of the flattened matrix. | 
|`np.median(m)`| Same as that of np.mean() except that this function will return the median of the flattened matrix. | 
|`np.std(m)`| Same as that of np.mean() except that this function will return the standard deviation of the flattened matrix. | 
|`np.linalg.inv(m)`| If you give a numpy matrix m of any dimension as the parameter to this function, it will return the inverse of m. | 
|`np.linalg.solve(a, b)`| Given two matrices a and b, this function will return a matrix containing the solution, which is X as per the following formula: \[a \cdot X = b\]| 

These were some basic numpy functions that you need to know to get started. For further reading, you can check out the [numpy documentation](https://numpy.org/doc/stable/reference/index.html).

### This was all for now, but I'll keep updating this document as and when I get time. I plan to add the basics of pandas here as well in the future. But for now, I hope this helps you out.

#####~ Soumyadeep Bose, 2k23 😅
