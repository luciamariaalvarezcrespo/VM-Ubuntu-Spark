# Source: https://github.com/h2oai/h2o-3/blob/master/h2o-py/demos/H2O_tutorial_eeg_eyestate.ipynb

import h2o

# Start an H2O Cluster on your local machine
h2o.init()

# This will not actually do anything since it's a fake IP address
# h2o.init(ip="123.45.67.89", port=54321)

# Download EEG Data
#csv_url = "http://www.stat.berkeley.edu/~ledell/data/eeg_eyestate_splits.csv"
csv_url = "https://h2o-public-test-data.s3.amazonaws.com/smalldata/eeg/eeg_eyestate_splits.csv"
data = h2o.import_file(csv_url)

# Dimension of the frame
data.shape

# Top of the frame
data.head()

# Column names
data.columns

# Select a subset of the columns to look at
columns = ['AF3', 'eyeDetection', 'split']
data[columns].head()

# Let's select a single column
y = 'eyeDetection'
data[y]

data[y].unique()

# Convert to a factor
data[y] = data[y].asfactor()

# Now we can check that there are two levels in our response column
data[y].nlevels()

# We can query the categorical "levels" as well
data[y].levels()

# To figure out which, if any, values are missing, we can use the isna method on the diagnosis column
data.isna()
data[y].isna()

# If there are no missing values, then summing over the whole column should produce a summand equal to 0.0
data[y].isna().sum()

data.isna().sum()

# Let's take a look at the distribution
data[y].table()

# Let's calculate the percentage that each class represents
n = data.shape[0]  # Total number of training samples
data[y].table()['Count']/n

# Split H2O Frame into a train and test set
# Subset the data H2O Frame on the "split" column
train = data[data['split']=="train"]
train.shape

valid = data[data['split']=="valid"]
valid.shape

test = data[data['split']=="test"]
test.shape

# Machine Learning in H2O
# Train and Test a GBM model
# Import H2O GBM:
from h2o.estimators.gbm import H2OGradientBoostingEstimator

# We first create a model object of class, "H2OGradientBoostingEstimator"
model = H2OGradientBoostingEstimator(distribution='bernoulli',
                                    ntrees=100,
                                    max_depth=4,
                                    learn_rate=0.1)

# Specify the predictor set and response
x = list(train.columns)
x

del x[14:16]  #Remove the 14th and 15th columns, 'eyeDetection' and 'split'
x

# Now that we have specified x and y, we can train the model
model.train(x=x, y=y, training_frame=train, validation_frame=valid)

# Inspect Model
print(model)

# Model Performance on a Test Set
perf = model.model_performance(test)
print(perf.__class__)

# Area Under the ROC Curve (AUC)
perf.auc()
perf.mse()

# Cross-validated Performance
cvmodel = H2OGradientBoostingEstimator(distribution='bernoulli',
                                       ntrees=100,
                                       max_depth=4,
                                       learn_rate=0.1,
                                       nfolds=5)

cvmodel.train(x=x, y=y, training_frame=data)

print(cvmodel.auc(train=True))
print(cvmodel.auc(xval=True))

# Grid Search
ntrees_opt = [5,50,100]
max_depth_opt = [2,3,5]
learn_rate_opt = [0.1,0.2]

hyper_params = {'ntrees': ntrees_opt, 
                'max_depth': max_depth_opt,
                'learn_rate': learn_rate_opt}

# Define an "H2OGridSearch" object by specifying the algorithm (GBM) and the hyper parameters
from h2o.grid.grid_search import H2OGridSearch

gs = H2OGridSearch(H2OGradientBoostingEstimator, hyper_params = hyper_params)

# Train all the models in the grid
gs.train(x=x, y=y, training_frame=train, validation_frame=valid)

# Compare Models
print(gs)

# print out the auc for all of the models
auc_table = gs.sort_by('auc(valid=True)',increasing=False)
print(auc_table)

# The "best" model in terms of validation set AUC is listed first in auc_table
best_model = h2o.get_model(auc_table['Model Id'][0])
best_model.auc()

# Generate predictions on the test set using the "best" model, and evaluate the test set AUC
best_perf = best_model.model_performance(test)
best_perf.auc()
