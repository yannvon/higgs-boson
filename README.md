# Higgs boson detection
In this project for the EPFL machine learning class, we will do exploratory data analysis to understand the dataset and its features, feature processing and engineering to clean the dataset and extract more meaningful information, implement and use machine learning methods on real data, analyze our model and generate predictions using  those methods and report our findings.

![The ATLAS experiment](pictures/atlas.jpg)

The dataset stems from one of the most popular machine learning challenges recently - finding the Higgs boson - using original data from CERN.

## Team members

- Benno Schneeberger
- Tiago Kieliger
- Yann Vonlanthen



## Original Data Description taken from kaggle.com

- **train.csv** - Training set of 250000 events. The file  starts with the ID column, then the label column (the y you have to  predict), and finally 30 feature columns.
- **test.csv** -The test set of around 568238 events - Everything as above, except the label is missing.
- **sample-submission.csv** - a sample submission file in the correct format. The sample submission always predicts -1, that is 'background'.

For detailed information on the semantics of the features, labels, and weights, see [the earlier official kaggle competition by CERN](https://www.kaggle.com/c/higgs-boson), or also [the technical documentation](http://higgsml.lal.in2p3.fr/documentation) from  the LAL website on the task. Note that here for the EPFL course, we use  a simpler evaluation metric instead (classification error).

**Some details to get started:**

- all variables are floating point, *except PRI_jet_num* which is integer
- variables prefixed with *PRI* (for *PRImitives*) are “raw” quantities about the bunch collision as measured by the detector.
- variables prefixed with *DER* (for *DERived*) are quantities computed from the primitive features, which were selected by the physicists of ATLAS.
- it can happen that for some entries some variables are meaningless  or cannot be computed; in this case, their value is −999.0, which is  outside the normal range of all variables.



## Approach

### Step 1: Implement basic ML functions



## Further ideas

- [ ] Use cross-validation
- [ ] Hyper parameter selection

