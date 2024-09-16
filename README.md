# ISFFSVM - Enhancing Imbalance Learning: A Novel Slack-Factor Fuzzy SVM Approach

Please cite the following paper if you are using this code. 

M. Tanveer, Anushka Tiwari, C.T. Lin. Enhancing Imbalance Learning: A Novel Slack-Factor Fuzzy SVM Approach. (Under revision in IEEE Transactions on Emerging Topics in Computational Intelligence)

In this paper, we propose an improved slack-factor-based FSVM (ISFFSVM) to tackle imbalance learning by introducing a novel location parameter. This parameter significantly advances the model by constraining the DEC hyperplane's extension, thereby mitigating the risk of misclassifying minority class samples. It ensures that majority class samples with slack factor scores approaching the location threshold are assigned lower fuzzy memberships, enhancing the model's discrimination capability.

*Installation*

Our ISFFSVM implementation requires the following dependencies:

- Python (>=3.7)
- NumPy (>=1.13)
- SciPy
- Scikit-Learn

Usage

Documentation

improvedSlackFactorFSVM.py

Parameter | Description
-----------|-------------
C          | float, optional (default=100). Regularization parameter. The strength of the regularization is inversely proportional to C. Must be positive. The penalty is a squared l2 term.
gamma      | float, optional (default='auto'). The kernel width of the radial basis function (rbf).
mu         | float, optional (default=0). Smoothing parameter. Its interval is [0, 1].
a          | float, optional (default=2). Location parameter.

Methods

Method                   | Description
--------------------------|-------------
fit(self, X, y)           | Build an ISFFSVM classifier on the training set (X, y).
predict(self, X)          | Predict class for X.
predict_proba(self, X)    | Predict class probabilities for X.
score(self, X, y)         | Return the average precision score on the given test data and labels.
calc_xi(self, X, y)       | Calculate the slack variables of samples X.
calc_dec_fun(self, X)     | Calculate the value of the decision function of samples X.

Example

demorun.py

In this Python script, we provide an example of how to use our implementation of ISFFSVM methods to perform classification.

Run the script with:

python demorun.py -data ./dataset/moon_1000_200_2.csv -n 5 



Dataset link: https://sci2s.ugr.es/keel/studies.php?cat=imb

