# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 09:12:56 2014

@author: vincentli2010
"""


import numpy as np
from sklearn import linear_model
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import (LassoLarsCV, RandomizedLasso)


class LinearAll:
    """
    A repertoire of Linear Variable Selection and Prediction Models

    Parameters
    ----------
    n_jobs : int, optional
        Number of jobs to run in parallel (default 1).
        If -1 all CPUs are used. This will only provide speedup for
        n_targets > 1 and sufficient large problems
    pre_dispatch : int, or string, optional
        Controls the number of jobs that get dispatched during parallel execution. Reducing this number can be useful to avoid an explosion of memory consumption when more jobs get dispatched than CPUs can process. This parameter can be:
        None, in which case all the jobs are immediately created and spawned. Use this for lightweight and fast-running jobs, to avoid delays due to on-demand spawning of the jobs
        An int, giving the exact number of total jobs that are spawned
        A string, giving an expression as a function of n_jobs, as in ‘2*n_jobs’
    refit : boolean
        Refit the best estimator with the entire dataset. If “False”,
        it is impossible to make predictions using this GridSearchCV
        instance after fitting.
    iid : boolean, optional
        If True, the data is assumed to be identically distributed across
        the folds, and the score is computed from all samples individually,
        and not the mean loss across the folds.
        (If the number of data points is the same across folds, either
        returns the same thing)

    Attributes
    ----------
    ols_train,
    predictions models before variable selection
    predictions models after variable selection
    """

    def __init__ (self, cv=20, scoring = 'mean_squared_error',
                  n_jobs=1, refit=False, iid=False, pre_pred=True,
                  param_ridge_post=list(np.arange(1,3,0.1)),
                    rlasso_selection_threshold = 0.5):
        #self.__name__ = '__main__'
        """
        CAUTION: we changed to __main__ so that parallelization works
        """
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.refit = refit
        self.iid = iid
        self.pre_pred =pre_pred
        self.param_ridge_post = param_ridge_post
        self.rlasso_selection_threshold = rlasso_selection_threshold

    def run_models(self, X, y, param_ridge):
        """

        Prediction Models.

        OLS, PLS, Ridge

        """

        ##################################
        ## OLS CV
        ##################################
        #ols = linear_model.LinearRegression(fit_intercept=True,
        #                                          normalize=False,
        #                                          copy_X=True)
        #ols_cv_score = cross_validation.cross_val_score(
        #        ols, X, y,
        #        cv=self.cv, scoring=self.scoring,
        #        n_jobs=self.n_jobs)
        """
        self.ols_cv_score.shape = (cv,)
        """

        ##################################
        ## PLS CV
        ##################################
        tuned_parameters = [{'n_components': range(1, 5)}]
        pls = PLSRegression()
        pls_cv = GridSearchCV(pls, tuned_parameters,
                                cv=self.cv, scoring=self.scoring,
                                n_jobs=self.n_jobs,
                                refit=self.refit, iid=self.iid)
        pls_cv.fit(X, y)


        ##################################
        ## Ridge CV
        ##################################
        tuned_parameters = [{'alpha': param_ridge}]
        ridge = linear_model.Ridge(alpha = 1)
        ridge_cv = GridSearchCV(ridge, tuned_parameters,
                                     cv=self.cv, scoring=self.scoring,
                                     n_jobs=self.n_jobs,
                                     refit=self.refit, iid=self.iid)
        ridge_cv.fit(X, y)

        return (pls_cv, ridge_cv)

    def fit(self, X, y):
        """
        Variable Selection and Prediction.

        Variable Selection Model: lasso
        Prediction Models: see self.predict()

        Parameters
        ----------
        X : numpy array or sparse matrix of shape [n_samples,n_features]
            Training data
        y : numpy array of shape [n_samples, n_targets]
            Target values

        Returns
        -------
        self : returns an instance of self.
        """


        ##################################
        ## OLS Train
        ##################################
        #ols_train = linear_model.LinearRegression(fit_intercept=True,
        #                                         normalize=False,
        #                                          copy_X=True)
        #ols_train.fit(X, y)
        #self.rss_ols_train = np.sum((ols_train.predict(X) - y) ** 2)
        """
        fit_intercept=True, center the data
        copy=True, because centering data invovles X -= X_mean

        CAUTION:
        normalization=False, otherwise involves taking squares of X, lose precision

        self.rss_ols_train.shape = (1,1)
        """

        ##################################
        ## Pre Variable Selection Predictions
        ##################################
        self.pre_pred = False
        if self.pre_pred:
            print "Computing ... "
            param_ridge_pre = list(np.arange(1e9,2e9,1e8))
            self.pls_pre, self.ridge_pre = \
                self.run_models(X, y, param_ridge_pre)

        ##################################
        ## Lasso Variable Selection
        ##################################
        self.lasso_cv = LassoLarsCV(fit_intercept=True, normalize=True, precompute='auto',
                            max_iter=X.shape[1]+1000, max_n_alphas=X.shape[1]+1000,
                            eps= 2.2204460492503131e-16,copy_X=True,
                            cv=self.cv, n_jobs=self.n_jobs)
        self.lasso_cv.fit(X, y)
        """
        normalize=True, lasso seems to be able to handle itself
        """

        if self.rlasso_selection_threshold == 0:
            self.lasso_refit = linear_model.LassoLars(alpha=self.lasso_cv.alpha_,
                                fit_intercept=True, normalize=True, precompute='auto',
                                max_iter=X.shape[1]+1000,
                                eps=2.2204460492503131e-16, copy_X=True,
                                fit_path=False)
            self.lasso_refit.fit(X, y)
            self.active = self.lasso_refit.coef_ != 0
            self.active = self.active[0,:]
            X_selected = X[:, self.active]
        else:
            self.rlasso = RandomizedLasso(alpha=self.lasso_cv.alpha_, scaling=0.5,
                                          sample_fraction=0.75, n_resampling=200,
                                          selection_threshold=self.rlasso_selection_threshold, fit_intercept=True,
                                          verbose=False, normalize=True, precompute='auto',
                                          max_iter=500, eps=2.2204460492503131e-16,
                                          random_state=None, n_jobs=self.n_jobs, pre_dispatch='3*n_jobs',)
            self.rlasso.fit(X, y)
            X_selected = self.rlasso.transform(X)

        ##################################
        ## Post Variable Selection Predictions
        ##################################
        self.pls_post, self.ridge_post = \
            self.run_models(X_selected, y, self.param_ridge_post)


        return self

    def predict(self, X_test):
        assert(self.refit == True)
        if self.pls_post.best_score_ > self.ridge_post.best_score_:
            self.best_model = self.pls_post
            print "Chosen Model: pls"
        else:
            self.best_model = self.ridge_post
            print "Chosen Model: ridge"

        if self.rlasso_selection_threshold == 0:
            X_test_selected = X_test[:, self.active]
        else:
            X_test_selected = self.rlasso.transform(X_test)
        return self.best_model.best_estimator_.predict(X_test_selected)
















