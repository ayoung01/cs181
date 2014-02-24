# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 09:12:56 2014

@author: vincentli2010
"""

class LinearModels:
    
    def __init__ (self):
        pass
    
    def fit(self, X, y, 
            cv=20, scoring = 'mean_squared_error', 
            n_jobs=1, pre_dispatch=None,
            refit=False, iid=False):
        
        """
        Fit linear models.

        Parameters
        ----------
        X : numpy array or sparse matrix of shape [n_samples,n_features]
            Training data
        y : numpy array of shape [n_samples, n_targets]
            Target values
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
            the folds, and the loss minimized is the total loss per sample,
            and not the mean loss across the folds.
        Returns
        -------
        self : returns an instance of self.
        """
        
        import numpy as np
        from sklearn import linear_model
        from sklearn import cross_validation
        from sklearn.grid_search import GridSearchCV
        from sklearn.cross_decomposition import PLSRegression
        ##################################
        ## OLS Train
        ##################################        
        ols_train = linear_model.LinearRegression(fit_intercept=True, 
                                                  normalize=False,
                                                  copy=True)             
        ols_train.fit(X, y)      
        self.rss_ols_train = np.sum((ols_train.predict(X) - y) ** 2)        
        """                                         
        fit_intercept=True, center the data           
        copy=True, because centering data invovles X -= X_mean
        
        CAUTION:
        normalization=False, otherwise involves taking squares of X, lose precision    
        
        self.rss_ols_train.shape = (1,1)        
        """        
        
        
        ##################################
        ## OLS CV
        ##################################        
        ols_cv = linear_model.LinearRegression(fit_intercept=True, 
                                                  normalize=False,
                                                  copy=True) 
        self.rss_ols_cv = cross_validation.cross_val_score(
                ols_cv, X, y,
                cv=cv, scoring=scoring,
                n_jobs=n_jobs, pre_dispatch=pre_dispatch)    
        """

        
        self.rss_ols_cv.shape = (cv,)
        """

        ##################################
        ## PLS CV
        ##################################        
        tuned_parameters = [{'n_components': range(1, 20)}]        
        pls = PLSRegression()
        pls_tune = GridSearchCV(pls, tuned_parameters, 
                                cv=cv, scoring=scoring, 
                                n_jobs=n_jobs, pre_dispatch=pre_dispatch,
                                refit=refit, iid=iid)
        pls_tune.fit(X, y)
        self.pls_grid_scores = pls_tune.grid_scores_






















        
        