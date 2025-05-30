# Surrogate-Based Optimization with Simulated Annealing

This project implements a complete framework for surrogate-based optimization using Simulated Annealing (SA). It supports three surrogate models:
    - Radial Basis Function Interpolator ('rbf')
    - Random Forest ('rf')
    - Neural Network ('nn')

The SA optimizer operates on discrete feature values (0,1,2) with an option to restrict the number of spargers (1) to a fixed value (max_saprgers).

- Preprocessing the data ('get_csv.py')
    - reads the 'configs.pkl' and 'results.pkl' files from a study
    - saves the configuration in 'Xdata_{study_name}.csv' file
    - save the qoi and qoi_error in 'ydata_{study_name}.csv' file 

- Surrogate modeling and optimization ('get_optimal.py'/'get_optimal_with_constraints.py')
    - run_optimization(...) function sets up the surrogate-based optimization:
        - Inputs:
            - X: read from 'Xdata_{study_name}.csv' file
            - y: read from 'ydata_{study_name}.csv' file
            - model_type: type of surrogate model (default = 'rbf')
                - model_type = 'rbf': Radial Basis Function
                - model_type = 'rf': Random Forest
                - model_type = 'nn': Neural Network
            - max_spargers: maximum number of spargers (only in 'get_optimal_with_constraints.py') (default = 8)
            - n_runs: number of bootstrap runs (default = 10)
            - max_iters: maximum number of iterations of SA (default = 1000)
            - bootstrap_size: sample size of each boostrap (default = 100)
        - For each bootstrap run, the model hyperparameters are tuned using 5-fold cross validation. 
        - The simulated_annealing_surrogate(...) function runs the optimization:
            - If SA is too slow or fails to converge, you can change the following parameters:
                - temp: maximum temperature of SA. Controls exploration.
                - alpha: the rate at which temperature changes every iteration.
            - It returen the optimal solutions along with the optimization logs which are used to make plots in postprocessing.
        - Once the optimization is done the best solution is saved in a csv file and the following plots are made:
            - Mean-CI plot of the objective function (qoi)
            - Mean-CI plot of the distance of an iterate from the optimal solution (this is not done in 'get_optimal_with_constraints.py').

Python dependencies:
- numpy
- pandas
- scikit-learn
- scipy
- optuna
- matplotlib
- warnings
- random
- tensorflow
