# Distributional Forecasting in Electricity Markets: Prediction Interval Averaging vs QRA

This notebook includes suplementary data and code for master's thesis "Distributional Forecasting in Electricity Markets: Prediction Interval Averaging vs QRA".
We use Turkish day-ahead electricity prices to test whether prediction interval combinations can outperform Quantile Regression Averaging (QRA).
While QRA solely uses point forecasts to generate distributional forecast, PI combinations are gathered from different models including Autoregressive (ARX), multi-day Autoregressive and threshold autoregressive (TARX).
In addition We include two benchmark models, namely Naive and Autoregressive benchmark.
We check reliability of PI forecasts with Unconditional (UC) and Conditional (CC) Coverage.
We condition the CC on 1st lag, yet, the function is written in a way to allow for different lags.
To statistically draw a conclusion we use Kupiec and Christoffersen test for UC and CC respectively.
To assess the forecasting performances we use the Winkler score.
And we utilize one-sided Diebold-Mariano test to compare Winkler score series of models, consequently, the their forecasting performance.

Currently the master's thesis is not published here since it is still in evaluation process. Once the final grade is available, the master's thesis will also be available here.

To check the explanatory view notebook, [click](https://github.com/burakbalaban/distributional_forecasting/blob/main/explanatory_view.ipynb) or [![nbviewer](https://user-images.githubusercontent.com/2791223/29387450-e5654c72-8294-11e7-95e4-090419520edb.png)](https://nbviewer.jupyter.org/github/burakbalaban/distributional_forecasting/blob/main/explanatory_view.ipynb)

To check the modeling and forecasting process, [click](https://github.com/burakbalaban/distributional_forecasting/blob/main/Analysis.ipynb) or [![nbviewer](https://user-images.githubusercontent.com/2791223/29387450-e5654c72-8294-11e7-95e4-090419520edb.png)](https://nbviewer.jupyter.org/github/burakbalaban/distributional_forecasting/blob/main/Analysis.ipynb)

To check the PI generation and plotting, [click](https://github.com/burakbalaban/distributional_forecasting/blob/main/PI_generation_and_plots.ipynb) or [![nbviewer](https://user-images.githubusercontent.com/2791223/29387450-e5654c72-8294-11e7-95e4-090419520edb.png)](https://nbviewer.jupyter.org/github/burakbalaban/distributional_forecasting/blob/main/PI_generation_and_plots.ipynb)

