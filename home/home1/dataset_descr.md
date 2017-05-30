On previous lecture we were introduced to data cleaning.
Three main problems:
- Missing value treatment
- Outlier detection and treatment
- Feature engineering and transformation
For now we should mainly care about missing values and variables transformation, as other things will come up later on course.
Good summary and guide on data cleaning:
    https://www.analyticsvidhya.com/blog/2016/01/guide-data-exploration/


Homework task:
    Sample dataset: 'dataset.csv' - subset of Lending Club loan dataset.

There are two main types of variables in dataset - continuous an discrete (categorical).
All steps could be done with Pandas built-in functionality.

As a result you should provide an output dataset in csv and be able to explain transformation you've done.


1) Output dataframe should not contain missing values (NaN) - either drop or replace them with relevant values (they should be treated in each case individually).

2) Examine continuous type columns and make sure they follow normal distribution. If not, try to transform those (eg. apply log() function, etc.).

3) Locate columns with discrete (categorical) type and convert them into number representation (you can use label-encoding or one-hot-encoding)

4) If there is column with dates, output column should be in ISO format
    ( https://docs.python.org/3.5/library/datetime.html#datetime.date.isoformat )

5) Dump resulting dataframe to CSV-file "clean_dataset.csv".

In case you have troubles with any of steps, you may find some insights here:
    https://www.analyticsvidhya.com/blog/2016/01/complete-tutorial-learn-data-science-python-scratch-2/