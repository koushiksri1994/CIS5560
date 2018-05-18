
<a href="http://www.calstatela.edu/centers/hipic"><img align="left" src="https://avatars2.githubusercontent.com/u/4156894?v=3&s=100"><image/>
</a>
<img align="right" alt="California State University, Los Angeles" src="http://www.calstatela.edu/sites/default/files/groups/California%20State%20University%2C%20Los%20Angeles/master_logo_full_color_horizontal_centered.svg" style="width: 360px;"/>

# CIS5560 Term Project Tutorial


------
#### Authors: [Shanmathi Arul Murugan](https://www.linkedin.com/in/shanmathiarul/); [Ashwin Karthik](https://www.linkedin.com/in/ashwin-karthik-b26ab172/); [Kaushik Sridharan](https://www.linkedin.com/in/kaushik-sridharan-35738865/)


#### Instructor: [Jongwook Woo](https://www.linkedin.com/in/jongwook-woo-7081a85)

#### Date: 05/18/2017



------
## Predictive Analysis of Salary for different Job Titles

------

### Objective:
The objective of this tutorial is to predict the Salary for different Job Titles in New York city based on the features present in the dataset by utilizing Machine Learning algorithms available with SparkML

------
### Creating a Cluster:

Sign in to your Databricks account and click on 'Clusters' option on the left pane. Press the '+Create Cluster' button to create a new cluster. Specify a name for the cluster and click on 'Create Cluster' button. The page looks like the following image. 

<img alt="Cluster" src="https://github.com/koushiksri1994/CIS5560/blob/master/cluster.JPG" style="width: 600px;"/>

#### Cluster Configurations:
* **Spark Version:** Apache Spark 2.3.0, Scala 2.11
* **Memory:** 6.0 GB Memory, 0.88 Cores, 1 DBU
* **File System:** DBFS (Data Bricks File System)

------
### Create table and prepare the data

To import the dataset click on 'Data' option on the left pane. Click on '+' button at the top to create a new table. Click on the 'browse' link to upload the dataset. Once the dataset is uploaded provide a suitable name for the table and click on 'Preview table with UI'. Select the appropriate cluster.

<img alt="Table" src="https://github.com/koushiksri1994/CIS5560/blob/master/table.JPG" style="width: 600px;"/>

Select the apppropriate DataType for each column for the dataset.
* int - Numeric values
* float - Decimal values
* bignit - Values greater that 65000
* string - Character values



------
### Import Libraries

Import the necessary Spark SQL and Spark ML Libraries to prepare the data.

```python
from pyspark.sql.types import *
from pyspark.sql.functions import *

from pyspark.ml import Pipeline
from pyspark.ml.regression import GBTRegressor, LinearRegression
from pyspark.ml.feature import VectorAssembler, OneHotEncoder
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit, CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator, RegressionEvaluator
from pyspark.ml.classification import DecisionTreeClassifier , RandomForestClassifier
```

### Table Location

This step is to display the location of the table and its size.

```python
%fs ls /FileStore/tables/Book3a.csv
```


### Import the data

This step is to import the data that is stored in the table created with the help of SQL query. Only the necessary columns for the analysis are selected using the WHERE command from the table. The columns selected are Job_ID, Posting_Type, Title_Code_No, Level, FullTime_PartTime, Salary_Range_To, Salary_Frequency and Hours_Shift. The column Salary_Range_To is renamed as label using the alias method. 



```python
jobs = spark.sql("SELECT cast(Job_ID as double),cast(Posting_Type as int),cast(Title_Code_No as double),cast(Level as int),cast(FullTime_PartTime as int),cast(Salary_Range_To as double),cast(Salary_Frequency as int),cast(Hours_Shift as int) FROM nycjobs")
```


```python
data = jobs1.select('Job_ID','Posting_Type','Title_Code_No','Level','FullTime_PartTime',col('Salary_Range_To').alias('label'),'Salary_Frequency','Hours_Shift')
```

### Split the Data

Supervised learning requires to split the data into two parts, one to train the model and the other to test the trained model. Here 70% of the data is used for training and 30% for testing.

```python
splits = data.randomSplit([0.7, 0.3])
train = splits[0]
test = splits[1].withColumnRenamed("label", "trueLabel")
```

## Gradient Booster Tree Regression

The algorithm used for the first model to train and test the data for prediction is Gradient Booster Tree Regression. It has the label column passed as the parameter.

### Vector Assembler

Create a Vector Assembler that would assemble the all the columns that are selected as features for the prediction and prepare it for the pipeline. A Vector Assembler transforms all the feature columns into a vector. 

```python
assembler = VectorAssembler(inputCols = ['Job_ID','Posting_Type','Title_Code_No','Level','FullTime_PartTime','Salary_Frequency','Hours_Shift'], outputCol="features")
gbt = GBTRegressor(labelCol="label")
```

### Tune Parameters

Tune parameters is used to find the best model for the data. The CrossValidator class can be used to evaluate each combination of parameters defined in a ParameterGrid against multiple folds of the data split into training and validation datasets, in order to find the best performing parameters. 

Using a training set and a validation set could result in over fitting the model which might not always produce the optimal model with the optimal parameters. Hence, a cross validator is being used.


```python
paramGrid = ParamGridBuilder()\
  .addGrid(gbt.maxDepth, [2, 5])\
  .addGrid(gbt.maxIter, [10, 100])\
  .build()
  
evaluator = RegressionEvaluator(metricName="rmse", labelCol=gbt.getLabelCol(), predictionCol=gbt.getPredictionCol())

cv = CrossValidator(estimator=gbt, evaluator=evaluator, estimatorParamMaps=paramGrid)
```

### Define the Pipeline

To train the regression model, a pipeline has to be defined that requires two stages as input. One is the Vector Assembler and the other is the Cross Validator. In the next line the train model is fitted.

```python
pipeline = Pipeline(stages=[assembler, cv])
pipelineModel = pipeline.fit(train)
```

### Test the Model

The pipeline Model can now be applied to the test data to arrive at a prediction. The feature columns, transformed test data and the label column are used for the prediction.

```python
predictions = pipelineModel.transform(test)
```

```python
predicted = predictions.select("features", "prediction", "trueLabel")
predicted.show(100)
```


### Plot the Values

The action values and the predicted values which is the result of prediction from the Transform, a bar chart is plotted to visualize the results.

```python
predicted.createOrReplaceTempView("regressionPredictions")
```

```python
dataPred = spark.sql("SELECT trueLabel, prediction FROM regressionPredictions")
display(dataPred)
```

### RMSE Analysis

Using the evaluation metric as RMSE(Root Mean Squared Error), the Gradient Regression model performance is calculated.


```python
evaluator  = RegressionEvaluator(labelCol="trueLabel", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print "Root Mean Square Error (RMSE) for Gradient Booster Tree Regression :", rmse
```

The RMSE(Root Mean Squared Error) value is 17023.2912958. The result depicts how much variation would a perdiction from this model would make. This is the value of Salary range.

## Linear Regression

The algorithm used for the first model to train and test the data for prediction is Linear Regression. It has the label column passed as the parameter.

### Vector Assembler

Create a Vector Assembler that would assemble the all the columns that are selected as features for the prediction and prepare it for the pipeline. A Vector Assembler transforms all the feature columns into a vector. 

### Define the Pipeline

To train the regression model, a pipeline has to be defined that requires two stages as input. One is the Vector Assembler and the other is the Linear Regresssion algorithm.

```python
assembler = VectorAssembler(inputCols = ['Job_ID','Posting_Type','Title_Code_No','Level','FullTime_PartTime','Salary_Frequency','Hours_Shift'], outputCol="features")
lr = LinearRegression(labelCol="label",featuresCol="features", maxIter=10, regParam=0.3)
pipeline1 = Pipeline(stages=[assembler, lr])
```

### Tune Parameters

Tune parameters is used to find the best model for the data. A TrainValidationSplit is used to evaluate each combination of parameters defined in a ParameterGrid against a subset of the training data in order to find the best performing parameters.

```python
paramGrid1 = ParamGridBuilder().addGrid(lr.regParam, [0.3, 0.01]).addGrid(lr.maxIter, [10, 5]).build()
trainval = TrainValidationSplit(estimator=pipeline1, evaluator=RegressionEvaluator(), estimatorParamMaps=paramGrid1, trainRatio=0.8)
```

```python
pipelineModel = trainval.fit(train)
```

### Test the Model

The pipeline Model can now be applied to the test data to arrive at a prediction. The feature columns, transformed test data and the label column are used for the prediction

```python
predictions = pipelineModel.transform(test)
```

```python
predicted = predictions.select("features", "prediction", "trueLabel")
predicted.show(100)
```

### Plot the Values

The action values and the predicted values which is the result of prediction from the Transform, a bar chart is plotted to visualize the results.

```python
predicted.createOrReplaceTempView("regressionPredictions")
```

```python
dataPred = spark.sql("SELECT trueLabel, prediction FROM regressionPredictions")
display(dataPred)
```

### RMSE Analysis

Using the evaluation metric as RMSE(Root Mean Squared Error), the Gradient Regression model performance is calculated.


```python
evaluator  = RegressionEvaluator(labelCol="trueLabel", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print "Root Mean Square Error (RMSE) for Linear Regression :", rmse
```

The RMSE(Root Mean Squared Error) value is 7955.1756615. The result depicts how much variation would a perdiction from this model would make. This is the value of Salary range.

### References:

1. [Essentials of Machine Learning Algorithms](https://www.analyticsvidhya.com/blog/2017/09/common-machine-learning-algorithms/)
1. [Classification and regression - Spark 2.3.0](https://people.apache.org/~pwendell/spark-nightly/spark-master-docs/latest/ml-classification-regression.html)
1. [Machine Learning Algorithms](https://github.com/apache/spark/tree/master/examples/src/main/python/mllib)
