## 참고 사이트
- [Machine Learning straight through SQL](https://mariadb.org/machine-learning-sql/)
- [Bike Sharing Demand](https://www.kaggle.com/c/bike-sharing-demand/)


## mariadb table test
~~~
DROP TABLE test.bike_data;

CREATE TABLE test.bike_data (
  datetime datetime DEFAULT NULL,
  season int(11) DEFAULT NULL,
  holiday int(11) DEFAULT NULL,
  workingday int(11) DEFAULT NULL,
  weather int(11) DEFAULT NULL,
  temp double DEFAULT NULL,
  atemp double DEFAULT NULL,
  humidity double DEFAULT NULL,
  windspeed double DEFAULT NULL,
  casual int(11) DEFAULT NULL,
  registered int(11) DEFAULT NULL,
  count int(11) DEFAULT NULL);
  
 LOAD DATA INFILE 'bike-sharing-demand/train.csv' INTO TABLE bike_data columns terminated by ','  IGNORE 1 LINES;
 
INSERT INTO mindsdb.predictors
       (name, predict, select_data_query)
VALUES ('bikes_model', 'count', 'SELECT * FROM test.bike_data');

~~~
