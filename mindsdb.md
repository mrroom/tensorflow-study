## 참고 사이트
- [What’s New: MindsDB October 2020 Product Updates and Community Shoutouts](https://www.mindsdb.com/blog)
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

SELECT count, count_confidence
         FROM mindsdb.bikes_model
         WHERE datetime='2011-01-20 00:00:00' AND
               season='1' AND
               holiday='0' AND
               workingday='1' AND
               weather='1' AND
               temp='10.66' AND
               atemp='11.365' AND
               humidity='56' AND
               windspeed='26.0027';
~~~
