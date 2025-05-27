package org.example;

import java.io.Serializable;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.io.Serializable;


import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.recommendation.ALS;
import org.apache.spark.ml.recommendation.ALSModel;
import scala.Tuple2;


public class CollabFilter {
    public static class Rating implements Serializable {
        private int userId;
        private int productId;
        private float rating;


        public Rating() {}

        public Rating(int userId, int productId, float rating) {
            this.userId = userId;
            this.productId = productId;
            this.rating = rating;

        }

        public int getUserId() {
            return userId;
        }

        public int getProductId() {
            return productId;
        }

        public float getRating() {
            return rating;
        }





        public static Rating parseRating(Tuple2<Tuple2<Integer, Integer>, Float> interaction) {
            int userId = interaction._1()._1();
            int productId = interaction._1()._2();
            float rating = interaction._2();

            return new Rating(userId, productId, rating);
        }
    }




    public static void main(String[] args) {
        SparkSession spark = SparkSession
                .builder()
                .appName("CollabFilter")
                .getOrCreate();


        JavaPairRDD<Tuple2<Integer, Integer>, Float> clicks = spark.read().textFile("/GP/input/data.csv")
                .javaRDD().filter(row -> (!row.contains("event_time") || !row.contains(",")))
                .mapToPair(s -> {
                    String[] row = s.split(",");
                    if(row.length < 8)
                        return new Tuple2<>(new Tuple2<>(-1, -1), (float)-1);
                    if(row.length == 1)
                        System.out.println(row);
                    float action = 0;
                    if(row[1].equals("view"))
                        action = (float)0.1;
                    if(row[1].equals("cart"))
                        action = (float)0.1;
                    if(row[1].equals("purchase"))
                        action = 1;
                    return new Tuple2<>(new Tuple2<>(Integer.parseInt(row[7]), Integer.parseInt(row[2])), action);
                })
                .reduceByKey((a,b) -> a + b);
        JavaPairRDD<Tuple2<Integer, Integer>, Float> rates = clicks.mapValues(s -> {
            float rating = 0;
            if(s <= 0.1)
                rating = 1;
            if(s < 1 && s > 0.1)
                rating = 2;
            if(s >= 1 && s < 2)
                rating = 3;
            if(s >= 2 && s < 4)
                rating = 4;
            if(s >= 4)
                rating = 5;
            return rating;
        });
        JavaRDD<Rating> userRating = rates.rdd().toJavaRDD().map(Rating::parseRating);




        Dataset<Row> ratings = spark.createDataFrame(userRating, Rating.class);
        Dataset<Row>[] splits = ratings.randomSplit(new double[]{0.8, 0.2});
        Dataset<Row> training = splits[0];
        Dataset<Row> test = splits[1];


        ALS als = new ALS()
                .setMaxIter(10)
                .setRegParam(0.1)
                .setUserCol("userId")
                .setItemCol("productId")
                .setRatingCol("rating");
        ALSModel model = als.fit(training);

        model.setColdStartStrategy("drop");

        Dataset<Row> predictions = model.transform(test);

        RegressionEvaluator evaluator = new RegressionEvaluator()
                .setMetricName("rmse")
                .setLabelCol("rating")
                .setPredictionCol("prediction");
        Double rmse = evaluator.evaluate(predictions);

        System.out.println();
        System.out.println("Root-mean-square error = " + rmse);
        System.out.println();



        Dataset<Row> userRecs = model.recommendForAllUsers(10);
        userRecs.show(20);




        spark.stop();




    }


}