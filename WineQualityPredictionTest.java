import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.RandomForest;
import org.apache.spark.mllib.tree.model.RandomForestModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class WineQualityPredictionTest {

    public static void main(String[] args) {
        SparkConf sparkConfig = new SparkConf().setAppName("WineQualityModelTrainer").setMaster("local");
        JavaSparkContext sparkContext = new JavaSparkContext(sparkConfig);
        SparkSession sparkSession = SparkSession.builder().config(sparkConfig).getOrCreate();

        String datasetPath = args[0];
        
        // Load the validation dataset
        Dataset<Row> validationData = sparkSession.read().format("csv").option("header", "true").option("sep", ";").load(datasetPath);
        validationData.printSchema();
        validationData.show();

        // Rename 'quality' column to 'label' and cast other columns to float
        for (String columnName : validationData.columns()) {
            if (!columnName.equals("quality")) {
                validationData = validationData.withColumn(columnName, validationData.col(columnName).cast("float"));
            }
        }
        validationData = validationData.withColumnRenamed("quality", "label");

        // Assemble features and label
        VectorAssembler featureAssembler = new VectorAssembler()
                .setInputCols(validationData.columns()).setOutputCol("featureVector");
        Dataset<Row> transformedData = featureAssembler.transform(validationData).select("featureVector", "label");
        transformedData.show();

        // Convert to JavaRDD<LabeledPoint>
        JavaRDD<LabeledPoint> trainingData = convertToLabeledPoint(sparkContext, transformedData);

        // Load the model
        RandomForestModel randomForestModel = RandomForestModel.load(sparkContext.sc(), "/winepredict/trainingmodel.model/");

        System.out.println("Model loaded successfully");

        // Predictions
        JavaRDD<Double> predictionValues = randomForestModel.predict(trainingData.map(LabeledPoint::features));

        // Label and Prediction RDD
        JavaRDD<Tuple2<Double, Double>> labelPredictionRDD = trainingData.map(dataPoint -> new Tuple2<>(dataPoint.label(), randomForestModel.predict(dataPoint.features())));

        // Convert RDD to DataFrame
        Dataset<Row> labelPredictionDF = sparkSession.createDataFrame(labelPredictionRDD, Tuple2.class).toDF("label", "prediction");
        labelPredictionDF.show();

        // Calculate the F1 score and other metrics
        MulticlassMetrics metrics = new MulticlassMetrics(labelPredictionDF);
        double f1Score = metrics.fMeasure();
        System.out.println("F1-score: " + f1Score);
        System.out.println(metrics.confusionMatrix());
        System.out.println("Precision: " + metrics.weightedPrecision());
        System.out.println("Recall: " + metrics.weightedRecall());
        System.out.println("Accuracy: " + metrics.accuracy());

        // Calculate test error
        long errorCount = labelPredictionRDD.filter(pair -> !pair._1().equals(pair._2())).count();
        System.out.println("Test Error = " + (double) errorCount / trainingData.count());

        sparkContext.close();
    }

    private static JavaRDD<LabeledPoint> convertToLabeledPoint(JavaSparkContext context, Dataset<Row> dataset) {
        return dataset.toJavaRDD().map(row -> {
            double labelValue = row.getDouble(row.fieldIndex("label"));
            Vector featureVector = Vectors.dense(row.<Double>getAs("featureVector"));
            return new LabeledPoint(labelValue, featureVector);
        });
    }
}