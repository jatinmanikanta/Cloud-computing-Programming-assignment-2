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

public class WineQualityPredictionTraining {

    public static void main(String[] args) {
        SparkConf sparkConfig = new SparkConf().setAppName("QualityPrediction").setMaster("local");
        JavaSparkContext sparkCtx = new JavaSparkContext(sparkConfig);
        SparkSession sparkSession = SparkSession.builder().config(sparkConfig).getOrCreate();

        String datasetPath = args[0];
        
        // Loading the dataset
        Dataset<Row> dataset = sparkSession.read().format("csv").option("header", "true").option("sep", ";").load(datasetPath);
        dataset.printSchema();
        dataset.show();

        // Renaming 'quality' column to 'label'
        for (String columnName : dataset.columns()) {
            if (!columnName.equals("quality")) {
                dataset = dataset.withColumn(columnName, dataset.col(columnName).cast("float"));
            }
        }
        dataset = dataset.withColumnRenamed("quality", "label");

        // Feature assembly
        VectorAssembler featureAssembler = new VectorAssembler().setInputCols(dataset.columns()).setOutputCol("featureVector");
        Dataset<Row> transformedData = featureAssembler.transform(dataset).select("featureVector", "label");
        transformedData.show();

        // LabeledPoint conversion
        JavaRDD<LabeledPoint> labeledPoints = toLabeledPoints(sparkCtx, transformedData);

        // Model loading
        RandomForestModel model = RandomForestModel.load(sparkCtx.sc(), "/model/wine_quality.model/");

        System.out.println("Model loaded successfully");

        // Prediction
        JavaRDD<Double> predictionResults = model.predict(labeledPoints.map(LabeledPoint::features));

        // Label and Prediction pairing
        JavaRDD<Tuple2<Double, Double>> labelsAndPredictions = labeledPoints.map(lp -> new Tuple2<>(lp.label(), model.predict(lp.features())));

        // RDD to DataFrame conversion
        Dataset<Row> predictionDF = sparkSession.createDataFrame(labelsAndPredictions, Tuple2.class).toDF("label", "prediction");
        predictionDF.show();

        // Metrics calculation
        MulticlassMetrics evaluationMetrics = new MulticlassMetrics(predictionDF);
        double f1Score = evaluationMetrics.fMeasure();
        System.out.println("F1-score: " + f1Score);
        System.out.println(evaluationMetrics.confusionMatrix());
        System.out.println("Precision: " + evaluationMetrics.weightedPrecision());
        System.out.println("Recall: " + evaluationMetrics.weightedRecall());
        System.out.println("Accuracy: " + evaluationMetrics.accuracy());

        // Test error calculation
        long errorCount = labelsAndPredictions.filter(pair -> !pair._1().equals(pair._2())).count();
        System.out.println("Test Error = " + (double) errorCount / labeledPoints.count());

        sparkCtx.close();
    }

    private static JavaRDD<LabeledPoint> toLabeledPoints(JavaSparkContext ctx, Dataset<Row> dataFrame) {
        return dataFrame.toJavaRDD().map(row -> {
            double labelValue = row.getDouble(row.fieldIndex("label"));
            Vector featureVec = Vectors.dense(row.<Double>getAs("featureVector"));
            return new LabeledPoint(labelValue, featureVec);
        });
    }
}