using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Vision;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

public class ImageData
{
    [LoadColumn(0)]
    public string ImagePath { get; set; } = string.Empty;

    [LoadColumn(1)]
    public string Label { get; set; } = string.Empty;
}

public class ImagePrediction
{
    [ColumnName("PredictedLabel")]
    public string PredictedLabel { get; set; } = string.Empty;

    public float[] Score { get; set; } = Array.Empty<float>();
}

class Program
{
    static readonly string DataSetFolder = "dataset";
    static readonly string ModelPath = "trash_model.zip";
    static readonly string[] ValidLabels = { "organik", "anorganik", "b3", "kertas" };

    static void Main(string[] args)
    {
        var mlContext = new MLContext(seed: 123);

        if (!Directory.Exists(DataSetFolder))
        {
            Console.WriteLine($"❌ Folder '{DataSetFolder}' tidak ditemukan.");
            return;
        }

        var csvPath = CreateImageLabelCsv(DataSetFolder, "dataset_labels.csv");
        Console.WriteLine($"✅ File CSV dibuat: {csvPath}");

        var imageData = mlContext.Data.LoadFromTextFile<ImageData>(csvPath, hasHeader: false, separatorChar: ',');

        var split = mlContext.Data.TrainTestSplit(imageData, testFraction: 0.2);
        Console.WriteLine("📊 Data dibagi: 80% latih, 20% uji");

        // --- Use LoadRawImageBytes for ImagePixels (VarVector<Byte>)
        // --- Map the string label to a key column ("LabelKey") expected by the trainer
        var preprocessing = mlContext.Transforms
            .LoadRawImageBytes(outputColumnName: "ImagePixels", imageFolder: DataSetFolder, inputColumnName: nameof(ImageData.ImagePath))
            .Append(mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "LabelKey", inputColumnName: nameof(ImageData.Label)));

        // Trainer options
        var options = new ImageClassificationTrainer.Options
        {
            FeatureColumnName = "ImagePixels",
            LabelColumnName = "LabelKey", // use the key column produced above
            Epoch = 50,
            BatchSize = 32,
            MetricsCallback = metrics =>
            {
                try
                {
                    var props = metrics.GetType().GetProperties();
                    Console.WriteLine("   Epoch metrics:");
                    foreach (var p in props)
                    {
                        var val = p.GetValue(metrics);
                        Console.WriteLine($"     - {p.Name}: {val}");
                    }
                }
                catch
                {
                    Console.WriteLine("   Epoch Selesai");
                }
            }
        };

        var trainer = mlContext.MulticlassClassification.Trainers.ImageClassification(options);

        // Append MapKeyToValue to convert predicted key back to original string
        var pipeline = preprocessing
            .Append(trainer)
            .Append(mlContext.Transforms.Conversion.MapKeyToValue(outputColumnName: "PredictedLabel", inputColumnName: "PredictedLabel"));

        Console.WriteLine("\n🚀 Memulai pelatihan model...\n");
        var model = pipeline.Fit(split.TrainSet);
        Console.WriteLine("\n✅ Pelatihan selesai!");

        var predictions = model.Transform(split.TestSet);

        // Evaluate using the key label column the trainer expects
        var metrics = mlContext.MulticlassClassification.Evaluate(predictions, labelColumnName: "LabelKey", scoreColumnName: "Score");

        Console.WriteLine("\n📈 METRIK EVALUASI MODEL:");
        Console.WriteLine($"   Akurasi (Micro): {metrics.MicroAccuracy:P2}");
        Console.WriteLine($"   Akurasi (Macro): {metrics.MacroAccuracy:P2}");
        Console.WriteLine($"   Log Loss:        {metrics.LogLoss:F4}");

        mlContext.Model.Save(model, split.TrainSet.Schema, ModelPath);
        Console.WriteLine($"\n💾 Model disimpan sebagai: {ModelPath}");

        // Prediksi contoh
        var testImages = Directory.GetFiles(Path.Combine(DataSetFolder, "organik"), "*.jpg").Take(1).ToList();
        if (!testImages.Any())
            testImages = Directory.GetFiles(DataSetFolder, "*.jpg", SearchOption.AllDirectories).Take(1).ToList();

        if (testImages.Any())
        {
            Console.WriteLine("\n🔍 Contoh Prediksi:");
            PredictSingleImage(mlContext, ModelPath, testImages.First());
        }

        Console.WriteLine("\n✅ Selesai. Tekan Enter untuk keluar.");
        Console.ReadLine();
    }

    static string CreateImageLabelCsv(string rootFolder, string outputPath)
    {
        var validSet = new HashSet<string>(ValidLabels);
        using var writer = new StreamWriter(outputPath);
        foreach (var dir in Directory.GetDirectories(rootFolder))
        {
            var label = Path.GetFileName(dir)?.ToLower().Trim();
            if (string.IsNullOrEmpty(label) || !validSet.Contains(label)) continue;

            foreach (var file in Directory.GetFiles(dir, "*.*"))
            {
                var ext = Path.GetExtension(file).ToLowerInvariant();
                if (ext is ".jpg" or ".jpeg" or ".png")
                {
                    var relativePath = Path.GetRelativePath(rootFolder, file).Replace("\\", "/");
                    writer.WriteLine($"{relativePath},{label}");
                }
            }
        }
        return Path.GetFullPath(outputPath);
    }

    static void PredictSingleImage(MLContext mlContext, string modelPath, string imagePath)
    {
        if (!File.Exists(imagePath))
        {
            Console.WriteLine($"Gambar tidak ditemukan: {imagePath}");
            return;
        }

        var loadedModel = mlContext.Model.Load(modelPath, out var schema);
        var predictionEngine = mlContext.Model.CreatePredictionEngine<ImageData, ImagePrediction>(loadedModel);

        var relativePath = Path.GetRelativePath(DataSetFolder, imagePath).Replace("\\", "/");

        // MapValueToKey is part of the pipeline, so the prediction input must include a label value
        // that exists in the training label vocabulary. Provide a safe default (first valid label).
        var input = new ImageData { ImagePath = relativePath, Label = ValidLabels[0] };

        var prediction = predictionEngine.Predict(input);

        Console.WriteLine($"   File: {Path.GetFileName(imagePath)}");
        Console.WriteLine($"   Prediksi: {prediction.PredictedLabel}");
        Console.WriteLine($"   Skor per kelas:");

        var labels = ValidLabels;
        var scoresWithLabels = labels.Zip(prediction.Score, (lbl, score) => new { Label = lbl, Score = score })
                                     .OrderByDescending(x => x.Score)
                                     .Take(3);

        foreach (var item in scoresWithLabels)
        {
            Console.WriteLine($"     - {item.Label}: {item.Score:P2}");
        }
    }
}