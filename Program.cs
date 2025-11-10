using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Vision;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Http;

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

        
        var preprocessing = mlContext.Transforms
            .LoadRawImageBytes(outputColumnName: "ImagePixels", imageFolder: DataSetFolder, inputColumnName: nameof(ImageData.ImagePath))
            .Append(mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "LabelKey", inputColumnName: nameof(ImageData.Label)));

        var options = new ImageClassificationTrainer.Options
        {
            FeatureColumnName = "ImagePixels",
            LabelColumnName = "LabelKey", 
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

        var pipeline = preprocessing
            .Append(trainer)
            .Append(mlContext.Transforms.Conversion.MapKeyToValue(outputColumnName: "PredictedLabel", inputColumnName: "PredictedLabel"));

        Console.WriteLine("\n🚀 Memulai pelatihan model...\n");

        try
        {
            EnsureResnetMetaFile();
        }
        catch (Exception ex)
        {
            Console.WriteLine("❌ Gagal menyiapkan file TensorFlow yang diperlukan:");
            Console.WriteLine(ex.Message);
            Console.WriteLine("\nSilakan salin file 'resnet_v2_50_299.meta' ke folder berikut:");
            Console.WriteLine($"   {Path.Combine(Path.GetTempPath(), "MLNET")}");
            Console.WriteLine("Atau periksa koneksi internet Anda lalu jalankan ulang program.");
            return;
        }

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

    
    static void EnsureResnetMetaFile()
    {
        var url = "https://aka.ms/mlnet-resources/meta/resnet_v2_50_299.meta";
        var dir = Path.Combine(Path.GetTempPath(), "MLNET");
        Directory.CreateDirectory(dir);
        var filePath = Path.Combine(dir, "resnet_v2_50_299.meta");

        if (File.Exists(filePath))
        {
            Console.WriteLine($"ℹ️ Meta file sudah ada: {filePath}");
            return;
        }

        Console.WriteLine("🔄 Meta file TensorFlow tidak ditemukan. Mencoba mengunduh...");

        using var http = new HttpClient() { Timeout = TimeSpan.FromMinutes(2) };

        http.DefaultRequestHeaders.UserAgent.ParseAdd("Mozilla/5.0 (compatible; MLNETDownloader/1.0)");

        var resp = http.GetAsync(url).GetAwaiter().GetResult();
        if (!resp.IsSuccessStatusCode)
        {
            throw new Exception($"Gagal mengunduh meta file. Status code: {resp.StatusCode}");
        }

        var bytes = resp.Content.ReadAsByteArrayAsync().GetAwaiter().GetResult();
        File.WriteAllBytes(filePath, bytes);
        Console.WriteLine($"✅ Meta file berhasil diunduh ke: {filePath}");
    }
}
