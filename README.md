```markdown
# TrashClassifier (C# / ML.NET)

TrashClassifier adalah proyek klasifikasi sampah berbasis gambar yang diimplementasikan dengan C# menggunakan ML.NET. README ini menjelaskan cara menyiapkan lingkungan, melatih model menggunakan ML.NET (ImageClassification / transfer learning), melakukan inferensi, serta struktur data dan tips pengembangan.

## Fitur
- Pelatihan model klasifikasi gambar menggunakan ML.NET (Image Classification / transfer learning)
- Script/console app untuk inferensi (mengklasifikasikan gambar baru)
- Struktur dataset yang mudah digunakan oleh ML.NET
- Contoh kode C# untuk training dan prediksi

## Prasyarat
- .NET SDK 6.0 atau lebih baru (disarankan .NET 6/7/8 sesuai kebutuhan)
  - Unduh dari https://dotnet.microsoft.com/
- Paket NuGet utama (ditambahkan ke project):
  - Microsoft.ML
  - Microsoft.ML.ImageAnalytics
  - Microsoft.ML.Vision
  - Microsoft.ML.Data
- (Opsional) ML.NET CLI / Model Builder:
  - dotnet tool install -g mlnet

Contoh perintah instalasi paket:
```bash
dotnet add PACKAGE_NAME package Microsoft.ML
dotnet add PACKAGE_NAME package Microsoft.ML.ImageAnalytics
dotnet add PACKAGE_NAME package Microsoft.ML.Vision
```

## Struktur Dataset
ML.NET menerima gambar yang diberi label lewat dua pola umum:
1. Folder per label (recommended):
```
data/
  train/
    plastik/
      img1.jpg
      img2.jpg
    organik/
    kertas/
    kaca/
    logam/
  test/
    plastik/
    organik/
    ...
```
2. File CSV/TSV dengan dua kolom: ImagePath,Label
```
ImagePath,Label
data/train/plastik/img1.jpg,plastik
data/train/organik/img2.jpg,organik
```

Pilih salah satu sesuai implementasi di project. Jika menggunakan folder-per-label, buatlah path relatif yang sesuai saat membangun IDataView.

## Contoh: Pelatihan dengan ML.NET (ringkasan)
Berikut adalah alur umum training menggunakan ImageClassification (transfer learning) dalam C#:

- Buat class untuk input dan output:
```csharp
public class ImageData
{
    public string ImagePath { get; set; }
    public string Label { get; set; }
}

public class ImagePrediction
{
    public string PredictedLabel { get; set; }
    public float[] Score { get; set; }
}
```

- Contoh pipeline training singkat:
```csharp
var mlContext = new MLContext(seed: 1);

// load images into IDataView (misal load from IEnumerable<ImageData> or CSV)
IDataView fullData = mlContext.Data.LoadFromTextFile<ImageData>(path: "data/train_labels.csv", hasHeader: true, separatorChar: ',');

// split data
var trainTest = mlContext.Data.TrainTestSplit(fullData, testFraction: 0.2);
var trainData = trainTest.TrainSet;
var testData = trainTest.TestSet;

// pipeline
var pipeline = mlContext.Transforms.LoadImages(outputColumnName: "input", imageFolder: "", inputColumnName: nameof(ImageData.ImagePath))
    .Append(mlContext.Transforms.ResizeImages(outputColumnName: "input", imageWidth: 224, imageHeight: 224, inputColumnName: "input"))
    .Append(mlContext.Transforms.ExtractPixels(outputColumnName: "input"))
    .Append(mlContext.Model.ImageClassification("input", "Label", arch: ImageClassificationTrainer.Architecture.ResnetV2101, epochs: 50, batchSize: 16, learningRate: 0.01f))
    .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel", "PredictedLabel"));

// train
ITransformer trainedModel = pipeline.Fit(trainData);

// evaluate
var predictions = trainedModel.Transform(testData);
var metrics = mlContext.MulticlassClassification.Evaluate(predictions, labelColumnName: "Label", predictedLabelColumnName: "PredictedLabel");
Console.WriteLine($"MicroAccuracy: {metrics.MicroAccuracy}, MacroAccuracy: {metrics.MacroAccuracy}");
```

Catatan:
- Parameter seperti arsitektur, epochs, learningRate harus disesuaikan berdasarkan dataset dan GPU availability.
- ImageClassification trainer menggunakan TensorFlow/transfer learning di belakang layar; butuh dependencies native tertentu pada beberapa konfigurasi.

## Menyimpan dan Memuat Model
Simpan:
```csharp
mlContext.Model.Save(trainedModel, trainData.Schema, "model.zip");
```
Muat & buat PredictionEngine:
```csharp
ITransformer loadedModel = mlContext.Model.Load("model.zip", out var modelInputSchema);
var predictor = mlContext.Model.CreatePredictionEngine<ImageData, ImagePrediction>(loadedModel);

var sample = new ImageData { ImagePath = "sample.jpg" };
var result = predictor.Predict(sample);
Console.WriteLine($"Prediksi: {result.PredictedLabel}");
```

Untuk performa batch inference, gunakan Transform() pada IDataView, bukan PredictionEngine (PredictionEngine hanya untuk single prediction, tidak thread-safe).

## Menjalankan (build & run)
1. Clone repository:
```bash
git clone https://github.com/AyyMhmd/TrashClassifier.git
cd TrashClassifier
```

2. Restore & build:
```bash
dotnet restore
dotnet build -c Release
```

3. Jalankan aplikasi/skrip:
- Jika ada console app di folder src/ atau TrashClassifier.Console:
```bash
dotnet run --project src/TrashClassifier.Console --configuration Release -- --mode train --dataDir ./data
```
- Atau jalankan project training yang sesuai (sesuaikan argumen sesuai implementasi).

## Tips & Best Practices
- Gunakan GPU saat melatih model besar / epochs banyak. Pastikan runtime dan dependencies TensorFlow untuk GPU terpasang bila diperlukan.
- Lakukan augmentasi gambar (rotasi, flip, crop) sebelum training untuk meningkatkan generalisasi.
- Perhatikan class imbalance: gunakan sampling atau weighted loss bila beberapa kelas lebih sedikit.
- Pantau training loss dan accuracy; gunakan early stopping jika diperlukan.
- Jika dataset kecil, pertimbangkan transfer learning (ImageClassification trainer sudah melakukan ini).

## Struktur Proyek (contoh)
Sesuaikan dengan implementasi aktual repository Anda:
```
TrashClassifier/
├─ data/                # dataset (tidak di-commit)
├─ src/
│  ├─ TrashClassifier.Console/   # aplikasi CLI (train/infer)
│  ├─ TrashClassifier.Lib/       # kode training & inference (ML.NET)
│  └─ TrashClassifier.Tests/
├─ models/              # model.zip hasil training
├─ notebooks/           # (opsional) demo/analisis
├─ README.md
└─ LICENSE
```

## Kontribusi
- Fork repo -> buat branch -> commit -> pull request
- Sertakan deskripsi eksperimen, dataset sample, dan cara menjalankan ulang eksperimen.

## License
Tambahkan file LICENSE (mis. MIT) bila belum tersedia.

## Kontak
Pemilik repo: AyyMhmd
```
