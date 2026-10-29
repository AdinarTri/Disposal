# Deteksi pelanggaran area kendaraan

Jalankan dari folder ini setelah memasang `requirements.txt`:

```powershell
python main.py --video video.mp4 --model vehicle.pt --config config.json --output outputs/run_001
```

`config.json` memakai koordinat asli 2560×1440 sesuai polygon yang diberikan. Resolusi video harus sama dengan `reference_width` dan `reference_height`; aplikasi akan berhenti dengan pesan jelas bila berbeda. Video contoh dalam folder ini berukuran 1920×1080, sehingga `config_video_example.json` menyediakan salinan polygon yang diskalakan 0,75 untuk video tersebut.

```powershell
python main.py --video 00000006772000900.mp4 --model 1786706505245-Vehicle_Detection_Disposal_CK_V7.pt --config config_video_example.json --output outputs/run_001
```

Hasil setiap run: `annotated.mp4`, `violations.csv`, `config_used.json`, dan `summary.json`. Folder output harus belum ada. Model hanya dimuat sekali; deteksi dan laporan memakai koordinat frame asli. `target_classes` memilih kelas yang dinilai melanggar; `occluder_classes` memilih kelas kendaraan yang ikut diperiksa sebagai penghalang. Nama kelas harus sama dengan yang tersimpan di model.

Occlusion versi ini memakai overlap bbox: kedua kendaraan dikecualikan jika luas irisannya positif. Track ID memakai pencocokan IoU sederhana; pergantian ID akibat gerakan cepat atau overlap berat dapat menghasilkan event baru dan perlu diperiksa pada video hasil.
