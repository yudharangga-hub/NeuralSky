def get_weather_impact(cloud_type):
    """
    Expert System: Menerjemahkan klasifikasi awan menjadi data operasional & Simulasi Masa Depan.
    Menggunakan pendekatan probabilistik untuk validitas ilmiah.
    """
    cloud = cloud_type.lower()

    knowledge_base = {
        'cirriform clouds': {
            'nama_indo': 'AWAN CIRRUS (Awan Tinggi)',
            'risiko_penerbangan': 'RENDAH (Low)',
            'potensi_hujan': 'Nihil',
            'visibilitas': 'Sangat Baik (> 10 km)',
            'rekomendasi': 'Kondisi aman untuk penerbangan VFR/IFR. Tidak ada signifikansi cuaca ekstrem.',
            'simulasi': 'PROYEKSI T+1: Stabilitas atmosfer lapisan atas terpantau stabil. Potensi kondisi cerah berawan (Partly Cloudy) cenderung bertahan tanpa pertumbuhan vertikal signifikan.',
            'kode_warna': (0, 255, 0)
        },
        'clear sky': {
            'nama_indo': 'LANGIT CERAH (Tanpa Awan)',
            'risiko_penerbangan': 'AMAN',
            'potensi_hujan': 'Nihil',
            'visibilitas': 'Sangat Baik',
            'rekomendasi': 'Cuaca cerah. Ideal untuk semua aktivitas operasional.',
            'simulasi': 'PROYEKSI T+1: Kondisi visual CAVOK (Ceiling and Visibility OK) diproyeksikan bertahan. Radiasi matahari maksimum mencapai permukaan tanah.',
            'kode_warna': (0, 255, 0)
        },
        'cumulonimbus clouds': {
            'nama_indo': 'AWAN CUMULONIMBUS (Awan Badai)',
            'risiko_penerbangan': 'BAHAYA (CRITICAL)',
            'potensi_hujan': 'Tinggi (Hujan Lebat/Petir/Angin Kencang)',
            'visibilitas': 'Buruk pada area inti (< 1 km)',
            'rekomendasi': 'HINDARI PENERBANGAN. Potensi Microburst dan Turbulensi Hebat. Terbitkan SIGMET segera.',
            'simulasi': 'PROYEKSI T+1: DETEKSI KONVEKTIF KUAT. Indikasi fase matang sel awan yang berpotensi memicu hujan lebat, kilat, dan angin kencang (Gust) dalam radius sel.',
            'kode_warna': (255, 0, 0)
        },
        'cumulus clouds': {
            'nama_indo': 'AWAN CUMULUS (Awan Putih)',
            'risiko_penerbangan': 'SEDANG (Turbulensi Ringan)',
            'potensi_hujan': 'Rendah (Hujan Lokal sesaat)',
            'visibilitas': 'Baik',
            'rekomendasi': 'Waspada pertumbuhan menjadi Cumulonimbus pada siang/sore hari karena konveksi.',
            'simulasi': 'PROYEKSI T+1: Fase pertumbuhan vertikal awal. Jika pemanasan permukaan berlanjut, awan dapat berkembang menjadi Cumulus Congestus dengan potensi hujan lokal.',
            'kode_warna': (255, 165, 0)
        },
        'high cumuliform clouds': {
            'nama_indo': 'AWAN ALTOCUMULUS (Berawan)',
            'risiko_penerbangan': 'SEDANG',
            'potensi_hujan': 'Rendah',
            'visibilitas': 'Baik',
            'rekomendasi': 'Indikasi ketidakstabilan atmosfer lapisan atas. Monitor perkembangan awan.',
            'simulasi': 'PROYEKSI T+1: Indikasi instabilitas udara menengah. Berpotensi mengalami penebalan lapisan (Stratiform) yang dapat mengurangi intensitas cahaya matahari.',
            'kode_warna': (255, 255, 0)
        },
        'stratiform clouds': {
            'nama_indo': 'AWAN STRATUS (Mendung)',
            'risiko_penerbangan': 'RENDAH - SEDANG',
            'potensi_hujan': 'Gerimis/Hujan Ringan Merata',
            'visibilitas': 'Sedang (Kabut Tipis)',
            'rekomendasi': 'Perhatikan Icing (Pembekuan) pada sayap pesawat di ketinggian tertentu.',
            'simulasi': 'PROYEKSI T+1: Kecenderungan presipitasi ringan (Drizzle) yang persisten. Waspadai penurunan visibilitas bertahap akibat partikel air halus (Mist/Fog).',
            'kode_warna': (200, 200, 200)
        },
        'stratocumulus clouds': {
            'nama_indo': 'AWAN STRATOCUMULUS',
            'risiko_penerbangan': 'RENDAH (Guncangan Ringan)',
            'potensi_hujan': 'Nihil atau Gerimis Tipis',
            'visibilitas': 'Baik di bawah dasar awan',
            'rekomendasi': 'Awan rendah stabil. Tidak berpotensi cuaca buruk signifikan.',
            'simulasi': 'PROYEKSI T+1: Lapisan awan rendah relatif stabil. Tutupan awan diproyeksikan persisten dengan probabilitas rendah untuk pembentukan badai.',
            'kode_warna': (100, 255, 100)
        }
    }

    # Default jika tidak dikenali
    return knowledge_base.get(cloud, {
        'nama_indo': cloud_type.upper(), 
        'risiko_penerbangan': 'UNKNOWN', 'potensi_hujan': '-', 
        'visibilitas': '-', 'rekomendasi': 'Perlu analisis manual forecaster.',
        'simulasi': 'Data visual tidak mencukupi untuk melakukan proyeksi cuaca.',
        'kode_warna': (128, 128, 128)
    })