from fpdf import FPDF
import datetime
import os

class BMKGReport(FPDF):
    def header(self):
        if os.path.exists('static/images/logo_unpam.png'):
            self.image('static/images/logo_unpam.png', 12, 10, 22) 
        if os.path.exists('static/images/logo_bmkg.png'):
            self.image('static/images/logo_bmkg.png', 176, 10, 22)

        self.set_y(11)
        self.set_font('Arial', 'B', 14)
        self.cell(0, 6, 'BADAN METEOROLOGI, KLIMATOLOGI, DAN GEOFISIKA', 0, 1, 'C')
        self.set_font('Arial', 'B', 12)
        self.cell(0, 6, 'DRAFT UJI COBA MODEL CNN KLASIFIKASI AWAN', 0, 1, 'C')
        
        self.ln(2)
        self.set_font('Arial', '', 9)
        self.cell(0, 4, 'Disusun Oleh:', 0, 1, 'C')
        self.set_font('Arial', 'B', 10)
        self.cell(0, 5, 'YUDHA RANGGA WP (NIM: 241012000151)', 0, 1, 'C')
        self.set_font('Arial', 'I', 9)
        self.cell(0, 5, 'Prodi Magister Teknik Informatika - Universitas Pamulang', 0, 1, 'C')
        
        self.ln(5)
        self.set_line_width(0.5)
        self.line(10, 42, 200, 42)
        self.set_line_width(0.2)
        self.line(10, 43, 200, 43)
        self.ln(8)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 7)
        self.set_text_color(100, 100, 100)
        self.cell(0, 5, f'Halaman {self.page_no()}', 0, 0, 'L')
        self.set_x(-70) 
        self.cell(60, 5, 'Copyright 2025 - UNIVERSITAS PAMULANG', 0, 0, 'R')

# UPDATE: Tambah parameter cloud_coverage & oktas_val
def create_pdf(filename, prediction, confidence, impacts, heatmap_path, cloud_coverage=0, oktas_val=0):
    pdf = BMKGReport()
    pdf.set_auto_page_break(auto=False)
    pdf.add_page()
    
    def add_data_row(label, value, is_bold_value=False):
        pdf.set_font('Arial', '', 9)
        pdf.cell(50, 6, label, 0, 0)
        pdf.cell(5, 6, ':', 0, 0)
        pdf.set_font('Arial', 'B' if is_bold_value else '', 9)
        pdf.cell(0, 6, str(value), 0, 1)

    # I. HASIL
    pdf.set_y(48) 
    pdf.set_font('Arial', 'B', 10)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(0, 7, ' I. DATA HASIL PENGUJIAN (INFERENCE RESULT)', 0, 1, 'L', True)
    pdf.ln(2)
    
    nama_awan_final = impacts.get('nama_indo', prediction).upper()

    add_data_row('ID Sampel Citra', filename)
    add_data_row('Waktu Pengujian', datetime.datetime.now().strftime("%Y-%m-%d %H:%M WIB"))
    add_data_row('Prediksi Kelas Awan', nama_awan_final, is_bold_value=True)
    
    # --- DATA OKTAS (BARU) ---
    oktas_str = f"{oktas_val}/8 OKTAS (Estimasi Tutupan: {cloud_coverage}%)"
    add_data_row('Kuantitas (Oktas)', oktas_str)
    
    add_data_row('Skor Keyakinan', confidence, is_bold_value=True)
    pdf.ln(4)

    # II. VISUALISASI
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(0, 7, ' II. VISUALISASI FITUR & ATENSI (GRAD-CAM)', 0, 1, 'L', True)
    pdf.ln(3)
    img_y = pdf.get_y()
    
    if os.path.exists(f'static/uploads/{filename}'):
        pdf.image(f'static/uploads/{filename}', x=15, y=img_y, w=80, h=55)
    if os.path.exists(heatmap_path):
        pdf.image(heatmap_path, x=115, y=img_y, w=80, h=55)
    
    pdf.set_y(img_y + 55 + 2)
    pdf.set_font('Arial', 'I', 8)
    pdf.cell(95, 5, 'Gbr 1. Input Citra RGB', 0, 0, 'C')
    pdf.cell(95, 5, 'Gbr 2. Peta Atensi Model (Heatmap)', 0, 1, 'C')
    pdf.ln(5)

    # III. ANALISIS
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(0, 7, ' III. ANALISIS DAMPAK & REKOMENDASI (SISTEM PAKAR)', 0, 1, 'L', True)
    pdf.ln(3)
    
    box_start_y = pdf.get_y()
    pdf.set_fill_color(252, 252, 252)
    pdf.rect(10, box_start_y, 190, 48, 'F')
    
    pdf.set_xy(15, box_start_y + 3)
    add_data_row('Tingkat Risiko', impacts.get('risiko_penerbangan', '-'), is_bold_value=True)
    pdf.set_x(15)
    add_data_row('Potensi Cuaca', impacts.get('potensi_hujan', '-'))
    pdf.set_x(15)
    add_data_row('Visibilitas', impacts.get('visibilitas', '-'))
    
    pdf.ln(2)
    pdf.set_x(15)
    pdf.set_font('Arial', 'B', 9)
    pdf.cell(0, 6, 'CATATAN MITIGASI:', 0, 1)
    pdf.set_x(15)
    pdf.set_font('Arial', '', 9)
    pdf.multi_cell(180, 5, impacts.get('rekomendasi', '-'))
    
    pdf.set_y(235) 
    pdf.set_x(120)
    pdf.set_font('Arial', '', 9)
    pdf.cell(70, 5, 'Jakarta, ' + datetime.datetime.now().strftime("%d %B %Y"), 0, 1, 'C')
    pdf.set_x(120)
    pdf.cell(70, 5, 'Peneliti / Mahasiswa,', 0, 1, 'C')
    pdf.ln(18) 
    pdf.set_x(120)
    pdf.set_font('Arial', 'B', 9)
    pdf.cell(70, 5, 'YUDHA RANGGA WP', 0, 1, 'C') 
    pdf.set_x(120)
    pdf.set_font('Arial', '', 8)
    pdf.cell(70, 5, 'NIM. 241012000151', 0, 1, 'C')

    output_filename = f'Laporan_Riset_{filename.split(".")[0]}.pdf'
    output_path = os.path.join('static/reports', output_filename)
    os.makedirs('static/reports', exist_ok=True)
    pdf.output(output_path)
    return output_path