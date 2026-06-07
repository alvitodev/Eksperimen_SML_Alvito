import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def run_preprocessing(input_path, output_dir):
    print("Memulai proses otomatisasi preprocessing...")
    
    # 1. Memuat Dataset
    df = pd.read_csv(input_path)
    
    # 2. Membersihkan Data
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)
        
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    
    # 3. Encoding Kategorikal
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    df = pd.get_dummies(df, drop_first=True)
    
    # 4. Memisahkan Fitur dan Target
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # 5. Membagi Data (Train & Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 6. Standarisasi (Scaling)
    scaler = StandardScaler()
    numerik_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    X_train.loc[:, numerik_cols] = scaler.fit_transform(X_train[numerik_cols])
    X_test.loc[:, numerik_cols] = scaler.transform(X_test[numerik_cols])
    
    # 7. Menyimpan Hasil Preprocessing agar bisa dipakai oleh Model nanti
    os.makedirs(output_dir, exist_ok=True)
    
    # Menggabungkan kembali X dan Y untuk disimpan sebagai CSV terpisah
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    
    train_data.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    test_data.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    
    print(f"Data berhasil diproses dan disimpan di folder: {output_dir}")

if __name__ == "__main__":
    # Mengatur path agar dinamis, membaca dari folder telco_raw dan menyimpan di telco_preprocessing
    RAW_DATA_PATH = os.path.join("..", "telco_raw", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    OUTPUT_DIR = os.path.join("..", "telco_preprocessing")
    
    try:
        run_preprocessing(RAW_DATA_PATH, OUTPUT_DIR)
    except FileNotFoundError:
        print(f"Error: File tidak ditemukan di {RAW_DATA_PATH}. Pastikan dijalankan dari dalam folder 'preprocessing'.")