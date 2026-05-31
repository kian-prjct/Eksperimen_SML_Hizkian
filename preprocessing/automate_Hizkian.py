"""
automate_Nama-Siswa.py
======================
Script otomatisasi preprocessing dataset Wine Quality.
Mengkonversi seluruh tahapan eksperimen dari notebook menjadi
pipeline yang dapat dijalankan secara otomatis.

Usage:
    python automate_Nama-Siswa.py

Output:
    - winequality_preprocessing/winequality_train.csv
    - winequality_preprocessing/winequality_test.csv
    - winequality_preprocessing/winequality_clean.csv
"""

import pandas as pd
import numpy as np
import os
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# ── Setup Logging ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# FUNGSI-FUNGSI PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def load_data() -> pd.DataFrame:
    """
    Memuat dataset Wine Quality dari UCI ML Repository.
    Menggabungkan red wine dan white wine menjadi satu dataframe.

    Returns:
        pd.DataFrame: Dataset gabungan red + white wine.
    """
    logger.info("Memuat dataset dari UCI ML Repository...")

    url_red = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases"
        "/wine-quality/winequality-red.csv"
    )
    url_white = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases"
        "/wine-quality/winequality-white.csv"
    )

    df_red = pd.read_csv(url_red, sep=';')
    df_white = pd.read_csv(url_white, sep=';')

    df_red['wine_type'] = 'red'
    df_white['wine_type'] = 'white'

    df = pd.concat([df_red, df_white], ignore_index=True)

    # Simpan raw dataset
    os.makedirs('../winequality_raw', exist_ok=True)
    df.to_csv('../winequality_raw/winequality_raw.csv', index=False)

    logger.info(f"Dataset berhasil dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")
    return df


def create_binary_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Membuat kolom target biner dari kolom 'quality'.
    quality >= 6 → 1 (good), quality < 6 → 0 (bad)

    Args:
        df: DataFrame input.

    Returns:
        pd.DataFrame: DataFrame dengan kolom 'quality_label'.
    """
    df = df.copy()
    df['quality_label'] = (df['quality'] >= 6).astype(int)
    logger.info(
        f"Target biner dibuat — distribusi: "
        f"{df['quality_label'].value_counts().to_dict()}"
    )
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menghapus baris duplikat dari dataframe.

    Args:
        df: DataFrame input.

    Returns:
        pd.DataFrame: DataFrame tanpa duplikat.
    """
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    logger.info(f"Duplikasi dihapus: {removed} baris | Sisa: {len(df)} baris")
    return df


def handle_missing_values(df: pd.DataFrame, numerical_cols: list) -> pd.DataFrame:
    """
    Mengisi missing values dengan median untuk kolom numerik.

    Args:
        df: DataFrame input.
        numerical_cols: List nama kolom numerik.

    Returns:
        pd.DataFrame: DataFrame tanpa missing values.
    """
    df = df.copy()
    total_missing = df[numerical_cols].isnull().sum().sum()

    if total_missing == 0:
        logger.info("Tidak ada missing values ditemukan.")
        return df

    for col in numerical_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            logger.info(f"  Kolom '{col}' diisi median = {median_val:.4f}")

    logger.info(f"Missing values selesai ditangani.")
    return df


def remove_outliers_iqr(
    df: pd.DataFrame,
    cols: list,
    multiplier: float = 1.5
) -> pd.DataFrame:
    """
    Menghapus outlier menggunakan metode IQR (Interquartile Range).

    Args:
        df: DataFrame input.
        cols: List kolom yang akan di-filter outlier-nya.
        multiplier: Faktor pengali IQR (default: 1.5).

    Returns:
        pd.DataFrame: DataFrame tanpa outlier.
    """
    df_out = df.copy()
    before_total = len(df_out)

    for col in cols:
        Q1 = df_out[col].quantile(0.25)
        Q3 = df_out[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - multiplier * IQR
        upper = Q3 + multiplier * IQR
        before = len(df_out)
        df_out = df_out[(df_out[col] >= lower) & (df_out[col] <= upper)]
        removed = before - len(df_out)
        if removed > 0:
            logger.info(f"  Outlier '{col}': {removed} baris dihapus")

    total_removed = before_total - len(df_out)
    logger.info(
        f"Total outlier dihapus: {total_removed} | "
        f"Sisa: {len(df_out)} baris ({len(df_out)/before_total*100:.1f}%)"
    )
    return df_out


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Melakukan label encoding pada kolom 'wine_type'.
    red=0, white=1

    Args:
        df: DataFrame input.

    Returns:
        pd.DataFrame: DataFrame dengan kolom 'wine_type_encoded'.
    """
    df = df.copy()
    le = LabelEncoder()
    df['wine_type_encoded'] = le.fit_transform(df['wine_type'])
    mapping = dict(zip(le.classes_, le.transform(le.classes_).tolist()))
    logger.info(f"Encoding 'wine_type': {mapping}")
    return df


def scale_features(
    df: pd.DataFrame,
    feature_cols: list
) -> pd.DataFrame:
    """
    Melakukan standarisasi (StandardScaler) pada fitur numerik.

    Args:
        df: DataFrame input.
        feature_cols: List kolom fitur yang akan di-scale.

    Returns:
        pd.DataFrame: DataFrame dengan fitur yang sudah di-standarisasi.
    """
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(df[feature_cols])
    df_scaled = pd.DataFrame(scaled_values, columns=feature_cols, index=df.index)
    logger.info(f"Standarisasi selesai untuk {len(feature_cols)} fitur.")
    return df_scaled


def split_and_save(
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> None:
    """
    Membagi data menjadi train-test split dan menyimpan ke CSV.

    Args:
        X: DataFrame fitur.
        y: Series target.
        output_dir: Direktori output.
        test_size: Proporsi data test (default: 0.2).
        random_state: Seed untuk reproduktifitas (default: 42).
    """
    os.makedirs(output_dir, exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Gabungkan fitur dan target
    train_df = pd.concat(
        [X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1
    )
    test_df = pd.concat(
        [X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1
    )
    full_df = pd.concat(
        [X.reset_index(drop=True), y.reset_index(drop=True)], axis=1
    )

    train_path = os.path.join(output_dir, 'winequality_train.csv')
    test_path  = os.path.join(output_dir, 'winequality_test.csv')
    clean_path = os.path.join(output_dir, 'winequality_clean.csv')

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    full_df.to_csv(clean_path, index=False)

    logger.info(f"Data train disimpan: {train_path} ({len(train_df)} baris)")
    logger.info(f"Data test disimpan:  {test_path} ({len(test_df)} baris)")
    logger.info(f"Data clean disimpan: {clean_path} ({len(full_df)} baris)")


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE UTAMA
# ══════════════════════════════════════════════════════════════════════════════

def run_preprocessing_pipeline() -> None:
    """
    Menjalankan seluruh pipeline preprocessing secara otomatis.
    Tahapan:
        1. Load data
        2. Buat target biner
        3. Hapus duplikasi
        4. Handle missing values
        5. Handle outlier (IQR)
        6. Encoding kategorikal
        7. Standarisasi fitur
        8. Split & simpan
    """
    logger.info("=" * 60)
    logger.info("MEMULAI PIPELINE PREPROCESSING WINE QUALITY")
    logger.info("=" * 60)

    # ── Step 1: Load data ──────────────────────────────────────
    df = load_data()

    # ── Step 2: Buat target biner ──────────────────────────────
    df = create_binary_target(df)

    # ── Step 3: Hapus duplikasi ────────────────────────────────
    df = remove_duplicates(df)

    # ── Step 4: Handle missing values ─────────────────────────
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_cols = [c for c in numerical_cols if c not in ['quality', 'quality_label']]
    df = handle_missing_values(df, numerical_cols)

    # ── Step 5: Handle outlier ─────────────────────────────────
    df = remove_outliers_iqr(df, numerical_cols)

    # ── Step 6: Encoding kategorikal ───────────────────────────
    df = encode_categorical(df)

    # ── Step 7: Standarisasi fitur ─────────────────────────────
    feature_cols = numerical_cols + ['wine_type_encoded']
    X_scaled = scale_features(df, feature_cols)
    y = df['quality_label'].reset_index(drop=True)

    # ── Step 8: Split & simpan ─────────────────────────────────
    split_and_save(X_scaled, y, output_dir='winequality_preprocessing')

    logger.info("=" * 60)
    logger.info("✅ PIPELINE PREPROCESSING SELESAI — Data siap untuk modeling!")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_preprocessing_pipeline()
