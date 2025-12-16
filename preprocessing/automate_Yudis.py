import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(input_path, output_path):
    df = pd.read_csv(input_path)
    df_prep = df.copy()

    # drop id
    df_prep.drop(columns=['student_id'], inplace=True, errors='ignore')

    # drop duplicates
    df_prep = df_prep.drop_duplicates()

    # normalize column names
    df_prep.columns = (
        df_prep.columns
        .str.strip()
        .str.lower()
        .str.replace(' ', '_')
    )

    # categorical encoding
    categorical_cols = [
        'gender',
        'course',
        'internet_access',
        'study_method',
        'exam_difficulty',
        'sleep_quality',
        'facility_rating'
    ]
    categorical_cols = [c for c in categorical_cols if c in df_prep.columns]

    df_prep = pd.get_dummies(df_prep, columns=categorical_cols, drop_first=True)

    # numeric scaling
    numeric_cols = [
        'age',
        'study_hours',
        'class_attendance',
        'sleep_hours'
    ]
    numeric_cols = [c for c in numeric_cols if c in df_prep.columns]

    scaler = StandardScaler()
    df_prep[numeric_cols] = scaler.fit_transform(df_prep[numeric_cols])

    df_prep.to_csv(output_path, index=False)
    print("Preprocessing selesai")


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    input_path = os.path.join(
        BASE_DIR, "..", "exam_score_raw", "exam_score_raw.csv"
    )

    output_path = os.path.join(
        BASE_DIR, "exam_score_preprocessed.csv"
    )

    preprocess_data(input_path, output_path)

