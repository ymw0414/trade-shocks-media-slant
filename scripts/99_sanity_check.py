"""
FILE: debug_akron_1986.py
DESCRIPTION:
    - 문제가 된 'Akron Beacon Journal (OH)'의 1986년 데이터를
      Parquet 파일에서 직접 읽어서 화면에 출력합니다.
    - slant 컬럼이 NaN인지, 0인지, 아니면 정상 값인지 확인합니다.
"""

import os
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_DIR = Path(os.environ.get("SHIFTING_SLANT_DIR", r"C:\Users\ymw04\Dropbox\shifting_slant"))
SLANT_DIR = BASE_DIR / "data" / "processed" / "newspapers" / "slant"

# 문제가 된 타겟 설정
TARGET_PAPER = "Akron Beacon Journal (OH)"
TARGET_YEAR = 1986


def main():
    print(f">>> Scanning files for {TARGET_PAPER} in {TARGET_YEAR}...")

    # Parquet 파일 찾기 (보통 연도별이나 의회 회기별로 나뉘어 있을 수 있음)
    files = sorted(list(SLANT_DIR.glob("news_slant_congress_*.parquet")))

    found_data = False

    for f in files:
        # 일단 가볍게 로드해서 연도와 신문사 확인
        try:
            # 필요한 컬럼만 로드
            df = pd.read_parquet(f, columns=['paper', 'date', 'slant', 'used_terms'])

            # 연도 추출
            df['year'] = pd.to_datetime(df['date'], errors='coerce').dt.year

            # 필터링
            mask = (df['paper'] == TARGET_PAPER) & (df['year'] == TARGET_YEAR)
            subset = df[mask]

            if not subset.empty:
                found_data = True
                print(f"\n[FOUND] Data in file: {f.name}")
                print("-" * 50)
                print(f"Total Articles: {len(subset)}")
                print(f"Slant 'NaN' count: {subset['slant'].isna().sum()}")
                print(f"Slant '0.0' count: {(subset['slant'] == 0.0).sum()}")
                print("-" * 50)
                print("First 5 rows of raw data:")
                print(subset[['date', 'slant', 'used_terms']].head())
                print("-" * 50)

                # 실제 계산 테스트
                subset_valid = subset.dropna(subset=['slant', 'used_terms'])
                weighted_sum = (subset_valid['slant'] * subset_valid['used_terms']).sum()
                total_terms = subset_valid['used_terms'].sum()

                print(f"DEBUG CALCULATION:")
                print(f"Sum(Slant * Terms) = {weighted_sum}")
                print(f"Sum(Terms)         = {total_terms}")
                if total_terms > 0:
                    print(f"Weighted Mean      = {weighted_sum / total_terms}")
                else:
                    print("Weighted Mean      = Undefined (Total Terms is 0)")
                print("=" * 50)

        except Exception as e:
            continue

    if not found_data:
        print(f"Error: Could not find any rows for {TARGET_PAPER} in {TARGET_YEAR}.")


if __name__ == "__main__":
    main()