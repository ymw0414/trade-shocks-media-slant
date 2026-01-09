import pandas as pd
import os
from pathlib import Path

# 경로 설정
BASE_DIR = Path(os.environ["SHIFTING_SLANT_DIR"])
FILE_PATH = BASE_DIR / "data" / "intermediate" / "newspapers" / "yearly" / "newspapers_1992.parquet"

# 파일 열어서 컬럼명만 확인
try:
    df = pd.read_parquet(FILE_PATH)
    print("\n" + "="*30)
    print(f"📂 파일 내부의 실제 컬럼 목록:")
    print(df.columns.tolist())
    print("="*30 + "\n")
    
    if "paper" in df.columns:
        print("✅ 'paper'가 맞습니다! 작성하신 코드를 그대로 쓰셔도 됩니다.")
    elif "paper_name" in df.columns:
        print("❌ 'paper'가 아니라 'paper_name'으로 고치셔야 합니다.")
    else:
        print("⚠️ 신문사 이름으로 추정되는 다른 컬럼을 찾아보세요.")

except Exception as e:
    print(f"에러 발생: {e}")