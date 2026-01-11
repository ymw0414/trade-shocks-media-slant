import os
import re
import pandas as pd
import requests
import io

# ---------------------------------------------------------------------------
# 1. 설정
# ---------------------------------------------------------------------------
BASE_DIR = r"C:\Users\ymw04\Dropbox\shifting_slant\data\raw\crosswalks"
INPUT_TXT = os.path.join(BASE_DIR, "dma_2016_raw.txt")
OUTPUT_DTA = os.path.join(BASE_DIR, "county_dma.dta")

# FIPS 코드 매칭용 마스터 파일 (GitHub에서 가져옴 - 매우 안정적)
FIPS_URL = "https://raw.githubusercontent.com/kjhealy/fips-codes/master/state_and_county_fips_master.csv"


def parse_dma_line(line):
    """ 'DMA Name -- County List' 형태의 줄을 파싱 """
    if "--" not in line: return []

    parts = line.split("--")
    dma_raw = parts[0].strip()
    counties_part = parts[1].strip()

    # DMA 이름에서 주(State) 제거 (예: "ABILENE-SWEETWATER, TX" -> "ABILENE-SWEETWATER")
    dma_name = dma_raw.split(",")[0].strip()

    # 주별로 구분된 카운티 리스트 처리 (세미콜론 기준)
    # 예: "Brown, Callahan Counties, TX; Other County, OK."
    records = []
    groups = re.split(r';', counties_part)

    for group in groups:
        group = group.strip()
        if not group: continue

        # 마지막 두 글자를 주(State)로 인식
        match = re.search(r'([A-Z]{2})\.?$', group)
        if match:
            state = match.group(1)
            # 주 약어 및 "Counties", "Parish" 등 불용어 제거
            text_clean = group[:match.start()].strip()
            text_clean = re.sub(r'\b(Counties|Parishes|County|Parish|and)\b', '', text_clean, flags=re.IGNORECASE)
            text_clean = text_clean.rstrip(',').strip()
        else:
            continue  # 주 정보 없으면 스킵

        # 쉼표로 카운티 분리
        counties = [c.strip() for c in text_clean.split(',')]

        for county in counties:
            if county:
                records.append({
                    'dma_name': dma_name,
                    'state_abbr': state,
                    'county_name': county
                })
    return records


def main():
    print("-" * 60)
    print(">>> [Step 1] 텍스트 파일 파싱 시작...")

    if not os.path.exists(INPUT_TXT):
        print(f"❌ 에러: 파일을 찾을 수 없습니다: {INPUT_TXT}")
        print("   메모장에 텍스트를 붙여넣고 해당 위치에 저장해주세요.")
        return

    # 1. 텍스트 파싱
    parsed_data = []
    with open(INPUT_TXT, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parsed_data.extend(parse_dma_line(line))

    df_dma = pd.DataFrame(parsed_data)
    print(f"   -> {len(df_dma)}개 카운티-DMA 연결 정보 추출됨.")

    # 2. FIPS 코드 매칭 (GitHub에서 다운로드)
    print(">>> [Step 2] FIPS 코드 매칭 중...")
    try:
        s = requests.get(FIPS_URL).content
        df_fips = pd.read_csv(io.StringIO(s.decode('utf-8')))

        # 이름 정리 (매칭률 높이기 위해)
        # fips 데이터: "Autauga County" -> "Autauga"
        df_fips['county_clean'] = df_fips['name'].str.replace(' County', '').str.replace(' Parish', '').str.replace(
            ' Borough', '')
        df_fips['state_abbr'] = df_fips['state']  # fips 파일의 컬럼명 확인 필요

        # 병합 (주 + 카운티 이름 기준)
        # 대소문자 통일 등 전처리
        df_dma['key_name'] = df_dma['county_name'].str.upper()
        df_dma['key_state'] = df_dma['state_abbr'].str.upper()

        df_fips['key_name'] = df_fips['county_clean'].str.upper()
        df_fips['key_state'] = df_fips['state'].str.upper()  # fips 파일엔 'state'가 약어(AL, TX)임

        merged = pd.merge(df_dma, df_fips, left_on=['key_name', 'key_state'], right_on=['key_name', 'key_state'],
                          how='inner')

        # 3. 최종 정리
        final_df = merged[['fips', 'dma_name']].copy()
        final_df.rename(columns={'fips': 'county'}, inplace=True)

        # DMA 이름을 숫자로 변환 (Stata 호환용)
        final_df['dma_code'] = final_df['dma_name'].astype('category').cat.codes + 1

        # Stata 파일로 저장
        final_df.to_stata(OUTPUT_DTA, write_index=False, version=118)

        print("-" * 60)
        print("✅ [성공] 변환 완료!")
        print(f"📂 생성된 파일: {OUTPUT_DTA}")
        print(f"📊 매칭된 카운티 수: {len(final_df)}")
        print("-" * 60)

    except Exception as e:
        print(f"❌ 에러 발생: {e}")


if __name__ == "__main__":
    main()