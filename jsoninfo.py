import pandas as pd
import os

def load_dataset(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # 엑셀 불러오기
    df = pd.read_excel(file_path)

    # 필요하다면 기존 코드에서 했던 것처럼 컬럼명 번역/변경
    df.rename(columns={
        "품목일련번호": "product_serial_number",
        "제품명": "product_name",
        "업체명": "company_name",
        "주성분": "active_ingredient",
        # 그 외에 필요한 부분은 알아서...
    }, inplace=True)

    return df

if __name__ == '__main__':
    file_path = "./edruginfo.xlsx"
    df_full = load_dataset(file_path)

    # 1) 네 가지 컬럼만 추출 (원본 df에서 필요 없는 컬럼 제외)
    columns_needed = [
        "product_serial_number",
        "product_name",
        "company_name",
        "active_ingredient"
    ]
    df_subset = df_full[columns_needed].copy()

    # 2) 주성분을 쉼표 기준으로 분할하여 리스트(배열) 형태로 변환
    #   - 빈 문자열/NaN 대비 .fillna('')
    #   - strip()으로 앞뒤 공백 제거
    #   - 만약 "아세트아미노펜, , 카페인무수물" 처럼 중간에 공백만 있으면 제외
    df_subset['active_ingredient_array'] = (
        df_subset['active_ingredient']
        .fillna('')
        .apply(lambda x: [part.strip() for part in x.split(',') if part.strip()])
    )

    # 원본 active_ingredient 컬럼을 굳이 유지할 필요가 없다면 제거
    df_subset.drop(columns=['active_ingredient'], inplace=True)

    # 3) JSON 파일로 저장
    #   - orient='records' : 각 행을 하나의 JSON 객체로, 배열 형태로 저장
    #   - force_ascii=False : 한글을 유니코드 이스케이프(\uAC00)로 바꾸지 않음
    output_file = "edruginfo_extracted.json"
    df_subset.to_json(output_file, orient='records', force_ascii=False)

    print(f"완료: {output_file} 파일에 저장되었습니다.")