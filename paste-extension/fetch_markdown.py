import os
import sys
import requests
from bs4 import BeautifulSoup
from markdownify import markdownify
import re
from datetime import datetime


def add_metadata(markdown_content):
    """
    Markdown 콘텐츠 맨 앞에 `{{< keras/original checkedAt="현재 날짜" >}}`를 추가합니다.
    """
    # 현재 날짜를 'YYYY-MM-DD' 형식으로 가져오기
    current_date = datetime.now().strftime("%Y-%m-%d")
    metadata = f'{{{{< keras/original checkedAt="{current_date}" >}}}}\n\n'
    return metadata + markdown_content


def get_url_from_file_path(file_path):
    """
    현재 파일 경로를 받아서 해당 파일에 대응하는 URL 생성.
    """
    base_url = "https://keras.io"
    relative_path = file_path.split(
        "/content/english/docs")[-1].replace("_index.md", "")
    return f"{base_url}{relative_path}"


def save_to_file(content, file_path):
    """
    내용을 지정된 파일에 저장.
    """
    try:
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(content)
        print(f"Markdown content saved to {file_path}")
    except Exception as e:
        print(f"Error saving to file {file_path}: {e}", file=sys.stderr)


def clean_markdown(markdown_content):
    """
    Markdown 내용 후처리:
    - 코드 블록(````) 언어 지정.
    """
    lines = markdown_content.split("\n")
    cleaned_lines = []
    buffer = []
    inside_code_block = False

    for i, line in enumerate(lines):
        if line.startswith("```"):  # 코드 블록 시작/종료
            if inside_code_block:
                # 코드 블록 종료: 마지막 빈 줄 제거
                buffer = [l for l in buffer if l.strip()]  # 빈 줄 제거
                cleaned_lines.append("\n".join(buffer))
                buffer = []
                cleaned_lines.append("```")  # 코드 블록 종료
            else:
                # 코드 블록 시작: 언어 설정
                code_language = "python"  # 기본값
                if i + 1 < len(lines):  # 다음 줄이 존재하는지 확인
                    next_line = lines[i + 1].strip()
                    if next_line.startswith("$") or next_line.startswith(">>>"):
                        code_language = "console"
                cleaned_lines.append(f"```{code_language}")
            inside_code_block = not inside_code_block
        elif inside_code_block:
            # 코드 블록 내의 줄 추가
            buffer.append(line)
        else:
            # 코드 블록 외부: 그대로 추가
            cleaned_lines.append(line)

    if buffer:
        cleaned_lines.append("\n".join(buffer))  # 마지막 줄 처리

    return "\n".join(cleaned_lines)


def transform_source_links(markdown_content):
    """
    `[[source]](URL)` 형식을 `{{< keras/source link="URL" >}}`로 변환.
    """
    # 정규식 변환
    transformed_content = re.sub(
        r"\[\[source\]\]\((.+?)\)", r'{{< keras/source link="\1" >}}', markdown_content
    )
    return transformed_content


def change_link(markdown_content):
    # 정규식 변환
    transformed_content = re.sub(
        r"\((\/[^\)]+)\)", r'({{< relref "/docs\1" >}})', markdown_content
    )
    return transformed_content


def change_keras_link(markdown_content):
    # 정규식 변환
    transformed_content = re.sub(
        r"\[([^\]]+)\]\(https:\/\/keras\.io(\/[^\)]+)\)", r'[\1]({{< relref "/docs\2" >}})', markdown_content
    )
    return transformed_content


def remove_h1_and_hr(markdown_content):
    """
    Markdown 내용에서 h1 제목(`# title`)과 줄긋기(`---`)를 제거하되, 코드 블록 내부는 무시합니다.
    """
    lines = markdown_content.split("\n")
    cleaned_lines = []
    inside_code_block = False

    for line in lines:
        if line.startswith("```"):  # 코드 블록의 시작/종료
            inside_code_block = not inside_code_block
            cleaned_lines.append(line)
            continue

        if inside_code_block:
            # 코드 블록 내부는 그대로 유지
            cleaned_lines.append(line)
            continue

        # h1 제목 제거
        if line.startswith("# ") and len(line.strip()) > 1:
            continue

        # --- 줄긋기 제거
        if line.strip() == "---":
            continue

        # 나머지 줄 추가
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def fetch_and_clean_content(url):
    """
    URL에서 <div class='k-content'> 내용을 가져와 Markdown으로 변환 후 클린업.
    """
    try:
        # 웹페이지 가져오기
        response = requests.get(url, timeout=10, verify=False)
        response.raise_for_status()
        html_content = response.text

        # HTML 파싱 및 k-content 영역 추출
        soup = BeautifulSoup(html_content, "html.parser")
        k_content_div = soup.find("div", class_="k-content")
        if not k_content_div:
            print("Error: <div class='k-content'> not found on the page.",
                  file=sys.stderr)
            return None

        # HTML -> Markdown 변환
        markdown_content = markdownify(str(k_content_div), heading_style="ATX")

        # Markdown 후처리
        markdown_content = clean_markdown(markdown_content)

        # `[[source]]` 링크 변환
        markdown_content = transform_source_links(markdown_content)

        # rel 링크 변환
        markdown_content = change_link(markdown_content)
        markdown_content = change_keras_link(markdown_content)

        # h1 제목과 줄긋기 제거
        markdown_content = remove_h1_and_hr(markdown_content)

        # 메타데이터 추가
        markdown_content = add_metadata(markdown_content)

        return markdown_content

    except Exception as e:
        print(f"Error fetching or parsing URL: {e}", file=sys.stderr)
        return None


def main():
    """
    파일 경로를 기반으로 URL을 생성하고, Markdown으로 변환한 후 출력.
    """
    file_path = sys.argv[1] if len(sys.argv) > 1 else None
    if file_path:
        url = get_url_from_file_path(file_path)
        print(f"Generated URL: {url}", file=sys.stderr)

        # URL에서 Markdown 가져오기
        markdown_content = fetch_and_clean_content(url)
        if markdown_content:
            # 결과 저장 (임시 파일)
            temp_file = "/tmp/output.md"  # 임시 파일 경로
            save_to_file(markdown_content, temp_file)
            print(markdown_content)  # 결과 출력
        else:
            print("Failed to fetch content.", file=sys.stderr)
    else:
        print("Error: No file path provided.", file=sys.stderr)


if __name__ == "__main__":
    main()
