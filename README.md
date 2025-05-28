# YouTube Comments Analysis Framework
## 유튜브 댓글 종합 분석 프레임워크

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![uv](https://img.shields.io/badge/uv-compatible-green.svg)](https://github.com/astral-sh/uv)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)]()
[![Korean](https://img.shields.io/badge/Language-Korean-orange.svg)]()
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey.svg)]()
[![AI Generated](https://img.shields.io/badge/AI%20Generated-Code-purple.svg)]()

> 한국어 YouTube 댓글에 대한 종합적인 감성 분석, 토픽 모델링, 시간적 패턴 분석을 수행하는 고급 자연어처리 프레임워크

## 📋 목차

- [프로젝트 개요](#-프로젝트-개요)
- [주요 기능](#-주요-기능)
- [기술 스택](#-기술-스택)
- [빠른 시작](#-빠른-시작)
- [설치 및 실행](#-설치-및-실행)
- [데이터 준비](#-데이터-준비)
- [분석 방법론](#-분석-방법론)
- [사용법](#-사용법)
- [결과물](#-결과물)
- [프로젝트 구조](#-프로젝트-구조)

## 🎯 프로젝트 개요

본 프로젝트는 YouTube 댓글 데이터를 활용하여 **여론 동향**, **감성 패턴**, **토픽 진화**를 종합적으로 분석하는 고급 자연어처리 프레임워크입니다. 특히 한국어 텍스트에 최적화되어 있으며, 시간적 변화를 고려한 동적 분석을 제공합니다.

### 🔬 연구 목적
- **여론 흐름 분석**: 시간에 따른 대중 의견의 변화 패턴 파악
- **감성 동향 추적**: 긍정/부정 감성 및 6가지 세부 감정의 시계열 변화
- **토픽 진화 모니터링**: 주요 담론의 등장, 발전, 소멸 과정 추적
- **이상치 탐지**: 급격한 여론 변화나 특이 패턴 식별

## ✨ 주요 기능

### 🔍 기본 분석
- **적응적 시간 분할**: 데이터 분포에 따른 동적 시간 구간 설정
- **다층 감성 분석**: 이진 분류 + 6감정 분류
- **하이브리드 토픽 모델링**: BERTopic + LDA 결합
- **품질 기반 필터링**: 다차원 댓글 품질 평가

### 🚀 고급 분석
- **시간 기반 여론 흐름**: 감성 변화의 시계열 패턴 분석
- **변곡점 탐지**: PELT 알고리즘 기반 급변점 식별
- **문맥 임베딩 클러스터링**: SBERT + HDBSCAN
- **키워드 공출현 네트워크**: 의미 관계망 구축

### 📊 시각화 및 보고서
- **인터랙티브 대시보드**: Plotly 기반 동적 시각화
- **종합 분석 보고서**: HTML/Markdown/JSON 형식
- **시간별 워드클라우드**: 기간별 키워드 변화 시각화

## 🛠 기술 스택

### 핵심 라이브러리
```python
# 자연어처리
transformers>=4.30.0      # Hugging Face Transformers
sentence-transformers>=2.2.0  # 문장 임베딩
bertopic>=0.15.0         # BERTopic 토픽 모델링
gensim>=4.3.0           # LDA 토픽 모델링
konlpy>=0.6.0           # 한국어 형태소 분석

# 머신러닝
torch>=2.0.0            # PyTorch
scikit-learn>=1.3.0     # 전통적 ML 알고리즘
umap-learn>=0.5.3       # 차원 축소
hdbscan>=0.8.29         # 밀도 기반 클러스터링

# 데이터 처리
pandas>=2.0.0           # 데이터 조작
numpy>=1.24.0           # 수치 연산
ruptures>=1.1.7         # 변곡점 탐지

# 시각화
matplotlib>=3.7.0       # 기본 시각화
seaborn>=0.12.0         # 통계 시각화
plotly>=5.14.0          # 인터랙티브 시각화
wordcloud>=1.9.0        # 워드클라우드
networkx>=3.1.0         # 네트워크 그래프
```

### 지원 환경
- **Python**: 3.8+ (3.9+ 권장)
- **패키지 관리자**: uv (권장), pip, conda
- **GPU**: CUDA 지원 (선택사항, 성능 향상)
- **OS**: Windows, macOS, Linux
- **메모리**: 최소 8GB RAM 권장 (16GB+ 최적)

## ⚡ 빠른 시작

### 1분 만에 시작하기 (uv 사용)

```bash
# 1. 저장소 클론
git clone https://github.com/your-username/youtube-comments-analysis.git
cd youtube-comments-analysis

# 2. uv로 가상환경 생성 및 의존성 설치
uv venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

uv pip install -r requirements.txt

# 3. 샘플 데이터로 즉시 실행
uv run python main.py --basic-only
```

**🎉 완료!** 샘플 데이터로 분석이 시작됩니다.

## 🚀 설치 및 실행

### 방법 1: uv 사용 (권장)

[uv](https://github.com/astral-sh/uv)는 빠르고 안정적인 Python 패키지 관리자입니다.

```bash
# uv 설치 (아직 설치하지 않은 경우)
curl -LsSf https://astral.sh/uv/install.sh | sh  # Linux/macOS
# 또는 Windows: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 프로젝트 설정
git clone https://github.com/your-username/youtube-comments-analysis.git
cd youtube-comments-analysis

# 가상환경 생성 및 활성화
uv venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# 의존성 설치
uv pip install -r requirements.txt
```

### 방법 2: conda 사용

```bash
# conda 환경 생성
conda create -n youtube-analysis python=3.9
conda activate youtube-analysis

# 의존성 설치
pip install -r requirements.txt
```

### 방법 3: pip 사용

```bash
# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# 의존성 설치
pip install -r requirements.txt
```

### 한국어 형태소 분석기 설치

```bash
# Ubuntu/Debian
sudo apt-get install mecab libmecab-dev mecab-ipadic-utf8

# macOS
brew install mecab mecab-ko mecab-ko-dic

# Windows
# https://github.com/Pusnow/mecab-ko-msvc 참조
```

## 📊 데이터 준비

### 즉시 사용 가능한 샘플 데이터

프로젝트에는 즉시 테스트할 수 있는 샘플 데이터가 포함되어 있습니다:

```
data/
├── sample_youtube_comments.csv  # 20개 샘플 댓글 (한국어)
├── example_data_format.csv      # 최소 형식 예시
└── README.md                    # 데이터 가이드
```

### 자신의 데이터 사용하기

#### 1. 필수 데이터 형식

CSV 파일에 다음 컬럼이 포함되어야 합니다:

| 컬럼명 | 타입 | 필수여부 | 설명 | 예시 |
|--------|------|----------|------|------|
| `text` | string | ✅ 필수 | 댓글 내용 | "정말 충격적인 뉴스네요..." |
| `date` | datetime | ✅ 필수 | 댓글 작성일 | "2023-01-15 14:30:25" |
| `upvotes` | int | 선택 | 추천수 | 15 |
| `downvotes` | int | 선택 | 비추천수 | 2 |
| `author` | string | 선택 | 작성자 | "user123" |
| `video_title` | string | 선택 | 영상 제목 | "뉴스 제목" |

#### 2. 데이터 배치 및 설정

```bash
# 1. 데이터 파일을 data/ 디렉토리에 배치
cp your_data.csv data/

# 2. config.py에서 파일 경로 수정
# DATA_FILES = {
#     'youtube_comments': os.path.join(DATA_DIR, 'your_data.csv'),
# }
```

#### 3. 데이터 보안 및 프라이버시

- **실제 데이터는 Git에 업로드되지 않습니다** (`.gitignore`로 보호)
- `sample_*` 또는 `example_*`로 시작하는 파일만 추적됩니다
- 개인정보가 포함된 데이터는 익명화 후 사용하세요

### 데이터 품질 가이드

최적의 분석 결과를 위해:
- **최소 1,000개 이상의 댓글** 권장
- **여러 시간대에 걸친 데이터** 포함
- **UTF-8 인코딩** 사용
- **일관된 날짜 형식** 유지
- **명백한 스팸/봇 댓글** 사전 제거

## 🔬 분석 방법론

### 1. 적응적 시간 분할 (Adaptive Time Segmentation)

데이터 분포에 따라 동적으로 시간 구간을 설정합니다:

```python
# 월별 → 주별 → 일별 세분화
if comment_ratio > 0.25:  # 25% 이상의 댓글이 특정 기간에 집중
    subdivide_period()  # 더 세밀한 분석 적용
```

### 2. 다층 감성 분석

#### 이진 감성 분류
- **모델**: `monologg/koelectra-small-finetuned-nsmc`
- **출력**: 긍정/부정 + 신뢰도 점수

#### 6감정 분류
- **모델**: `hun3359/mdistilbertV3.1-sentiment`
- **감정**: 분노, 슬픔, 불안, 상처, 기쁨, 당황

### 3. 하이브리드 토픽 모델링

#### BERTopic
- **임베딩**: `jhgan/ko-sroberta-multitask` (한국어 SBERT)
- **클러스터링**: HDBSCAN
- **차원축소**: UMAP

#### LDA (Latent Dirichlet Allocation)
- **형태소 분석**: Mecab/Okt
- **최적 토픽 수**: 자동 계산
- **한국어 불용어**: 500+ 단어 사전 내장

### 4. 고급 분석 기법

#### 변곡점 탐지 (PELT)
```python
# 급격한 여론 변화 시점 탐지
changepoints = detect_pelt_changepoints(sentiment_series)
```

#### 문맥 임베딩 클러스터링
```python
# SBERT + HDBSCAN으로 의미적 유사 댓글 그룹화
clusters = analyze_contextual_embeddings(comments)
```

## 💻 사용법

### 기본 분석 실행

```bash
# 샘플 데이터로 기본 분석
uv run python main.py --basic-only

# 자신의 데이터로 기본 분석 (config.py 수정 후)
uv run python main.py --basic-only
```

### 고급 분석 실행

```bash
# 전체 분석 (기본 + 고급)
uv run python main.py

# 고급 프레임 분석만
uv run python main_advanced.py
```

### 설정 커스터마이징

`config.py`에서 다양한 설정을 조정할 수 있습니다:

```python
# 분석 대상 설정
TARGET_NAME = "Your Analysis Name"

# 필터링 설정
COMMENT_FILTERING = {
    'upvote_filtering': {
        'enabled': True,
        'min_upvotes': 5  # 최소 추천수
    },
    'keyword_filtering': {
        'enabled': True,
        'required_keywords': ['your', 'keywords']  # 필수 키워드
    }
}

# 토픽 모델링 설정
TOPIC_MODELS = {
    'bertopic': {
        'min_topic_size': 10,  # 최소 토픽 크기
        'nr_topics': 6         # 토픽 수
    }
}

# 시간 분석 설정
ADAPTIVE_TIME_ANALYSIS = {
    'min_time_unit': 'monthly',  # 'daily', 'weekly', 'monthly'
    'high_ratio_threshold': 0.25  # 세분화 임계값
}
```

### 프로그래밍 방식 사용

```python
from config import AnalysisConfig
from src.data_processor import DataProcessor
from src.sentiment_analyzer import SentimentAnalyzer
from src.topic_analyzer import TopicAnalyzer

# 설정 로드
config = AnalysisConfig()

# 데이터 처리
processor = DataProcessor(config)
df = processor.load_data('data/your_data.csv')
processed_df = processor.preprocess_comments(df, 'analysis_name')

# 감성 분석
sentiment_analyzer = SentimentAnalyzer(config)
sentiment_results = sentiment_analyzer.analyze_batch_sentiment(
    processed_df['cleaned_text'].tolist()
)

# 토픽 분석
topic_analyzer = TopicAnalyzer(config)
topic_results = topic_analyzer.analyze_topics_bertopic(
    processed_df['cleaned_text'].tolist(), 'analysis_name'
)
```

## 📈 결과물

### 생성되는 파일들

분석 완료 후 `output/` 디렉토리에 다음 파일들이 생성됩니다:

```
output/
├── visualizations/                    # 시각화 파일
│   ├── sentiment_trends.png          # 감성 트렌드 차트
│   ├── topic_dashboard.png           # 토픽 분석 대시보드
│   ├── comprehensive_dashboard.html  # 인터랙티브 대시보드
│   └── wordcloud_*.png              # 기간별 워드클라우드
├── reports/                          # 보고서 파일
│   ├── comprehensive_report.html    # HTML 종합 보고서
│   ├── analysis_report.md           # Markdown 보고서
│   └── analysis_summary.json        # JSON 요약 데이터
└── data_processed/                   # 처리된 데이터
    ├── analysis_results.pkl         # 전체 분석 결과
    ├── time_summary.csv            # 시간별 요약 통계
    └── keywords_detailed.csv       # 키워드 상세 분석
```

### 주요 분석 지표

#### 감성 분석
- **긍정/부정 비율**: 시간별 감성 분포
- **감정 강도**: 6가지 세부 감정의 강도 변화
- **감성 변동성**: 감성 변화의 표준편차
- **변곡점**: 급격한 감성 변화 시점

#### 토픽 분석
- **토픽 일관성**: Coherence Score (C_v)
- **토픽 다양성**: 토픽 간 유사도 분포
- **토픽 진화**: 시간에 따른 토픽 변화
- **키워드 순위**: 토픽별 주요 키워드

#### 시간 분석
- **적응적 세분화**: 데이터 밀도에 따른 동적 구간 설정
- **주기성 분석**: 주기적 패턴의 강도
- **트렌드 분석**: 장기적 변화 방향

## 📁 프로젝트 구조

```
youtube-comments-analysis/
├── 📄 README.md                     # 이 파일
├── 📄 requirements.txt              # Python 의존성
├── 📄 pyproject.toml                # uv 프로젝트 설정
├── 📄 config.py                     # 분석 설정
├── 📄 main.py                       # 기본 분석 실행
├── 📄 main_advanced.py              # 고급 분석 실행
├── 📄 .gitignore                    # Git 무시 파일 (데이터 보호)
├── 📁 src/                          # 소스 코드
│   ├── 📄 data_processor.py         # 데이터 전처리
│   ├── 📄 data_filter.py            # 데이터 필터링
│   ├── 📄 sentiment_analyzer.py     # 감성 분석
│   ├── 📄 topic_analyzer.py         # 토픽 분석
│   ├── 📄 adaptive_time_analyzer.py # 적응적 시간 분석
│   ├── 📄 visualizer.py             # 기본 시각화
│   ├── 📄 advanced_frame_analyzer.py # 고급 프레임 분석
│   ├── 📄 advanced_visualizer.py    # 고급 시각화
│   └── 📄 report_generator.py       # 보고서 생성
├── 📁 data/                         # 데이터 디렉토리
│   ├── 📄 README.md                 # 데이터 가이드
│   ├── 📄 sample_youtube_comments.csv # 샘플 데이터
│   ├── 📄 example_data_format.csv   # 형식 예시
│   └── 📄 .gitkeep                  # 디렉토리 유지
├── 📁 fonts/                        # 한글 폰트
│   └── 📄 AppleGothic.ttf          # 한글 폰트 파일
└── 📁 output/                       # 분석 결과 (런타임 생성)
    ├── 📁 visualizations/           # 시각화 파일
    ├── 📁 reports/                  # 보고서 파일
    ├── 📁 data_processed/           # 처리된 데이터
    └── 📁 models/                   # 저장된 모델
```

### 핵심 모듈 설명

#### 데이터 처리 파이프라인
- **`DataProcessor`**: CSV 로드, 텍스트 전처리, 날짜 파싱
- **`DataFilter`**: 품질 필터링, 키워드 필터링, 중복 제거
- **`AdaptiveTimeAnalyzer`**: 동적 시간 구간 설정

#### 분석 엔진
- **`SentimentAnalyzer`**: KoELECTRA 기반 감성 분석
- **`TopicAnalyzer`**: BERTopic + LDA 하이브리드 토픽 모델링
- **`AdvancedFrameAnalyzer`**: 변곡점 탐지, 네트워크 분석

#### 시각화 및 보고서
- **`Visualizer`**: 기본 차트 및 대시보드
- **`AdvancedVisualizer`**: 고급 시각화
- **`ReportGenerator`**: HTML/Markdown/JSON 보고서

## 🔧 문제 해결

### 일반적인 문제들

#### 1. 한국어 폰트 문제
```bash
# 폰트 설치 확인
python -c "import matplotlib.font_manager as fm; print([f.name for f in fm.fontManager.ttflist if 'Gothic' in f.name])"

# 폰트 캐시 재생성
python -c "import matplotlib.font_manager as fm; fm._rebuild()"
```

#### 2. Mecab 설치 문제
```bash
# Ubuntu
sudo apt-get update
sudo apt-get install mecab libmecab-dev mecab-ipadic-utf8

# macOS
brew install mecab mecab-ko mecab-ko-dic

# 설치 확인
python -c "from konlpy.tag import Mecab; print('Mecab 설치 완료')"
```

#### 3. GPU 메모리 부족
```python
# config.py에서 배치 크기 조정
TOPIC_MODELS = {
    'bertopic': {
        'low_memory': True,  # 저메모리 모드 활성화
        'calculate_probabilities': False  # 확률 계산 비활성화
    }
}
```

#### 4. 데이터 형식 오류
```python
# 날짜 형식 확인
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# 텍스트 인코딩 확인
df = pd.read_csv('data.csv', encoding='utf-8')
```

### 성능 최적화

#### GPU 사용 설정
```python
# config.py에서 자동 디바이스 선택
DEVICE = get_device()  # CUDA > MPS > CPU 순으로 선택
```

#### 메모리 사용량 최적화
```python
# 샘플링으로 메모리 절약
ANALYSIS_PARAMS = {
    'sample_size': 10000,  # 큰 데이터셋의 경우 샘플링
}
```

## 📚 참고 문헌

- Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure.
- Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent dirichlet allocation.
- Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks.

---

