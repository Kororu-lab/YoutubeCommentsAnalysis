# YouTube 댓글 분석 - 데이터 필터링 가이드

## 개요

이 프로젝트는 YouTube 댓글의 품질과 관련성을 높이기 위한 다양한 필터링 기능을 제공합니다. 필터링을 통해 분석의 정확도를 높이고 노이즈를 줄일 수 있습니다.

## 필터링 기능

### 1. 추천수 기반 필터링 (`upvote_filtering`)

댓글의 추천수(upvotes)와 순추천수(upvotes - downvotes)를 기준으로 필터링합니다.

```python
'upvote_filtering': {
    'enabled': True,  # 필터링 활성화 여부
    'min_upvotes': 1,  # 최소 추천수 (이 값 미만은 제외)
    'min_net_upvotes': 0,  # 최소 순추천수 (upvotes - downvotes)
    'use_net_upvotes_only_when_positive': True,  # 순추천수가 양수일 때만 필터링 적용
}
```

**사용 예시:**
- `min_upvotes: 2`: 추천수가 2개 미만인 댓글 제외
- `min_net_upvotes: 1`: 순추천수가 1개 미만인 댓글 제외
- 품질 높은 댓글만 분석하여 신뢰도 향상

### 2. 키워드 기반 필터링 (`keyword_filtering`)

특정 키워드를 포함하거나 제외하는 댓글을 필터링합니다.

```python
'keyword_filtering': {
    'enabled': True,
    'required_keywords': [  # 반드시 포함해야 하는 키워드 (OR 조건)
        '유아인', '마약', '대마초', '프로포폴', '약물', '투약', '복용',
        '수사', '조사', '체포', '구속', '재판', '판결', '혐의',
        '반성', '사과', '해명', '책임', '잘못', '후회'
    ],
    'excluded_keywords': [  # 제외할 키워드
        '광고', '홍보', '스팸', '도배', '구독', '좋아요',
        '링크', 'http', '카톡', '전화번호', '도박', '투자'
    ],
    'case_sensitive': False,  # 대소문자 구분 여부
    'partial_match': True,  # 부분 매칭 허용 여부
    'min_keyword_matches': 1,  # 최소 필수 키워드 매칭 수
}
```

**키워드 설정 가이드:**

**필수 키워드 (required_keywords):**
- 분석 주제와 직접 관련된 키워드들
- OR 조건으로 작동 (하나라도 포함되면 통과)
- 유아인 약물 사건 관련: `'유아인', '마약', '약물', '투약'` 등
- 법적 절차 관련: `'수사', '재판', '판결', '혐의'` 등
- 반응/태도 관련: `'반성', '사과', '책임', '잘못'` 등

**제외 키워드 (excluded_keywords):**
- 스팸, 광고, 무관한 내용 제거
- AND 조건으로 작동 (모든 제외 키워드가 없어야 함)
- 일반적인 스팸: `'광고', '홍보', '구독', '좋아요'`
- 외부 링크: `'http', 'www', '링크'`
- 무관한 주제: `'도박', '투자', '대출'`

### 3. 품질 기반 필터링 (`quality_filtering`)

댓글의 언어적 품질을 기준으로 필터링합니다.

```python
'quality_filtering': {
    'enabled': True,
    'min_korean_ratio': 0.3,  # 최소 한글 비율 (30%)
    'max_special_char_ratio': 0.5,  # 최대 특수문자 비율 (50%)
    'max_repeated_char_ratio': 0.3,  # 최대 반복 문자 비율 (30%)
    'min_meaningful_words': 2,  # 최소 의미있는 단어 수
    'exclude_single_char_comments': True,  # 한 글자 댓글 제외
    'exclude_emoji_only_comments': True,  # 이모지만 있는 댓글 제외
}
```

**품질 기준:**
- **한글 비율**: 한국어 댓글 분석이므로 한글 비율이 낮은 댓글 제외
- **특수문자 비율**: 과도한 특수문자가 있는 댓글 제외
- **반복 문자**: "ㅋㅋㅋㅋㅋㅋㅋㅋ" 같은 과도한 반복 제외
- **의미있는 단어**: 최소한의 의미 전달이 가능한 댓글만 포함

### 4. 중복 댓글 필터링 (`duplicate_filtering`)

동일하거나 유사한 댓글을 제거합니다.

```python
'duplicate_filtering': {
    'enabled': True,
    'similarity_threshold': 0.9,  # 유사도 임계값 (90%)
    'keep_highest_upvotes': True,  # 중복 시 추천수가 높은 것 유지
    'exact_match_only': False,  # 완전 일치만 중복으로 처리할지 여부
}
```

**중복 제거 방식:**
- **완전 일치**: 텍스트가 정확히 같은 댓글
- **유사도 기반**: 90% 이상 유사한 댓글 (복사-붙여넣기, 약간의 변형 포함)
- **우선순위**: 추천수가 높은 댓글을 우선 유지

### 5. 사용자 기반 필터링 (`user_filtering`)

특정 사용자나 스팸 사용자를 필터링합니다.

```python
'user_filtering': {
    'enabled': False,  # 기본 비활성화
    'exclude_users': [],  # 제외할 사용자 목록
    'min_user_comments': 1,  # 사용자별 최소 댓글 수
    'max_user_comments': 100,  # 사용자별 최대 댓글 수 (스팸 방지)
}
```

## 설정 방법

### 1. 기본 설정 사용

기본 설정은 유아인 약물 사건 분석에 최적화되어 있습니다.

```python
# config.py에서 기본 설정 확인
COMMENT_FILTERING = {
    'upvote_filtering': {'enabled': True, 'min_upvotes': 1},
    'keyword_filtering': {'enabled': True, 'required_keywords': [...]},
    # ... 기타 설정
}
```

### 2. 사용자 정의 설정

특정 분석 목적에 맞게 설정을 조정할 수 있습니다.

```python
# config.py 수정 예시

# 엄격한 품질 필터링
COMMENT_FILTERING['quality_filtering'].update({
    'min_korean_ratio': 0.5,  # 한글 비율 50% 이상
    'min_meaningful_words': 3,  # 의미있는 단어 3개 이상
})

# 관련성 높은 댓글만 분석
COMMENT_FILTERING['keyword_filtering'].update({
    'required_keywords': ['유아인', '마약', '약물'],  # 핵심 키워드만
    'min_keyword_matches': 1,
})

# 고품질 댓글만 분석
COMMENT_FILTERING['upvote_filtering'].update({
    'min_upvotes': 5,  # 추천수 5개 이상
    'min_net_upvotes': 3,  # 순추천수 3개 이상
})
```

### 3. 필터링 비활성화

특정 필터링을 비활성화하려면:

```python
# 모든 필터링 비활성화
COMMENT_FILTERING = {
    'upvote_filtering': {'enabled': False},
    'keyword_filtering': {'enabled': False},
    'quality_filtering': {'enabled': False},
    'duplicate_filtering': {'enabled': False},
    'user_filtering': {'enabled': False},
}
```

## 필터링 결과 확인

### 1. 로그 확인

필터링 과정과 결과는 로그에서 확인할 수 있습니다:

```
🔍 데이터 필터링 시작: 42,176개 댓글
📊 추천수 필터링 후: 35,420개 댓글 (6,756개 제거)
🔑 키워드 필터링 후: 3,594개 댓글 (31,826개 제거)
✨ 품질 필터링 후: 3,201개 댓글 (393개 제거)
🔄 중복 필터링 후: 3,089개 댓글 (112개 제거)
📤 최종 댓글 수: 3,089개
🗑️ 총 제거된 댓글: 39,087개 (92.7%)
```

### 2. 필터링 통계

프로그램 실행 후 필터링 통계를 확인할 수 있습니다:

```python
# main.py 실행 결과에서
filter_stats = data_filter.get_filtering_stats()
print(f"원본: {filter_stats['original_count']:,}개")
print(f"최종: {filter_stats['final_count']:,}개")
print(f"제거율: {(1 - filter_stats['final_count']/filter_stats['original_count'])*100:.1f}%")
```

## 권장 설정

### 1. 탐색적 분석 (Exploratory Analysis)

```python
# 넓은 범위의 댓글 포함
'upvote_filtering': {'min_upvotes': 0},
'keyword_filtering': {'required_keywords': ['유아인']},  # 최소한의 관련성
'quality_filtering': {'min_korean_ratio': 0.2},
```

### 2. 정밀 분석 (Precision Analysis)

```python
# 고품질, 고관련성 댓글만
'upvote_filtering': {'min_upvotes': 3, 'min_net_upvotes': 2},
'keyword_filtering': {'required_keywords': ['유아인', '마약', '약물'], 'min_keyword_matches': 2},
'quality_filtering': {'min_korean_ratio': 0.5, 'min_meaningful_words': 3},
```

### 3. 감성 분석 중심

```python
# 감정 표현이 풍부한 댓글
'keyword_filtering': {
    'required_keywords': ['유아인', '실망', '화나다', '안타깝다', '응원', '반성'],
    'excluded_keywords': ['광고', '스팸', '링크']
},
'quality_filtering': {'min_meaningful_words': 4},
```

### 4. 토픽 분석 중심

```python
# 다양한 주제 포함
'keyword_filtering': {
    'required_keywords': ['유아인', '마약', '연예계', '법적', '사회적'],
    'min_keyword_matches': 1
},
'duplicate_filtering': {'similarity_threshold': 0.8},  # 유사 댓글 더 많이 제거
```

## 주의사항

### 1. 과도한 필터링 주의

- 너무 엄격한 필터링은 분석 대상을 과도하게 줄일 수 있음
- 최소 1,000개 이상의 댓글이 남도록 조정 권장

### 2. 키워드 선택

- **필수 키워드**: 분석 주제와 직접 관련된 핵심 키워드만 포함
- **제외 키워드**: 명확한 스팸/무관 내용만 제외
- 정기적으로 키워드 목록 검토 및 업데이트

### 3. 편향 방지

- 특정 관점의 댓글만 남지 않도록 주의
- 긍정/부정 의견이 균형있게 포함되도록 확인

### 4. 성능 고려

- 대용량 데이터에서는 품질 필터링과 중복 제거가 시간이 오래 걸릴 수 있음
- 필요시 샘플링 후 필터링 적용 고려

## 문제 해결

### 1. 필터링 후 댓글이 너무 적은 경우

```python
# 필터링 기준 완화
'upvote_filtering': {'min_upvotes': 0},
'keyword_filtering': {'required_keywords': ['유아인']},  # 핵심 키워드만
'quality_filtering': {'min_korean_ratio': 0.1},
```

### 2. 관련 없는 댓글이 많이 포함되는 경우

```python
# 키워드 필터링 강화
'keyword_filtering': {
    'required_keywords': ['유아인', '마약', '약물', '사건'],
    'min_keyword_matches': 2,  # 2개 이상 키워드 매칭
    'excluded_keywords': [...],  # 제외 키워드 추가
}
```

### 3. 중복 댓글이 많은 경우

```python
# 중복 제거 강화
'duplicate_filtering': {
    'similarity_threshold': 0.7,  # 70% 유사도로 낮춤
    'exact_match_only': False,
}
```

이 가이드를 참고하여 분석 목적에 맞는 최적의 필터링 설정을 구성하시기 바랍니다. 