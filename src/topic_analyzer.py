"""
Topic Analyzer Module
토픽 분석 모듈
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
import re
from collections import Counter

# BERTopic 관련
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer

# LDA 관련
from gensim import corpora, models
from gensim.models import LdaModel
from gensim.parsing.preprocessing import STOPWORDS
import pyLDAvis.gensim_models as gensimvis

# 한국어 처리
from konlpy.tag import Okt, Mecab
import warnings
warnings.filterwarnings('ignore')

# 한국어 토크나이저 클래스
class KoreanTokenizer:
    """한국어 형태소 분석 기반 토크나이저"""
    
    def __init__(self, use_mecab=True):
        """
        초기화
        Args:
            use_mecab: Mecab 사용 여부 (False면 Okt 사용)
        """
        self.tokenizer_name = "Okt"  # 기본값 설정
        
        try:
            if use_mecab:
                self.tokenizer = Mecab()
                self.tokenizer_name = "Mecab"
                print(f"✅ {self.tokenizer_name} 토크나이저 초기화 성공")
            else:
                self.tokenizer = Okt()
                self.tokenizer_name = "Okt"
                print(f"✅ {self.tokenizer_name} 토크나이저 초기화 성공")
        except Exception as e:
            print(f"⚠️ Mecab 초기화 실패, Okt로 대체: {e}")
            self.tokenizer = Okt()
            self.tokenizer_name = "Okt"
            print(f"✅ {self.tokenizer_name} 토크나이저로 대체 완료")
    
    def __call__(self, text):
        """
        텍스트를 토큰화
        Args:
            text: 입력 텍스트
        Returns:
            토큰 리스트
        """
        try:
            if self.tokenizer_name == "Mecab":
                # Mecab의 경우 형태소 분석 후 의미있는 품사만 선택
                morphs = self.tokenizer.pos(text)
                tokens = []
                for word, pos in morphs:
                    # 명사, 형용사, 동사만 선택
                    if pos.startswith(('NN', 'VA', 'VV')) and len(word) > 1:
                        tokens.append(word)
                return tokens
            else:
                # Okt의 경우
                return self.tokenizer.morphs(text, stem=True)
        except Exception as e:
            # 오류 발생시 공백으로 분할
            return text.split()

class TopicAnalyzer:
    """토픽 분석 클래스"""
    
    def __init__(self, config):
        """
        초기화
        Args:
            config: AnalysisConfig 객체
        """
        self.config = config
        self.logger = self._setup_logger()
        self.device = config.DEVICE
        
        # 한국어 형태소 분석기
        self.okt = Okt()
        
        # 모델 초기화
        self.bertopic_model = None
        self.lda_model = None
        self.embedding_model = None
        
        # 결과 저장
        self.results = {}
        
        self._initialize_models()
    
    def _setup_logger(self):
        """로거 설정"""
        logger = logging.getLogger(__name__)
        logger.setLevel(getattr(logging, self.config.LOGGING['level']))
        
        if not logger.handlers:
            handler = logging.FileHandler(self.config.LOGGING['file'], encoding='utf-8')
            formatter = logging.Formatter(self.config.LOGGING['format'])
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def _initialize_models(self):
        """토픽 모델 초기화"""
        try:
            self.logger.info("🔍 토픽 분석 모델 초기화 시작")
            
            # BERTopic용 한국어 임베딩 모델
            bertopic_config = self.config.TOPIC_MODELS['bertopic']
            self.logger.info(f"📥 한국어 임베딩 모델 로드: {bertopic_config['embedding_model']}")
            
            self.embedding_model = SentenceTransformer(bertopic_config['embedding_model'])
            
            # BERTopic 모델 설정 (한국어 최적화)
            umap_model = UMAP(
                n_neighbors=15,
                n_components=5,
                min_dist=0.0,
                metric='cosine',
                random_state=42
            )
            
            hdbscan_model = HDBSCAN(
                min_cluster_size=bertopic_config['min_topic_size'],
                metric='euclidean',
                cluster_selection_method='eom',
                prediction_data=True
            )
            
            # 한국어 형태소 분석 기반 토크나이저 초기화
            korean_tokenizer = KoreanTokenizer(use_mecab=True)
            
            # 한국어 불용어 설정
            korean_stopwords = set(self.config.KOREAN_STOPWORDS)
            
            # 한국어 최적화 CountVectorizer (동적 설정)
            vectorizer_model = CountVectorizer(
                tokenizer=korean_tokenizer,  # 한국어 형태소 분석기 사용
                ngram_range=(1, 2),
                stop_words=list(korean_stopwords),
                min_df=2,  # 최소 문서 빈도 감소
                max_df=0.95,  # 최대 문서 빈도 증가
                max_features=500,  # 최대 특성 수 감소
                lowercase=False  # 한국어는 대소문자 구분 없음
            )
            
            self.bertopic_model = BERTopic(
                embedding_model=self.embedding_model,
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                vectorizer_model=vectorizer_model,
                language='korean',
                calculate_probabilities=True,
                verbose=True
            )
            
            self.logger.info("✅ 한국어 최적화 토픽 분석 모델 초기화 완료")
            
        except Exception as e:
            self.logger.error(f"❌ 토픽 분석 모델 초기화 실패: {str(e)}")
            raise
    
    def preprocess_for_topic_analysis(self, texts: List[str]) -> List[str]:
        """
        토픽 분석용 텍스트 전처리
        Args:
            texts: 원본 텍스트 리스트
        Returns:
            전처리된 텍스트 리스트
        """
        try:
            self.logger.info(f"🔧 토픽 분석용 텍스트 전처리 시작: {len(texts):,}개")
            
            processed_texts = []
            korean_stopwords = set(self.config.KOREAN_STOPWORDS)
            
            for text in texts:
                if not text or len(text.strip()) < 5:
                    processed_texts.append("")
                    continue
                
                try:
                    # 형태소 분석 및 명사, 형용사 추출
                    morphs = self.okt.pos(text, stem=True)
                    
                    # 의미있는 품사만 선택 (명사, 형용사, 동사)
                    meaningful_words = []
                    for word, pos in morphs:
                        if (pos in ['Noun', 'Adjective', 'Verb'] and 
                            len(word) > 1 and 
                            word not in korean_stopwords and
                            not word.isdigit() and
                            not re.match(r'^[ㄱ-ㅎㅏ-ㅣ]+$', word)):  # 자음/모음만 있는 단어 제외
                            meaningful_words.append(word)
                    
                    # 최소 단어 수 확인
                    if len(meaningful_words) >= 2:
                        processed_texts.append(' '.join(meaningful_words))
                    else:
                        processed_texts.append("")
                        
                except Exception as e:
                    self.logger.warning(f"⚠️ 텍스트 전처리 실패: {str(e)}")
                    processed_texts.append("")
            
            # 빈 텍스트 제거
            valid_texts = [text for text in processed_texts if text.strip()]
            
            self.logger.info(f"✅ 전처리 완료: {len(valid_texts):,}개 유효 텍스트")
            return processed_texts
            
        except Exception as e:
            self.logger.error(f"❌ 토픽 분석용 전처리 실패: {str(e)}")
            raise
    
    def analyze_bertopic(self, texts: List[str], target_name: str) -> Dict:
        """
        BERTopic 분석
        Args:
            texts: 분석할 텍스트 리스트
            target_name: 분석 대상 이름
        Returns:
            BERTopic 분석 결과
        """
        try:
            self.logger.info(f"🎯 {target_name} BERTopic 분석 시작")
            
            # 텍스트 전처리
            processed_texts = self.preprocess_for_topic_analysis(texts)
            
            # 빈 텍스트 제거
            valid_texts = [text for text in processed_texts if text.strip()]
            valid_indices = [i for i, text in enumerate(processed_texts) if text.strip()]
            
            if len(valid_texts) < 10:
                self.logger.warning(f"⚠️ 유효한 텍스트가 너무 적습니다: {len(valid_texts)}개")
                return {
                    'topics': [],
                    'topic_labels': [],
                    'document_topics': [],
                    'topic_words': {},
                    'topic_info': pd.DataFrame()
                }
            
            self.logger.info(f"📊 BERTopic 모델 학습 중... ({len(valid_texts):,}개 문서)")
            
            # 문서 수에 따른 동적 파라미터 조정
            doc_count = len(valid_texts)
            
            # 동적 BERTopic 모델 생성 (더 보수적인 파라미터)
            umap_model = UMAP(
                n_neighbors=min(8, max(2, doc_count // 20)),  # 더 작은 이웃 수
                n_components=2,  # 고정된 차원
                min_dist=0.0,
                metric='cosine',
                random_state=42
            )
            
            hdbscan_model = HDBSCAN(
                min_cluster_size=max(2, min(4, doc_count // 40)),  # 더 작은 클러스터 크기
                metric='euclidean',
                cluster_selection_method='eom',
                prediction_data=False  # 예측 데이터 비활성화
            )
            
            # 한국어 형태소 분석 기반 토크나이저 초기화
            korean_tokenizer = KoreanTokenizer(use_mecab=True)
            
            # 한국어 불용어 설정
            korean_stopwords = set(self.config.KOREAN_STOPWORDS)
            
            # 한국어 최적화 CountVectorizer (더 간단한 설정)
            vectorizer_model = CountVectorizer(
                tokenizer=korean_tokenizer,  # 한국어 형태소 분석기 사용
                ngram_range=(1, 1),  # 단일 단어만 사용
                stop_words=list(korean_stopwords),
                min_df=1,  # 최소 문서 빈도 1로 설정
                max_df=0.99,  # 최대 문서 빈도 99%
                max_features=200,  # 최대 특성 수 더 감소
                lowercase=False  # 한국어는 대소문자 구분 없음
            )
            
            dynamic_bertopic_model = BERTopic(
                embedding_model=self.embedding_model,
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                vectorizer_model=vectorizer_model,
                language='korean',
                calculate_probabilities=False,  # 확률 계산 비활성화
                verbose=False  # 로그 출력 줄이기
            )
            
            # BERTopic 모델 학습
            topics, probabilities = dynamic_bertopic_model.fit_transform(valid_texts)
            
            # 토픽 정보 추출
            topic_info = dynamic_bertopic_model.get_topic_info()
            topic_words = {}
            topic_labels = []
            
            for topic_id in topic_info['Topic'].values:
                if topic_id != -1:  # 노이즈 토픽 제외
                    words = dynamic_bertopic_model.get_topic(topic_id)
                    topic_words[topic_id] = words
                    
                    # 토픽 라벨 생성 (상위 3개 단어)
                    top_words = [word for word, _ in words[:3]]
                    topic_labels.append(f"토픽{topic_id}: {', '.join(top_words)}")
            
            # 문서별 토픽 할당 (원본 인덱스에 맞춰 조정)
            document_topics = [-1] * len(texts)
            for i, topic in enumerate(topics):
                if i < len(valid_indices):
                    document_topics[valid_indices[i]] = topic
            
            result = {
                'topics': topics,
                'topic_labels': topic_labels,
                'document_topics': document_topics,
                'topic_words': topic_words,
                'topic_info': topic_info,
                'probabilities': probabilities,
                'model': dynamic_bertopic_model
            }
            
            self.logger.info(f"✅ BERTopic 분석 완료: {len(topic_words)}개 토픽 발견")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ BERTopic 분석 실패: {str(e)}")
            raise
    
    def find_optimal_topic_number(self, corpus, dictionary, documents, min_topics=2, max_topics=15):
        """
        최적 토픽 수 찾기 (휴리스틱 방법)
        Args:
            corpus: 코퍼스
            dictionary: 사전
            documents: 문서 리스트
            min_topics: 최소 토픽 수
            max_topics: 최대 토픽 수
        Returns:
            최적 토픽 수
        """
        try:
            # 문서 수에 따른 휴리스틱 방법 (빠르고 안정적)
            doc_count = len(documents)
            
            if doc_count < 50:
                optimal_topics = min(max(doc_count // 15, 2), 4)
            elif doc_count < 100:
                optimal_topics = min(max(doc_count // 20, 3), 6)
            elif doc_count < 500:
                optimal_topics = min(max(doc_count // 25, 4), 8)
            else:
                optimal_topics = min(max(doc_count // 30, 5), 10)
            
            # 범위 제한
            optimal_topics = max(min_topics, min(optimal_topics, max_topics))
            
            self.logger.info(f"📊 휴리스틱 최적 토픽 수: {optimal_topics} (문서 수: {doc_count})")
            return optimal_topics
                
        except Exception as e:
            self.logger.warning(f"⚠️ 최적 토픽 수 계산 실패, 기본값 사용: {str(e)}")
            return min_topics

    def analyze_lda(self, texts: List[str], target_name: str) -> Dict:
        """
        LDA 토픽 분석 (개선된 버전)
        Args:
            texts: 분석할 텍스트 리스트
            target_name: 분석 대상 이름
        Returns:
            LDA 분석 결과
        """
        try:
            self.logger.info(f"📚 {target_name} LDA 분석 시작")
            
            # 텍스트 전처리
            processed_texts = self.preprocess_for_topic_analysis(texts)
            
            # 빈 텍스트 제거 및 토큰화
            documents = []
            valid_indices = []
            
            for i, text in enumerate(processed_texts):
                if text.strip():
                    tokens = text.split()
                    if len(tokens) >= 2:  # 최소 2개 단어
                        documents.append(tokens)
                        valid_indices.append(i)
            
            if len(documents) < 10:
                self.logger.warning(f"⚠️ LDA 분석용 유효 문서가 너무 적습니다: {len(documents)}개")
                return {
                    'topics': [],
                    'topic_words': {},
                    'document_topics': [],
                    'coherence_score': 0.0,
                    'optimal_topic_count': 0
                }
            
            self.logger.info(f"📊 LDA 모델 학습 중... ({len(documents):,}개 문서)")
            
            # 사전 및 코퍼스 생성
            dictionary = corpora.Dictionary(documents)
            
            # 너무 빈번하거나 희귀한 단어 제거
            dictionary.filter_extremes(no_below=2, no_above=0.8)
            
            corpus = [dictionary.doc2bow(doc) for doc in documents]
            
            # 최적 토픽 수 결정
            max_possible_topics = min(15, len(documents) // 5)
            if len(documents) >= 50:  # 충분한 문서가 있을 때만 최적화 수행
                num_topics = self.find_optimal_topic_number(corpus, dictionary, documents, 
                                                          min_topics=2, max_topics=max_possible_topics)
            else:
                # 문서가 적을 때는 휴리스틱 방법 사용
                num_topics = min(max(len(documents) // 10, 2), max_possible_topics)
            
            # LDA 모델 학습 (메모리 절약 설정)
            lda_config = self.config.TOPIC_MODELS['lda']
            
            # gensim 버전에 따라 workers 파라미터 지원 여부가 다름
            lda_params = {
                'corpus': corpus,
                'id2word': dictionary,
                'num_topics': num_topics,
                'random_state': lda_config.get('random_state', 42),
                'passes': lda_config.get('passes', 2),
                'alpha': lda_config.get('alpha', 'auto'),
                'eta': lda_config.get('eta', 'auto'),
                'per_word_topics': lda_config.get('per_word_topics', False),
                'chunksize': lda_config.get('chunksize', 200),
                'iterations': lda_config.get('iterations', 30),
                'eval_every': lda_config.get('eval_every', None)
            }
            
            # workers 파라미터는 gensim 버전에 따라 지원되지 않을 수 있으므로 조건부 추가
            try:
                # 먼저 workers 파라미터 없이 시도
                self.lda_model = LdaModel(**lda_params)
            except Exception as e:
                self.logger.warning(f"⚠️ LDA 모델 생성 실패, 재시도: {str(e)}")
                # 파라미터를 더 단순화해서 재시도
                simple_params = {
                    'corpus': corpus,
                    'id2word': dictionary,
                    'num_topics': num_topics,
                    'random_state': 42,
                    'passes': 2,
                    'alpha': 'auto',
                    'eta': 'auto'
                }
                self.lda_model = LdaModel(**simple_params)
            
            # 토픽별 주요 단어 추출
            topic_words = {}
            topic_labels = []
            
            for topic_id in range(num_topics):
                words = self.lda_model.show_topic(topic_id, topn=10)
                topic_words[topic_id] = [(word, prob) for word, prob in words]
                
                # 토픽 라벨 생성 (더 의미있는 라벨)
                top_words = [word for word, _ in words[:3]]
                topic_labels.append(f"토픽{topic_id+1}: {', '.join(top_words)}")
            
            # 문서별 주요 토픽 할당
            document_topics = [-1] * len(texts)
            
            for i, doc in enumerate(corpus):
                topic_probs = self.lda_model.get_document_topics(doc)
                if topic_probs:
                    # 가장 높은 확률의 토픽 선택
                    main_topic = max(topic_probs, key=lambda x: x[1])[0]
                    document_topics[valid_indices[i]] = main_topic
            
            # 일관성 점수 계산 (간단한 버전으로 대체)
            try:
                coherence_score = self._calculate_simple_coherence(topic_words, documents)
            except Exception as e:
                self.logger.warning(f"⚠️ 일관성 점수 계산 실패: {str(e)}")
                coherence_score = 0.0
            
            # 토픽 품질 평가 추가
            topic_quality = self._evaluate_topic_quality(topic_words, documents)
            
            result = {
                'topics': list(range(num_topics)),
                'topic_words': topic_words,
                'topic_labels': topic_labels,
                'document_topics': document_topics,
                'coherence_score': coherence_score,
                'optimal_topic_count': num_topics,
                'topic_quality': topic_quality,
                'model': self.lda_model,
                'dictionary': dictionary,
                'corpus': corpus
            }
            
            self.logger.info(f"✅ LDA 분석 완료: {num_topics}개 토픽, 일관성: {coherence_score:.3f}, 품질: {topic_quality:.3f}")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ LDA 분석 실패: {str(e)}")
            raise
    
    def _calculate_simple_coherence(self, topic_words: Dict, documents: List[List[str]]) -> float:
        """
        매우 간단한 토픽 일관성 점수 계산
        Args:
            topic_words: 토픽별 단어 딕셔너리
            documents: 문서 리스트
        Returns:
            일관성 점수 (0-1)
        """
        try:
            if not topic_words or not documents:
                return 0.0
            
            # 토픽별 단어 다양성만 간단히 계산
            topic_scores = []
            
            for topic_id, words in topic_words.items():
                if len(words) < 2:
                    continue
                
                # 상위 3개 단어만 사용 (더 간단하게)
                top_words = [word for word, _ in words[:3]]
                
                # 단어들이 문서에 나타나는 빈도 계산
                word_doc_counts = []
                for word in top_words:
                    count = sum(1 for doc in documents if word in doc)
                    word_doc_counts.append(count)
                
                # 평균 문서 출현 빈도를 일관성 점수로 사용
                if word_doc_counts:
                    avg_frequency = np.mean(word_doc_counts) / len(documents)
                    topic_scores.append(min(avg_frequency, 1.0))
            
            return np.mean(topic_scores) if topic_scores else 0.0
            
        except Exception as e:
            self.logger.warning(f"⚠️ 간단한 일관성 계산 실패: {str(e)}")
            return 0.0

    def _evaluate_topic_quality(self, topic_words: Dict, documents: List[List[str]]) -> float:
        """
        토픽 품질 평가
        Args:
            topic_words: 토픽별 단어 딕셔너리
            documents: 문서 리스트
        Returns:
            품질 점수 (0-1)
        """
        try:
            # 토픽 간 단어 중복도 계산
            all_topic_words = set()
            topic_word_sets = []
            
            for topic_id, words in topic_words.items():
                topic_set = set([word for word, _ in words[:5]])  # 상위 5개 단어
                topic_word_sets.append(topic_set)
                all_topic_words.update(topic_set)
            
            # 중복도가 낮을수록 좋음
            total_words = len(all_topic_words)
            unique_words = sum(len(topic_set) for topic_set in topic_word_sets)
            
            if unique_words == 0:
                return 0.0
            
            diversity_score = total_words / unique_words
            
            # 토픽 내 단어 응집도 계산 (간단한 버전)
            coherence_scores = []
            for topic_set in topic_word_sets:
                if len(topic_set) > 1:
                    # 토픽 단어들이 같은 문서에 함께 나타나는 빈도
                    co_occurrence = 0
                    total_pairs = 0
                    
                    for doc in documents:
                        doc_set = set(doc)
                        topic_in_doc = topic_set.intersection(doc_set)
                        if len(topic_in_doc) > 1:
                            co_occurrence += len(topic_in_doc) * (len(topic_in_doc) - 1) / 2
                        total_pairs += len(topic_set) * (len(topic_set) - 1) / 2
                    
                    if total_pairs > 0:
                        coherence_scores.append(co_occurrence / total_pairs)
            
            coherence_score = np.mean(coherence_scores) if coherence_scores else 0.0
            
            # 최종 품질 점수 (다양성과 응집도의 조화평균)
            if diversity_score > 0 and coherence_score > 0:
                quality_score = 2 * diversity_score * coherence_score / (diversity_score + coherence_score)
            else:
                quality_score = 0.0
            
            return min(quality_score, 1.0)
            
        except Exception as e:
            self.logger.warning(f"⚠️ 토픽 품질 평가 실패: {str(e)}")
            return 0.0
    
    def analyze_monthly_topics(self, monthly_data: Dict[str, pd.DataFrame], target_name: str) -> Dict:
        """
        월별 토픽 분석
        Args:
            monthly_data: 월별 데이터 딕셔너리
            target_name: 분석 대상 이름
        Returns:
            월별 토픽 분석 결과
        """
        try:
            self.logger.info(f"📅 {target_name} 월별 토픽 분석 시작")
            
            monthly_results = {}
            
            for year_month, df in monthly_data.items():
                self.logger.info(f"📊 {year_month} 토픽 분석 중... ({len(df):,}개 댓글)")
                
                if len(df) < 10:  # 최소 문서 수 확인
                    self.logger.warning(f"⚠️ {year_month}: 문서 수가 너무 적어 토픽 분석 생략")
                    continue
                
                texts = df['cleaned_text'].tolist()
                
                # BERTopic 분석
                try:
                    bertopic_result = self.analyze_bertopic(texts, f"{target_name}_{year_month}")
                except Exception as e:
                    self.logger.warning(f"⚠️ {year_month} BERTopic 분석 실패: {str(e)}")
                    bertopic_result = {}
                
                # LDA 분석
                try:
                    lda_result = self.analyze_lda(texts, f"{target_name}_{year_month}")
                except Exception as e:
                    self.logger.warning(f"⚠️ {year_month} LDA 분석 실패: {str(e)}")
                    lda_result = {}
                
                monthly_results[year_month] = {
                    'total_comments': len(df),
                    'bertopic': bertopic_result,
                    'lda': lda_result
                }
                
                # 주요 토픽 로깅
                if bertopic_result and 'topic_labels' in bertopic_result:
                    self.logger.info(f"  BERTopic: {len(bertopic_result['topic_labels'])}개 토픽")
                if lda_result and 'topic_labels' in lda_result:
                    self.logger.info(f"  LDA: {len(lda_result['topic_labels'])}개 토픽")
            
            self.results[target_name] = monthly_results
            
            self.logger.info(f"✅ {target_name} 월별 토픽 분석 완료")
            return monthly_results
            
        except Exception as e:
            self.logger.error(f"❌ {target_name} 월별 토픽 분석 실패: {str(e)}")
            raise
    
    def analyze_time_grouped_topics(self, time_groups: Dict[str, pd.DataFrame], target_name: str) -> Dict:
        """
        적응적 시간 그룹별 토픽 분석
        Args:
            time_groups: 시간 그룹별 데이터 딕셔너리
            target_name: 분석 대상 이름
        Returns:
            시간 그룹별 토픽 분석 결과
        """
        try:
            self.logger.info(f"📅 {target_name} 시간 그룹별 토픽 분석 시작")
            
            time_group_results = {}
            
            for group_name, df in time_groups.items():
                self.logger.info(f"📊 {group_name} 토픽 분석 중... ({len(df):,}개 댓글)")
                
                if len(df) < 10:  # 최소 문서 수 확인
                    self.logger.warning(f"⚠️ {group_name}: 문서 수가 너무 적어 토픽 분석 생략")
                    continue
                
                texts = df['cleaned_text'].tolist()
                
                # BERTopic 분석
                try:
                    bertopic_result = self.analyze_bertopic(texts, f"{target_name}_{group_name}")
                except Exception as e:
                    self.logger.warning(f"⚠️ {group_name} BERTopic 분석 실패: {str(e)}")
                    bertopic_result = {}
                
                # LDA 분석
                try:
                    lda_result = self.analyze_lda(texts, f"{target_name}_{group_name}")
                except Exception as e:
                    self.logger.warning(f"⚠️ {group_name} LDA 분석 실패: {str(e)}")
                    lda_result = {}
                
                time_group_results[group_name] = {
                    'total_comments': len(df),
                    'bertopic': bertopic_result,
                    'lda': lda_result
                }
                
                # 주요 토픽 로깅
                if bertopic_result and 'topic_labels' in bertopic_result:
                    self.logger.info(f"  BERTopic: {len(bertopic_result['topic_labels'])}개 토픽")
                if lda_result and 'topic_labels' in lda_result:
                    self.logger.info(f"  LDA: {len(lda_result['topic_labels'])}개 토픽")
            
            self.results[target_name] = time_group_results
            
            self.logger.info(f"✅ {target_name} 시간 그룹별 토픽 분석 완료")
            return time_group_results
            
        except Exception as e:
            self.logger.error(f"❌ {target_name} 시간 그룹별 토픽 분석 실패: {str(e)}")
            raise
    
    def get_topic_evolution(self, target_name: str) -> Dict:
        """
        토픽 진화 분석
        Args:
            target_name: 분석 대상 이름
        Returns:
            토픽 진화 분석 결과
        """
        try:
            if target_name not in self.results:
                raise ValueError(f"{target_name}의 토픽 분석 결과가 없습니다.")
            
            monthly_results = self.results[target_name]
            sorted_months = sorted(monthly_results.keys())
            
            # BERTopic 진화
            bertopic_evolution = {
                'months': sorted_months,
                'topic_counts': [],
                'main_topics': [],
                'topic_words_evolution': {}
            }
            
            # LDA 진화
            lda_evolution = {
                'months': sorted_months,
                'topic_counts': [],
                'main_topics': [],
                'coherence_scores': []
            }
            
            for month in sorted_months:
                result = monthly_results[month]
                
                # BERTopic 진화 추적
                if 'bertopic' in result and result['bertopic']:
                    bertopic_data = result['bertopic']
                    bertopic_evolution['topic_counts'].append(len(bertopic_data.get('topic_labels', [])))
                    
                    if bertopic_data.get('topic_labels'):
                        bertopic_evolution['main_topics'].append(bertopic_data['topic_labels'][0])
                    else:
                        bertopic_evolution['main_topics'].append('없음')
                else:
                    bertopic_evolution['topic_counts'].append(0)
                    bertopic_evolution['main_topics'].append('없음')
                
                # LDA 진화 추적
                if 'lda' in result and result['lda']:
                    lda_data = result['lda']
                    lda_evolution['topic_counts'].append(len(lda_data.get('topic_labels', [])))
                    lda_evolution['coherence_scores'].append(lda_data.get('coherence_score', 0.0))
                    
                    if lda_data.get('topic_labels'):
                        lda_evolution['main_topics'].append(lda_data['topic_labels'][0])
                    else:
                        lda_evolution['main_topics'].append('없음')
                else:
                    lda_evolution['topic_counts'].append(0)
                    lda_evolution['coherence_scores'].append(0.0)
                    lda_evolution['main_topics'].append('없음')
            
            return {
                'bertopic_evolution': bertopic_evolution,
                'lda_evolution': lda_evolution
            }
            
        except Exception as e:
            self.logger.error(f"❌ {target_name} 토픽 진화 분석 실패: {str(e)}")
            raise
    
    def extract_keywords(self, texts: List[str], top_k: int = 20) -> List[Tuple[str, int]]:
        """
        키워드 추출
        Args:
            texts: 텍스트 리스트
            top_k: 상위 키워드 개수
        Returns:
            (키워드, 빈도) 튜플 리스트
        """
        try:
            self.logger.info(f"🔑 키워드 추출 시작: {len(texts):,}개 텍스트")
            
            # 텍스트 전처리
            processed_texts = self.preprocess_for_topic_analysis(texts)
            
            # 모든 단어 수집
            all_words = []
            for text in processed_texts:
                if text.strip():
                    all_words.extend(text.split())
            
            # 빈도 계산
            word_counts = Counter(all_words)
            
            # 상위 키워드 반환
            top_keywords = word_counts.most_common(top_k)
            
            self.logger.info(f"✅ 키워드 추출 완료: {len(top_keywords)}개")
            return top_keywords
            
        except Exception as e:
            self.logger.error(f"❌ 키워드 추출 실패: {str(e)}")
            raise 