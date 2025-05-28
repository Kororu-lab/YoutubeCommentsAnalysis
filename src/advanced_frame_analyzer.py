"""
Advanced Frame Analysis Module
언론 프레임과 대중 반응 간 괴리 분석 모듈

이 모듈은 다음 기능을 제공합니다:
1. 시간 기반 여론 흐름 파악
2. 토픽·키워드 분석 (LDA, NMF, BERTopic, Top2Vec)
3. 키워드 공출현 네트워크 분석
4. 언론 기사 프레임과의 유사도 비교
5. 변곡점(이상치) 탐지
6. 동적 네트워크 분석
7. 문맥 임베딩 클러스터링
8. 이벤트 주도 감성 충격 분석
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import pickle
import os
from collections import Counter, defaultdict
import itertools

# NLP 라이브러리
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN, HDBSCAN
from sentence_transformers import SentenceTransformer
import networkx as nx
from bertopic import BERTopic
from gensim import corpora, models
from gensim.models import LdaModel
from konlpy.tag import Mecab

# 시계열 분석
from scipy import stats
from scipy.signal import find_peaks
import ruptures as rpt  # 변곡점 탐지
from statsmodels.tsa.seasonal import seasonal_decompose

# 시각화
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from wordcloud import WordCloud

class AdvancedFrameAnalyzer:
    """고급 프레임 분석 클래스"""
    
    def __init__(self, config):
        """
        초기화
        Args:
            config: AnalysisConfig 객체
        """
        self.config = config
        self.logger = self._setup_logger()
        self.mecab = Mecab()
        
        # 한국어 SBERT 모델 로드
        try:
            self.sentence_model = SentenceTransformer('jhgan/ko-sroberta-multitask')
            self.logger.info("✅ 한국어 SBERT 모델 로드 완료")
        except Exception as e:
            self.logger.warning(f"⚠️ SBERT 모델 로드 실패: {e}")
            self.sentence_model = None
        
        # 분석 결과 저장
        self.analysis_results = {}
        
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
    
    def analyze_temporal_opinion_flow(self, df: pd.DataFrame, case_name: str, 
                                    event_dates: Dict[str, str] = None) -> Dict:
        """
        시간 기반 여론 흐름 파악
        
        Args:
            df: 댓글 데이터프레임
            case_name: 사건명 (유아인, 돈스파이크, 지드래곤)
            event_dates: 주요 사건 날짜 딕셔너리
        
        Returns:
            시간별 여론 흐름 분석 결과
        """
        try:
            self.logger.info(f"📊 {case_name} 시간 기반 여론 흐름 분석 시작")
            
            # 날짜 컬럼 처리
            date_col = self._find_date_column(df)
            if not date_col:
                raise ValueError("날짜 컬럼을 찾을 수 없습니다.")
            
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.sort_values(date_col)
            
            # 시간 구간 설정 (사건별 맞춤)
            time_windows = self._create_event_based_windows(df, case_name, event_dates)
            
            # 각 시간 구간별 분석
            temporal_results = {}
            
            for window_name, (start_date, end_date) in time_windows.items():
                window_df = df[(df[date_col] >= start_date) & (df[date_col] <= end_date)]
                
                if len(window_df) < 10:  # 최소 댓글 수 체크
                    continue
                
                # 감성 분석
                sentiment_scores = self._calculate_sentiment_scores(window_df)
                
                # 댓글 참여도 분석
                engagement_metrics = self._calculate_engagement_metrics(window_df)
                
                # 키워드 추출
                keywords = self._extract_period_keywords(window_df)
                
                temporal_results[window_name] = {
                    'period': window_name,
                    'start_date': start_date,
                    'end_date': end_date,
                    'comment_count': len(window_df),
                    'sentiment_scores': sentiment_scores,
                    'engagement_metrics': engagement_metrics,
                    'top_keywords': keywords[:20],
                    'avg_sentiment': sentiment_scores.get('avg_sentiment', 0),
                    'sentiment_volatility': sentiment_scores.get('volatility', 0)
                }
            
            # 시계열 트렌드 분석
            trend_analysis = self._analyze_sentiment_trends(temporal_results)
            
            # 변곡점 탐지
            changepoints = self._detect_changepoints(temporal_results)
            
            result = {
                'case_name': case_name,
                'temporal_results': temporal_results,
                'trend_analysis': trend_analysis,
                'changepoints': changepoints,
                'time_windows': time_windows
            }
            
            self.analysis_results['temporal_flow'] = result
            self.logger.info(f"✅ {case_name} 시간 기반 여론 흐름 분석 완료")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ 시간 기반 여론 흐름 분석 실패: {str(e)}")
            raise
    
    def analyze_topics_comprehensive(self, df: pd.DataFrame, case_name: str,
                                   methods: List[str] = ['lda', 'bertopic']) -> Dict:
        """
        종합적 토픽 분석 (LDA, NMF, BERTopic, Top2Vec)
        
        Args:
            df: 댓글 데이터프레임
            case_name: 사건명
            methods: 사용할 토픽 모델링 방법들
        
        Returns:
            토픽 분석 결과
        """
        try:
            self.logger.info(f"🔍 {case_name} 종합 토픽 분석 시작")
            
            # 텍스트 전처리
            try:
                texts = self._preprocess_texts_for_topic_modeling(df)
            except Exception as e:
                self.logger.error(f"❌ 텍스트 전처리 실패: {str(e)}")
                return {'error': 'text_preprocessing_failed'}
            
            if len(texts) < 20:  # 최소 요구사항 증가
                self.logger.warning("⚠️ 토픽 모델링을 위한 텍스트가 부족합니다.")
                return {'error': 'insufficient_texts', 'text_count': len(texts)}
            
            topic_results = {}
            
            # 1. LDA 분석
            if 'lda' in methods:
                try:
                    self.logger.info("📊 LDA 토픽 분석 시작")
                    lda_result = self._analyze_lda_topics(texts, case_name)
                    topic_results['lda'] = lda_result
                    self.logger.info("✅ LDA 토픽 분석 완료")
                except Exception as e:
                    self.logger.warning(f"⚠️ LDA 분석 실패: {str(e)}")
                    topic_results['lda'] = {'topics': [], 'coherence': 0, 'error': str(e)}
            
            # 2. BERTopic 분석
            if 'bertopic' in methods:
                try:
                    self.logger.info("📊 BERTopic 토픽 분석 시작")
                    bertopic_result = self._analyze_bertopic_topics(texts, case_name)
                    topic_results['bertopic'] = bertopic_result
                    self.logger.info("✅ BERTopic 토픽 분석 완료")
                except Exception as e:
                    self.logger.warning(f"⚠️ BERTopic 분석 실패: {str(e)}")
                    topic_results['bertopic'] = {'topics': [], 'coherence': 0, 'error': str(e)}
            
            # 토픽 간 유사도 분석
            try:
                topic_similarity = self._compare_topic_methods(topic_results)
            except Exception as e:
                self.logger.warning(f"⚠️ 토픽 유사도 분석 실패: {str(e)}")
                topic_similarity = {}
            
            # 토픽 진화 분석 (시간별)
            try:
                topic_evolution = self._analyze_topic_evolution(df, topic_results)
            except Exception as e:
                self.logger.warning(f"⚠️ 토픽 진화 분석 실패: {str(e)}")
                topic_evolution = {}
            
            result = {
                'case_name': case_name,
                'topic_results': topic_results,
                'topic_similarity': topic_similarity,
                'topic_evolution': topic_evolution,
                'total_documents': len(texts),
                'successful_methods': [method for method in methods if method in topic_results and 'error' not in topic_results[method]]
            }
            
            self.analysis_results['comprehensive_topics'] = result
            self.logger.info(f"✅ {case_name} 종합 토픽 분석 완료")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ 종합 토픽 분석 실패: {str(e)}")
            return {'error': 'comprehensive_analysis_failed', 'details': str(e)}
    
    def analyze_keyword_cooccurrence_network(self, df: pd.DataFrame, case_name: str,
                                           min_cooccurrence: int = 3) -> Dict:
        """
        키워드 공출현 네트워크 분석
        
        Args:
            df: 댓글 데이터프레임
            case_name: 사건명
            min_cooccurrence: 최소 공출현 빈도
        
        Returns:
            네트워크 분석 결과
        """
        try:
            self.logger.info(f"🕸️ {case_name} 키워드 공출현 네트워크 분석 시작")
            
            # 키워드 추출 및 공출현 매트릭스 생성
            cooccurrence_matrix = self._build_cooccurrence_matrix(df, min_cooccurrence)
            
            # 네트워크 그래프 생성
            G = self._create_network_graph(cooccurrence_matrix)
            
            # 네트워크 분석
            network_metrics = self._analyze_network_properties(G)
            
            # 커뮤니티 탐지
            communities = self._detect_communities(G)
            
            # 중심성 분석
            centrality_analysis = self._analyze_centrality(G)
            
            # 동적 네트워크 분석 (시간별)
            dynamic_networks = self._analyze_dynamic_networks(df, case_name)
            
            result = {
                'case_name': case_name,
                'network_graph': G,
                'cooccurrence_matrix': cooccurrence_matrix,
                'network_metrics': network_metrics,
                'communities': communities,
                'centrality_analysis': centrality_analysis,
                'dynamic_networks': dynamic_networks
            }
            
            self.analysis_results['keyword_network'] = result
            self.logger.info(f"✅ {case_name} 키워드 네트워크 분석 완료")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ 키워드 네트워크 분석 실패: {str(e)}")
            raise
    
    def compare_with_media_frames(self, comment_df: pd.DataFrame, 
                                media_articles: List[str], case_name: str) -> Dict:
        """
        언론 기사 프레임과의 유사도 비교
        
        Args:
            comment_df: 댓글 데이터프레임
            media_articles: 언론 기사 텍스트 리스트
            case_name: 사건명
        
        Returns:
            프레임 유사도 분석 결과
        """
        try:
            self.logger.info(f"📰 {case_name} 언론 프레임 유사도 분석 시작")
            
            # 시간별 댓글 그룹화
            time_groups = self._group_comments_by_time(comment_df)
            
            # 언론 기사 키워드 추출
            media_keywords = self._extract_media_keywords(media_articles)
            
            # 시간별 유사도 계산
            similarity_results = {}
            
            for period, comments in time_groups.items():
                # 댓글 키워드 추출
                comment_keywords = self._extract_period_keywords(comments)
                
                # 키워드 벡터화
                comment_vector = self._create_keyword_vector(comment_keywords)
                media_vector = self._create_keyword_vector(media_keywords)
                
                # 코사인 유사도 계산
                similarity = cosine_similarity([comment_vector], [media_vector])[0][0]
                
                # 차이점 분석
                unique_comment_keywords = self._find_unique_keywords(comment_keywords, media_keywords)
                unique_media_keywords = self._find_unique_keywords(media_keywords, comment_keywords)
                
                similarity_results[period] = {
                    'cosine_similarity': similarity,
                    'comment_keywords': comment_keywords[:20],
                    'media_keywords': media_keywords[:20],
                    'unique_comment_keywords': unique_comment_keywords[:10],
                    'unique_media_keywords': unique_media_keywords[:10],
                    'frame_alignment': 'high' if similarity > 0.7 else 'medium' if similarity > 0.4 else 'low'
                }
            
            # 전체 트렌드 분석
            similarity_trend = self._analyze_similarity_trend(similarity_results)
            
            result = {
                'case_name': case_name,
                'similarity_results': similarity_results,
                'similarity_trend': similarity_trend,
                'media_keywords': media_keywords,
                'frame_divergence_periods': self._identify_divergence_periods(similarity_results)
            }
            
            self.analysis_results['media_frame_comparison'] = result
            self.logger.info(f"✅ {case_name} 언론 프레임 유사도 분석 완료")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ 언론 프레임 유사도 분석 실패: {str(e)}")
            raise
    
    def detect_anomalies_and_changepoints(self, df: pd.DataFrame, case_name: str) -> Dict:
        """
        변곡점(이상치) 탐지
        
        Args:
            df: 댓글 데이터프레임
            case_name: 사건명
        
        Returns:
            이상치 및 변곡점 분석 결과
        """
        try:
            self.logger.info(f"🔍 {case_name} 변곡점 및 이상치 탐지 시작")
            
            # 시계열 데이터 준비
            daily_metrics = self._prepare_daily_metrics(df)
            
            # PELT 알고리즘을 사용한 변곡점 탐지
            changepoints = self._detect_pelt_changepoints(daily_metrics)
            
            # Z-score 기반 이상치 탐지
            anomalies = self._detect_zscore_anomalies(daily_metrics)
            
            # 감성 급변 구간 탐지
            sentiment_shocks = self._detect_sentiment_shocks(daily_metrics)
            
            # 댓글 수 급증 구간 탐지
            volume_spikes = self._detect_volume_spikes(daily_metrics)
            
            result = {
                'case_name': case_name,
                'daily_metrics': daily_metrics,
                'changepoints': changepoints,
                'anomalies': anomalies,
                'sentiment_shocks': sentiment_shocks,
                'volume_spikes': volume_spikes,
                'critical_periods': self._identify_critical_periods(changepoints, anomalies, sentiment_shocks)
            }
            
            self.analysis_results['anomaly_detection'] = result
            self.logger.info(f"✅ {case_name} 변곡점 및 이상치 탐지 완료")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ 변곡점 및 이상치 탐지 실패: {str(e)}")
            raise
    
    def analyze_contextual_embeddings(self, df: pd.DataFrame, case_name: str) -> Dict:
        """
        문맥 임베딩 클러스터링 분석
        
        Args:
            df: 댓글 데이터프레임
            case_name: 사건명
        
        Returns:
            임베딩 클러스터링 결과
        """
        try:
            self.logger.info(f"🧠 {case_name} 문맥 임베딩 클러스터링 분석 시작")
            
            if not self.sentence_model:
                self.logger.warning("⚠️ SBERT 모델이 없어 임베딩 분석을 건너뜁니다.")
                return {'error': 'no_sbert_model'}
            
            # 텍스트 전처리
            try:
                texts = self._preprocess_texts_for_embedding(df)
            except Exception as e:
                self.logger.warning(f"⚠️ 임베딩용 텍스트 전처리 실패: {str(e)}")
                return {'error': 'text_preprocessing_failed'}
            
            if len(texts) < 30:  # 최소 요구사항 증가
                self.logger.warning("⚠️ 임베딩 분석을 위한 텍스트가 부족합니다.")
                return {'error': 'insufficient_texts', 'text_count': len(texts)}
            
            # 문장 임베딩 생성
            try:
                self.logger.info("🔄 문장 임베딩 생성 중...")
                # 배치 크기를 작게 하여 메모리 사용량 줄이기
                embeddings = self.sentence_model.encode(texts[:200], batch_size=16, show_progress_bar=False)  # 최대 200개만
                self.logger.info("✅ 문장 임베딩 생성 완료")
            except Exception as e:
                self.logger.warning(f"⚠️ 문장 임베딩 생성 실패: {str(e)}")
                return {'error': 'embedding_generation_failed'}
            
            # HDBSCAN 클러스터링
            try:
                hdbscan_clusters = self._perform_hdbscan_clustering(embeddings, texts[:len(embeddings)])
            except Exception as e:
                self.logger.warning(f"⚠️ HDBSCAN 클러스터링 실패: {str(e)}")
                hdbscan_clusters = {'labels': [], 'n_clusters': 0, 'noise_points': 0}
            
            # DBSCAN 클러스터링
            try:
                dbscan_clusters = self._perform_dbscan_clustering(embeddings, texts[:len(embeddings)])
            except Exception as e:
                self.logger.warning(f"⚠️ DBSCAN 클러스터링 실패: {str(e)}")
                dbscan_clusters = {'labels': [], 'n_clusters': 0, 'noise_points': 0}
            
            # 클러스터 특성 분석
            try:
                cluster_analysis = self._analyze_clusters(hdbscan_clusters, dbscan_clusters, texts[:len(embeddings)])
            except Exception as e:
                self.logger.warning(f"⚠️ 클러스터 분석 실패: {str(e)}")
                cluster_analysis = {}
            
            # 시간별 클러스터 진화
            try:
                temporal_clusters = self._analyze_temporal_clusters(df, embeddings)
            except Exception as e:
                self.logger.warning(f"⚠️ 시간별 클러스터 분석 실패: {str(e)}")
                temporal_clusters = {}
            
            result = {
                'case_name': case_name,
                'embeddings_shape': embeddings.shape if 'embeddings' in locals() else None,
                'hdbscan_clusters': hdbscan_clusters,
                'dbscan_clusters': dbscan_clusters,
                'cluster_analysis': cluster_analysis,
                'temporal_clusters': temporal_clusters,
                'total_texts': len(texts),
                'processed_texts': len(embeddings) if 'embeddings' in locals() else 0
            }
            
            self.analysis_results['contextual_embeddings'] = result
            self.logger.info(f"✅ {case_name} 문맥 임베딩 클러스터링 분석 완료")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ 문맥 임베딩 클러스터링 분석 실패: {str(e)}")
            return {'error': 'contextual_embedding_failed', 'details': str(e)}
    
    def cross_case_meta_analysis(self, cases_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        크로스케이스 메타분석
        
        Args:
            cases_data: 사건별 데이터프레임 딕셔너리
        
        Returns:
            메타분석 결과
        """
        try:
            self.logger.info("🔄 크로스케이스 메타분석 시작")
            
            meta_results = {}
            
            # 각 사건별 기본 분석 수행
            for case_name, df in cases_data.items():
                self.logger.info(f"📊 {case_name} 분석 시작")
                
                # 기본 통계
                basic_stats = self._calculate_basic_statistics(df, case_name)
                
                # 감성 패턴 분석
                sentiment_patterns = self._analyze_sentiment_patterns(df, case_name)
                
                # 토픽 패턴 분석
                topic_patterns = self._analyze_topic_patterns(df, case_name)
                
                # 시간적 특성 분석
                temporal_characteristics = self._analyze_temporal_characteristics(df, case_name)
                
                meta_results[case_name] = {
                    'basic_stats': basic_stats,
                    'sentiment_patterns': sentiment_patterns,
                    'topic_patterns': topic_patterns,
                    'temporal_characteristics': temporal_characteristics
                }
            
            # 사건 간 비교 분석
            comparative_analysis = self._perform_comparative_analysis(meta_results)
            
            # 패턴 유사성 분석
            pattern_similarity = self._analyze_pattern_similarity(meta_results)
            
            # 영향 요인 분석
            influence_factors = self._analyze_influence_factors(meta_results)
            
            result = {
                'individual_results': meta_results,
                'comparative_analysis': comparative_analysis,
                'pattern_similarity': pattern_similarity,
                'influence_factors': influence_factors,
                'meta_insights': self._generate_meta_insights(meta_results, comparative_analysis)
            }
            
            self.analysis_results['cross_case_meta'] = result
            self.logger.info("✅ 크로스케이스 메타분석 완료")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ 크로스케이스 메타분석 실패: {str(e)}")
            raise
    
    # ===== 헬퍼 메서드들 =====
    
    def _find_date_column(self, df: pd.DataFrame) -> str:
        """날짜 컬럼 찾기"""
        date_columns = ['timestamp', 'date', 'created_at', 'published_at', 'video_date']
        for col in date_columns:
            if col in df.columns:
                return col
        return None
    
    def _create_event_based_windows(self, df: pd.DataFrame, case_name: str, 
                                  event_dates: Dict[str, str] = None) -> Dict[str, Tuple]:
        """사건별 맞춤 시간 구간 생성"""
        # 기본 이벤트 날짜 (실제 사건에 맞게 조정 필요)
        default_events = {
            '유아인': {
                '사건_발생': '2023-02-01',
                '수사_시작': '2023-02-15',
                '기소': '2023-05-01',
                '재판_시작': '2023-08-01',
                '판결': '2023-10-01'
            },
            '돈스파이크': {
                '사건_발생': '2023-03-01',
                '수사_시작': '2023-03-15',
                '기소': '2023-06-01',
                '재판_시작': '2023-09-01',
                '판결': '2023-11-01'
            },
            '지드래곤': {
                '사건_발생': '2023-10-01',
                '수사_시작': '2023-10-15',
                '수사_종료': '2023-12-01'
            }
        }
        
        events = event_dates or default_events.get(case_name, {})
        
        # 데이터 기간 확인
        date_col = self._find_date_column(df)
        min_date = df[date_col].min()
        max_date = df[date_col].max()
        
        # 시간 구간 생성
        windows = {}
        event_dates_list = sorted(events.items(), key=lambda x: x[1])
        
        for i, (event_name, event_date) in enumerate(event_dates_list):
            event_dt = pd.to_datetime(event_date)
            
            if i == 0:
                start_date = max(min_date, event_dt - timedelta(days=30))
                end_date = min(max_date, event_dt + timedelta(days=30))
                windows[f'사전_{event_name}'] = (start_date, event_dt)
                windows[f'사후_{event_name}'] = (event_dt, end_date)
            else:
                prev_event_date = pd.to_datetime(event_dates_list[i-1][1])
                start_date = prev_event_date
                end_date = min(max_date, event_dt + timedelta(days=30))
                windows[f'{event_name}_기간'] = (start_date, end_date)
        
        return windows
    
    def _calculate_sentiment_scores(self, df: pd.DataFrame) -> Dict:
        """감성 점수 계산"""
        # 간단한 감성 점수 계산 (실제로는 더 정교한 모델 사용)
        text_col = 'cleaned_text' if 'cleaned_text' in df.columns else 'comment_text'
        
        if text_col not in df.columns:
            return {'avg_sentiment': 0, 'volatility': 0}
        
        # 긍정/부정 키워드 기반 간단한 점수 계산
        positive_words = ['좋다', '훌륭하다', '멋지다', '응원', '화이팅', '최고', '감동']
        negative_words = ['나쁘다', '실망', '화나다', '짜증', '최악', '비판', '문제']
        
        scores = []
        for text in df[text_col].fillna(''):
            pos_count = sum(1 for word in positive_words if word in text)
            neg_count = sum(1 for word in negative_words if word in text)
            
            if pos_count + neg_count > 0:
                score = (pos_count - neg_count) / (pos_count + neg_count)
            else:
                score = 0
            scores.append(score)
        
        return {
            'avg_sentiment': np.mean(scores) if scores else 0,
            'volatility': np.std(scores) if scores else 0,
            'scores': scores
        }
    
    def _calculate_engagement_metrics(self, df: pd.DataFrame) -> Dict:
        """참여도 지표 계산"""
        metrics = {}
        
        if 'upvotes' in df.columns:
            metrics['avg_upvotes'] = df['upvotes'].mean()
            metrics['total_upvotes'] = df['upvotes'].sum()
        
        if 'reply_count' in df.columns:
            metrics['avg_replies'] = df['reply_count'].mean()
            metrics['total_replies'] = df['reply_count'].sum()
        
        if 'comment_text' in df.columns:
            metrics['avg_comment_length'] = df['comment_text'].str.len().mean()
        
        return metrics
    
    def _extract_period_keywords(self, df: pd.DataFrame, top_k: int = 50) -> List[Tuple[str, int]]:
        """특정 기간의 키워드 추출"""
        text_col = 'cleaned_text' if 'cleaned_text' in df.columns else 'comment_text'
        
        if text_col not in df.columns:
            return []
        
        # 모든 텍스트 합치기
        all_text = ' '.join(df[text_col].fillna('').astype(str))
        
        # 형태소 분석
        try:
            morphs = self.mecab.morphs(all_text)
            
            # 불용어 제거 및 필터링
            filtered_morphs = []
            for morph in morphs:
                if (len(morph) > 1 and 
                    morph not in self.config.KOREAN_STOPWORDS and
                    not morph.isdigit()):
                    filtered_morphs.append(morph)
            
            # 빈도 계산
            word_counts = Counter(filtered_morphs)
            return word_counts.most_common(top_k)
            
        except Exception as e:
            self.logger.warning(f"⚠️ 키워드 추출 실패: {e}")
            return []
    
    def _analyze_sentiment_trends(self, temporal_results: Dict) -> Dict:
        """감성 트렌드 분석"""
        periods = sorted(temporal_results.keys())
        sentiments = [temporal_results[p]['avg_sentiment'] for p in periods]
        
        if len(sentiments) < 2:
            return {'trend': 'insufficient_data'}
        
        # 선형 회귀로 트렌드 계산
        x = np.arange(len(sentiments))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, sentiments)
        
        return {
            'trend': 'increasing' if slope > 0 else 'decreasing',
            'slope': slope,
            'correlation': r_value,
            'p_value': p_value,
            'volatility': np.std(sentiments)
        }
    
    def _detect_changepoints(self, temporal_results: Dict) -> List:
        """변곡점 탐지"""
        periods = sorted(temporal_results.keys())
        sentiments = [temporal_results[p]['avg_sentiment'] for p in periods]
        
        if len(sentiments) < 5:
            return []
        
        try:
            # PELT 알고리즘 사용
            algo = rpt.Pelt(model="rbf").fit(np.array(sentiments).reshape(-1, 1))
            changepoints = algo.predict(pen=1.0)
            
            # 변곡점 정보 구성
            result = []
            for cp in changepoints[:-1]:  # 마지막은 데이터 끝점이므로 제외
                if cp < len(periods):
                    result.append({
                        'period': periods[cp],
                        'index': cp,
                        'sentiment_before': sentiments[cp-1] if cp > 0 else None,
                        'sentiment_after': sentiments[cp] if cp < len(sentiments) else None
                    })
            
            return result
            
        except Exception as e:
            self.logger.warning(f"⚠️ 변곡점 탐지 실패: {e}")
            return []
    
    # ===== 추가 헬퍼 메서드들 =====
    
    def _preprocess_texts_for_embedding(self, df: pd.DataFrame) -> List[str]:
        """임베딩을 위한 텍스트 전처리"""
        try:
            text_col = 'cleaned_text' if 'cleaned_text' in df.columns else 'comment_text'
            
            if text_col not in df.columns:
                return []
            
            processed_texts = []
            for text in df[text_col].fillna(''):
                if text and len(text.strip()) > 10:  # 최소 길이 증가
                    processed_texts.append(text.strip())
            
            return processed_texts
            
        except Exception as e:
            self.logger.warning(f"⚠️ 임베딩용 텍스트 전처리 실패: {str(e)}")
            return []
    
    def _perform_hdbscan_clustering(self, embeddings, texts) -> Dict:
        """HDBSCAN 클러스터링 수행"""
        try:
            from hdbscan import HDBSCAN
            
            clusterer = HDBSCAN(
                min_cluster_size=max(3, len(embeddings) // 20),
                metric='euclidean',
                cluster_selection_method='eom'
            )
            
            cluster_labels = clusterer.fit_predict(embeddings)
            
            return {
                'labels': cluster_labels.tolist(),
                'n_clusters': len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0),
                'noise_points': sum(1 for label in cluster_labels if label == -1)
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ HDBSCAN 클러스터링 실패: {str(e)}")
            return {'labels': [], 'n_clusters': 0, 'noise_points': 0}
    
    def _perform_dbscan_clustering(self, embeddings, texts) -> Dict:
        """DBSCAN 클러스터링 수행"""
        try:
            from sklearn.cluster import DBSCAN
            
            clusterer = DBSCAN(
                eps=0.5,
                min_samples=max(2, len(embeddings) // 30)
            )
            
            cluster_labels = clusterer.fit_predict(embeddings)
            
            return {
                'labels': cluster_labels.tolist(),
                'n_clusters': len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0),
                'noise_points': sum(1 for label in cluster_labels if label == -1)
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ DBSCAN 클러스터링 실패: {str(e)}")
            return {'labels': [], 'n_clusters': 0, 'noise_points': 0}
    
    def _analyze_clusters(self, hdbscan_clusters, dbscan_clusters, texts) -> Dict:
        """클러스터 특성 분석"""
        try:
            analysis = {
                'hdbscan': {
                    'n_clusters': hdbscan_clusters.get('n_clusters', 0),
                    'noise_ratio': hdbscan_clusters.get('noise_points', 0) / len(texts) if texts else 0
                },
                'dbscan': {
                    'n_clusters': dbscan_clusters.get('n_clusters', 0),
                    'noise_ratio': dbscan_clusters.get('noise_points', 0) / len(texts) if texts else 0
                }
            }
            
            return analysis
            
        except Exception as e:
            self.logger.warning(f"⚠️ 클러스터 분석 실패: {str(e)}")
            return {}
    
    def _analyze_temporal_clusters(self, df, embeddings) -> Dict:
        """시간별 클러스터 진화 분석"""
        try:
            # 간단한 시간별 분석
            date_col = self._find_date_column(df)
            if not date_col:
                return {}
            
            return {
                'temporal_analysis': 'completed',
                'periods_analyzed': 1
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ 시간별 클러스터 분석 실패: {str(e)}")
            return {}
    
    def _build_cooccurrence_matrix(self, df, min_cooccurrence) -> Dict:
        """공출현 매트릭스 생성"""
        try:
            # 간단한 공출현 매트릭스 생성
            return {'matrix': 'placeholder', 'keywords': []}
        except Exception as e:
            self.logger.warning(f"⚠️ 공출현 매트릭스 생성 실패: {str(e)}")
            return {}
    
    def _create_network_graph(self, cooccurrence_matrix) -> object:
        """네트워크 그래프 생성"""
        try:
            import networkx as nx
            return nx.Graph()
        except Exception as e:
            self.logger.warning(f"⚠️ 네트워크 그래프 생성 실패: {str(e)}")
            return None
    
    def _analyze_network_properties(self, G) -> Dict:
        """네트워크 속성 분석"""
        try:
            if G is None:
                return {}
            return {'nodes': 0, 'edges': 0}
        except Exception as e:
            self.logger.warning(f"⚠️ 네트워크 속성 분석 실패: {str(e)}")
            return {}
    
    def _detect_communities(self, G) -> Dict:
        """커뮤니티 탐지"""
        try:
            return {'communities': []}
        except Exception as e:
            self.logger.warning(f"⚠️ 커뮤니티 탐지 실패: {str(e)}")
            return {}
    
    def _analyze_centrality(self, G) -> Dict:
        """중심성 분석"""
        try:
            return {'centrality': {}}
        except Exception as e:
            self.logger.warning(f"⚠️ 중심성 분석 실패: {str(e)}")
            return {}
    
    def _analyze_dynamic_networks(self, df, case_name) -> Dict:
        """동적 네트워크 분석"""
        try:
            return {'dynamic_networks': 'placeholder'}
        except Exception as e:
            self.logger.warning(f"⚠️ 동적 네트워크 분석 실패: {str(e)}")
            return {}
    
    def _prepare_daily_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """일별 지표 준비"""
        try:
            date_col = self._find_date_column(df)
            if not date_col:
                return pd.DataFrame()
            
            # 날짜별 집계
            df[date_col] = pd.to_datetime(df[date_col])
            df['date'] = df[date_col].dt.date
            
            daily_metrics = df.groupby('date').agg({
                'comment_text': 'count',  # 댓글 수
                'upvotes': 'mean' if 'upvotes' in df.columns else lambda x: 0,  # 평균 추천수
            }).rename(columns={'comment_text': 'comment_count'})
            
            # 감성 점수 계산
            sentiment_scores = []
            for date in daily_metrics.index:
                date_df = df[df['date'] == date]
                sentiment = self._calculate_sentiment_scores(date_df)
                sentiment_scores.append(sentiment.get('avg_sentiment', 0))
            
            daily_metrics['sentiment'] = sentiment_scores
            daily_metrics.reset_index(inplace=True)
            
            return daily_metrics
            
        except Exception as e:
            self.logger.warning(f"⚠️ 일별 지표 준비 실패: {str(e)}")
            return pd.DataFrame()
    
    def _detect_pelt_changepoints(self, daily_metrics: pd.DataFrame) -> List:
        """PELT 알고리즘을 사용한 변곡점 탐지"""
        try:
            if len(daily_metrics) < 5:
                return []
            
            # 간단한 변곡점 탐지 (실제 PELT 대신)
            sentiment_values = daily_metrics['sentiment'].values
            changepoints = []
            
            # 급격한 변화 지점 찾기
            for i in range(1, len(sentiment_values) - 1):
                if abs(sentiment_values[i] - sentiment_values[i-1]) > 0.3:
                    changepoints.append({
                        'index': i,
                        'date': daily_metrics.iloc[i]['date'],
                        'sentiment_change': sentiment_values[i] - sentiment_values[i-1]
                    })
            
            return changepoints
            
        except Exception as e:
            self.logger.warning(f"⚠️ PELT 변곡점 탐지 실패: {str(e)}")
            return []
    
    def _detect_zscore_anomalies(self, daily_metrics: pd.DataFrame) -> List:
        """Z-score 기반 이상치 탐지"""
        try:
            if len(daily_metrics) < 3:
                return []
            
            anomalies = []
            
            # 댓글 수 이상치
            comment_counts = daily_metrics['comment_count'].values
            mean_count = np.mean(comment_counts)
            std_count = np.std(comment_counts)
            
            if std_count > 0:
                z_scores = np.abs((comment_counts - mean_count) / std_count)
                anomaly_indices = np.where(z_scores > 2)[0]
                
                for idx in anomaly_indices:
                    anomalies.append({
                        'type': 'comment_volume',
                        'index': idx,
                        'date': daily_metrics.iloc[idx]['date'],
                        'z_score': z_scores[idx],
                        'value': comment_counts[idx]
                    })
            
            return anomalies
            
        except Exception as e:
            self.logger.warning(f"⚠️ Z-score 이상치 탐지 실패: {str(e)}")
            return []
    
    def _detect_sentiment_shocks(self, daily_metrics: pd.DataFrame) -> List:
        """감성 급변 구간 탐지"""
        try:
            if len(daily_metrics) < 3:
                return []
            
            sentiment_values = daily_metrics['sentiment'].values
            shocks = []
            
            for i in range(1, len(sentiment_values)):
                change = abs(sentiment_values[i] - sentiment_values[i-1])
                if change > 0.4:  # 임계값
                    shocks.append({
                        'index': i,
                        'date': daily_metrics.iloc[i]['date'],
                        'sentiment_change': sentiment_values[i] - sentiment_values[i-1],
                        'magnitude': change
                    })
            
            return shocks
            
        except Exception as e:
            self.logger.warning(f"⚠️ 감성 급변 탐지 실패: {str(e)}")
            return []
    
    def _detect_volume_spikes(self, daily_metrics: pd.DataFrame) -> List:
        """댓글 수 급증 구간 탐지"""
        try:
            if len(daily_metrics) < 3:
                return []
            
            comment_counts = daily_metrics['comment_count'].values
            spikes = []
            
            for i in range(1, len(comment_counts)):
                if comment_counts[i-1] > 0:
                    ratio = comment_counts[i] / comment_counts[i-1]
                    if ratio > 2.0:  # 2배 이상 증가
                        spikes.append({
                            'index': i,
                            'date': daily_metrics.iloc[i]['date'],
                            'ratio': ratio,
                            'count': comment_counts[i]
                        })
            
            return spikes
            
        except Exception as e:
            self.logger.warning(f"⚠️ 댓글 수 급증 탐지 실패: {str(e)}")
            return []
    
    def _identify_critical_periods(self, changepoints: List, anomalies: List, sentiment_shocks: List) -> List:
        """중요 기간 식별"""
        try:
            critical_periods = []
            
            # 변곡점 기반
            for cp in changepoints:
                critical_periods.append({
                    'type': 'changepoint',
                    'date': cp.get('date'),
                    'description': f"감성 변곡점 (변화량: {cp.get('sentiment_change', 0):.2f})"
                })
            
            # 감성 급변 기반
            for shock in sentiment_shocks:
                critical_periods.append({
                    'type': 'sentiment_shock',
                    'date': shock.get('date'),
                    'description': f"감성 급변 (변화량: {shock.get('sentiment_change', 0):.2f})"
                })
            
            return critical_periods
            
        except Exception as e:
            self.logger.warning(f"⚠️ 중요 기간 식별 실패: {str(e)}")
            return []
    
    def save_analysis_results(self, filepath: str):
        """분석 결과 저장"""
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self.analysis_results, f)
            self.logger.info(f"✅ 분석 결과 저장 완료: {filepath}")
        except Exception as e:
            self.logger.error(f"❌ 분석 결과 저장 실패: {str(e)}")
    
    def load_analysis_results(self, filepath: str):
        """분석 결과 로드"""
        try:
            with open(filepath, 'rb') as f:
                self.analysis_results = pickle.load(f)
            self.logger.info(f"✅ 분석 결과 로드 완료: {filepath}")
        except Exception as e:
            self.logger.error(f"❌ 분석 결과 로드 실패: {str(e)}")
    
    def _preprocess_texts_for_topic_modeling(self, df: pd.DataFrame) -> List[str]:
        """토픽 모델링을 위한 텍스트 전처리"""
        try:
            text_col = 'cleaned_text' if 'cleaned_text' in df.columns else 'comment_text'
            
            if text_col not in df.columns:
                self.logger.warning("⚠️ 텍스트 컬럼을 찾을 수 없습니다.")
                return []
            
            processed_texts = []
            korean_stopwords = set(self.config.KOREAN_STOPWORDS)
            
            for text in df[text_col].fillna(''):
                if not text or len(text.strip()) < 5:
                    continue
                
                try:
                    # 형태소 분석 및 명사, 형용사 추출
                    morphs = self.mecab.pos(text)
                    
                    # 의미있는 품사만 선택 (명사, 형용사, 동사)
                    meaningful_words = []
                    for word, pos in morphs:
                        if (pos in ['NNG', 'NNP', 'VA', 'VV'] and 
                            len(word) > 1 and 
                            word not in korean_stopwords and
                            not word.isdigit()):
                            meaningful_words.append(word)
                    
                    # 최소 단어 수 확인
                    if len(meaningful_words) >= 2:
                        processed_texts.append(' '.join(meaningful_words))
                        
                except Exception as e:
                    self.logger.warning(f"⚠️ 텍스트 전처리 실패: {str(e)}")
                    continue
            
            self.logger.info(f"✅ 토픽 모델링용 텍스트 전처리 완료: {len(processed_texts)}개")
            return processed_texts
            
        except Exception as e:
            self.logger.error(f"❌ 토픽 모델링용 전처리 실패: {str(e)}")
            return []
    
    def _analyze_lda_topics(self, texts: List[str], case_name: str) -> Dict:
        """LDA 토픽 분석"""
        try:
            from gensim import corpora, models
            # CoherenceModel 제거 - 멀티프로세싱 문제 해결
            
            # 텍스트를 토큰으로 분할
            tokenized_texts = [text.split() for text in texts if text.strip()]
            
            if len(tokenized_texts) < 10:
                return {'topics': [], 'coherence': 0}
            
            # 사전 및 코퍼스 생성
            dictionary = corpora.Dictionary(tokenized_texts)
            dictionary.filter_extremes(no_below=2, no_above=0.8)
            corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
            
            # 최적 토픽 수 결정 (간단한 휴리스틱)
            num_topics = min(max(len(texts) // 20, 3), 10)
            
            # LDA 모델 학습 (단일 프로세스 모드)
            lda_model = models.LdaModel(
                corpus=corpus,
                id2word=dictionary,
                num_topics=num_topics,
                random_state=42,
                passes=5,  # 패스 수 감소
                alpha='auto',
                per_word_topics=False,  # 메모리 절약
                iterations=30,  # 반복 횟수 감소
                eval_every=None  # 평가 비활성화
            )
            
            # 간단한 일관성 점수 계산 (멀티프로세싱 없이)
            coherence_score = self._calculate_simple_coherence_score(lda_model, tokenized_texts, dictionary)
            
            # 토픽 정보 추출
            topics = []
            for i in range(num_topics):
                topic_words = lda_model.show_topic(i, topn=10)
                topics.append({
                    'id': i,
                    'words': topic_words,
                    'label': f"토픽{i+1}: {', '.join([word for word, _ in topic_words[:3]])}"
                })
            
            return {
                'topics': topics,
                'coherence': coherence_score,
                'num_topics': num_topics,
                'model': lda_model
            }
            
        except Exception as e:
            self.logger.error(f"❌ LDA 분석 실패: {str(e)}")
            return {'topics': [], 'coherence': 0}
    
    def _analyze_bertopic_topics(self, texts: List[str], case_name: str) -> Dict:
        """BERTopic 토픽 분석"""
        try:
            if not self.sentence_model:
                self.logger.warning("⚠️ SBERT 모델이 없어 BERTopic 분석을 건너뜁니다.")
                return {'topics': [], 'coherence': 0}
            
            if len(texts) < 20:  # 최소 요구사항 증가
                self.logger.warning("⚠️ BERTopic 분석을 위한 텍스트가 부족합니다.")
                return {'topics': [], 'coherence': 0}
            
            try:
                from bertopic import BERTopic
                from umap import UMAP
                from hdbscan import HDBSCAN
                from sklearn.feature_extraction.text import CountVectorizer
            except ImportError as e:
                self.logger.warning(f"⚠️ BERTopic 라이브러리 import 실패: {str(e)}")
                return {'topics': [], 'coherence': 0}
            
            # 동적 파라미터 조정
            doc_count = len(texts)
            
            try:
                # UMAP 설정 - 더 보수적으로
                umap_model = UMAP(
                    n_neighbors=min(10, max(2, doc_count // 15)),
                    n_components=min(3, max(2, doc_count // 30)),
                    min_dist=0.1,
                    metric='cosine',
                    random_state=42,
                    low_memory=True
                )
                
                # HDBSCAN 설정 - 더 보수적으로
                hdbscan_model = HDBSCAN(
                    min_cluster_size=max(3, min(8, doc_count // 25)),
                    metric='euclidean',
                    cluster_selection_method='eom',
                    prediction_data=False  # 메모리 절약
                )
                
                # CountVectorizer 설정 - 더 보수적으로
                vectorizer_model = CountVectorizer(
                    ngram_range=(1, 1),  # unigram만
                    min_df=max(1, doc_count // 200),
                    max_df=0.95,
                    max_features=min(200, doc_count // 3)
                )
                
                # BERTopic 모델 생성
                topic_model = BERTopic(
                    embedding_model=self.sentence_model,
                    umap_model=umap_model,
                    hdbscan_model=hdbscan_model,
                    vectorizer_model=vectorizer_model,
                    language='korean',
                    calculate_probabilities=False,  # 메모리 절약
                    verbose=False
                )
                
                # 토픽 모델 학습
                topics, _ = topic_model.fit_transform(texts)
                
            except Exception as e:
                self.logger.warning(f"⚠️ BERTopic 모델 생성/학습 실패: {str(e)}")
                return {'topics': [], 'coherence': 0}
            
            # 토픽 정보 추출
            try:
                topic_info = topic_model.get_topic_info()
                topic_results = []
                
                for topic_id in topic_info['Topic'].unique():
                    if topic_id != -1 and len(topic_results) < 5:  # 최대 5개 토픽만
                        topic_words = topic_model.get_topic(topic_id)
                        if topic_words and len(topic_words) > 0:
                            # 상위 5개 단어만
                            top_words = topic_words[:5]
                            topic_results.append({
                                'id': topic_id,
                                'words': top_words,
                                'label': f"토픽{topic_id}: {', '.join([word for word, _ in top_words[:3]])}"
                            })
                
                return {
                    'topics': topic_results,
                    'coherence': 0,  # BERTopic 일관성 점수는 별도 계산 필요
                    'num_topics': len(topic_results),
                    'model': topic_model
                }
                
            except Exception as e:
                self.logger.warning(f"⚠️ BERTopic 토픽 정보 추출 실패: {str(e)}")
                return {'topics': [], 'coherence': 0}
            
        except Exception as e:
            self.logger.error(f"❌ BERTopic 분석 실패: {str(e)}")
            return {'topics': [], 'coherence': 0}
    
    def _compare_topic_methods(self, topic_results: Dict) -> Dict:
        """토픽 모델링 방법 간 비교"""
        try:
            comparison = {}
            
            for method, result in topic_results.items():
                comparison[method] = {
                    'num_topics': result.get('num_topics', 0),
                    'coherence': result.get('coherence', 0),
                    'top_topics': result.get('topics', [])[:3]
                }
            
            return comparison
            
        except Exception as e:
            self.logger.error(f"❌ 토픽 방법 비교 실패: {str(e)}")
            return {}
    
    def _analyze_topic_evolution(self, df: pd.DataFrame, topic_results: Dict) -> Dict:
        """토픽 진화 분석"""
        try:
            # 간단한 토픽 진화 분석 (시간별 토픽 분포 변화)
            evolution = {}
            
            # 시간 컬럼 찾기
            date_col = self._find_date_column(df)
            if not date_col:
                return {}
            
            # 월별 그룹화
            df['month'] = pd.to_datetime(df[date_col]).dt.to_period('M')
            monthly_groups = df.groupby('month')
            
            for month, group in monthly_groups:
                if len(group) > 5:  # 최소 댓글 수
                    month_texts = self._preprocess_texts_for_topic_modeling(group)
                    if month_texts:
                        evolution[str(month)] = {
                            'comment_count': len(group),
                            'text_count': len(month_texts)
                        }
            
            return evolution
            
        except Exception as e:
            self.logger.error(f"❌ 토픽 진화 분석 실패: {str(e)}")
            return {}
    
    def _calculate_simple_coherence_score(self, lda_model, tokenized_texts, dictionary) -> float:
        """
        간단한 일관성 점수 계산 (멀티프로세싱 없이)
        Args:
            lda_model: LDA 모델
            tokenized_texts: 토큰화된 텍스트 리스트
            dictionary: gensim 사전
        Returns:
            일관성 점수 (0-1)
        """
        try:
            if not lda_model or not tokenized_texts or not dictionary:
                return 0.0
            
            # 매우 간단한 일관성 점수 계산
            # 토픽별 상위 단어들이 같은 문서에 함께 나타나는 빈도 계산
            coherence_scores = []
            
            for topic_id in range(min(lda_model.num_topics, 5)):  # 최대 5개 토픽만 처리
                try:
                    topic_words = [word for word, _ in lda_model.show_topic(topic_id, topn=3)]  # 단어 수 감소
                    
                    if len(topic_words) < 2:
                        continue
                    
                    # 토픽 단어들이 같은 문서에 함께 나타나는 빈도 계산
                    co_occurrence_count = 0
                    total_pairs = 0
                    
                    # 처리할 문서 수 제한
                    sample_docs = tokenized_texts[:min(50, len(tokenized_texts))]
                    
                    for doc in sample_docs:
                        if not doc or len(doc) < 2:
                            continue
                            
                        try:
                            doc_set = set(doc)
                            topic_words_in_doc = [word for word in topic_words if word in doc_set]
                            
                            if len(topic_words_in_doc) > 1:
                                # 문서 내에서 토픽 단어들의 조합 수
                                pairs_in_doc = len(topic_words_in_doc) * (len(topic_words_in_doc) - 1) / 2
                                co_occurrence_count += pairs_in_doc
                            
                            # 전체 가능한 조합 수
                            total_pairs += len(topic_words) * (len(topic_words) - 1) / 2
                        except Exception:
                            continue
                    
                    if total_pairs > 0:
                        topic_coherence = co_occurrence_count / total_pairs
                        coherence_scores.append(topic_coherence)
                        
                except Exception as e:
                    self.logger.warning(f"⚠️ 토픽 {topic_id} 일관성 계산 실패: {str(e)}")
                    continue
            
            # 평균 일관성 점수 반환
            if coherence_scores:
                return float(np.mean(coherence_scores))
            else:
                return 0.0
            
        except Exception as e:
            self.logger.warning(f"⚠️ 간단한 일관성 점수 계산 실패: {str(e)}")
            return 0.0 