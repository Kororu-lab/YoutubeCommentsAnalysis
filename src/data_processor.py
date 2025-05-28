"""
Data Processor Module
데이터 전처리 모듈
"""

import pandas as pd
import numpy as np
import re
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import pickle
import os

class DataProcessor:
    """유튜브 댓글 데이터 전처리 클래스"""
    
    def __init__(self, config):
        """
        초기화
        Args:
            config: AnalysisConfig 객체
        """
        self.config = config
        self.logger = self._setup_logger()
        self.raw_data = None
        self.processed_data = {}
        
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
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        데이터 로드
        Args:
            file_path: CSV 파일 경로
        Returns:
            로드된 데이터프레임
        """
        try:
            self.logger.info(f"📂 데이터 로드 시작: {file_path}")
            
            # CSV 파일 로드 (다양한 인코딩 시도)
            encodings = ['utf-8', 'cp949', 'euc-kr']
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    self.logger.info(f"✅ 데이터 로드 성공 (인코딩: {encoding})")
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError("지원되는 인코딩으로 파일을 읽을 수 없습니다.")
            
            # 기본 정보 출력
            self.logger.info(f"📊 총 댓글 수: {len(df):,}")
            self.logger.info(f"📊 컬럼: {list(df.columns)}")
            
            self.raw_data = df
            return df
            
        except Exception as e:
            self.logger.error(f"❌ 데이터 로드 실패: {str(e)}")
            raise
    
    def clean_text(self, text: str) -> str:
        """
        텍스트 정리 및 불용어 제거
        Args:
            text: 원본 텍스트
        Returns:
            정리된 텍스트
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # 기본 정리
        text = str(text).strip()
        
        # URL 제거
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # 이메일 제거
        text = re.sub(r'\S+@\S+', '', text)
        
        # 특수문자 정리 (한글, 영문, 숫자, 기본 문장부호만 유지)
        text = re.sub(r'[^\w\s가-힣ㄱ-ㅎㅏ-ㅣ.,!?]', ' ', text)
        
        # 연속된 공백 제거
        text = re.sub(r'\s+', ' ', text)
        
        # 연속된 문장부호 제거
        text = re.sub(r'[.,!?]{2,}', '.', text)
        
        return text.strip()
    
    def remove_stopwords(self, text: str) -> str:
        """
        불용어 제거
        Args:
            text: 입력 텍스트
        Returns:
            불용어가 제거된 텍스트
        """
        if not text:
            return ""
        
        # 한국어 형태소 분석을 위한 간단한 토큰화
        words = text.split()
        
        # 불용어 제거
        stopwords = set(self.config.KOREAN_STOPWORDS)
        filtered_words = []
        
        for word in words:
            # 단어 정리
            word = word.strip('.,!?()[]{}"\'-')
            
            # 길이가 1인 단어 제거 (조사 등)
            if len(word) <= 1:
                continue
                
            # 불용어 체크
            if word.lower() not in stopwords and word not in stopwords:
                # 숫자만으로 이루어진 단어 제거
                if not word.isdigit():
                    filtered_words.append(word)
        
        return ' '.join(filtered_words)
    
    def extract_target_comments(self, df: pd.DataFrame, target_name: str) -> pd.DataFrame:
        """
        특정 대상 관련 댓글 추출
        Args:
            df: 전체 데이터프레임
            target_name: 분석 대상 이름 (예: '유아인')
        Returns:
            해당 대상 관련 댓글 데이터프레임
        """
        try:
            self.logger.info(f"🎯 {target_name} 관련 댓글 추출 시작")
            
            target_config = self.config.TARGETS[target_name]
            keywords = target_config['keywords']
            
            # 키워드 매칭 (대소문자 무시)
            mask = pd.Series([False] * len(df))
            
            for keyword in keywords:
                # 댓글 내용에서 키워드 검색
                if 'comment_text' in df.columns:
                    mask |= df['comment_text'].str.contains(keyword, case=False, na=False)
                elif 'text' in df.columns:
                    mask |= df['text'].str.contains(keyword, case=False, na=False)
                elif 'content' in df.columns:
                    mask |= df['content'].str.contains(keyword, case=False, na=False)
                
                # 비디오 제목에서도 검색 (있는 경우)
                if 'video_title' in df.columns:
                    mask |= df['video_title'].str.contains(keyword, case=False, na=False)
                elif 'title' in df.columns:
                    mask |= df['title'].str.contains(keyword, case=False, na=False)
            
            target_df = df[mask].copy()
            
            self.logger.info(f"✅ {target_name} 관련 댓글 {len(target_df):,}개 추출 ({len(target_df)/len(df)*100:.2f}%)")
            
            return target_df
            
        except Exception as e:
            self.logger.error(f"❌ {target_name} 댓글 추출 실패: {str(e)}")
            raise
    
    def preprocess_comments(self, df: pd.DataFrame, target_name: str) -> pd.DataFrame:
        """
        댓글 전처리
        Args:
            df: 댓글 데이터프레임
            target_name: 분석 대상 이름
        Returns:
            전처리된 데이터프레임
        """
        try:
            self.logger.info(f"🔧 {target_name} 댓글 전처리 시작")
            
            processed_df = df.copy()
            
            # 댓글 텍스트 컬럼 통일
            text_columns = ['comment_text', 'text', 'content', 'comment']
            text_col = None
            for col in text_columns:
                if col in processed_df.columns:
                    text_col = col
                    break
            
            if text_col is None:
                raise ValueError("댓글 텍스트 컬럼을 찾을 수 없습니다.")
            
            # 텍스트 정리 및 불용어 제거
            processed_df['cleaned_text'] = processed_df[text_col].apply(self.clean_text)
            processed_df['filtered_text'] = processed_df['cleaned_text'].apply(self.remove_stopwords)
            
            # 빈 댓글 제거
            processed_df = processed_df[processed_df['cleaned_text'].str.len() > 0]
            
            # 길이 필터링
            min_len = self.config.ANALYSIS_PARAMS['min_comment_length']
            max_len = self.config.ANALYSIS_PARAMS['max_comment_length']
            
            processed_df = processed_df[
                (processed_df['cleaned_text'].str.len() >= min_len) &
                (processed_df['cleaned_text'].str.len() <= max_len)
            ]
            
            # 날짜 컬럼 처리
            date_columns = ['video_date', 'timestamp', 'published_at', 'date', 'created_at']
            date_col = None
            for col in date_columns:
                if col in processed_df.columns:
                    date_col = col
                    break
            
            if date_col:
                self.logger.info(f"📅 날짜 컬럼 사용: {date_col}")
                processed_df['date'] = pd.to_datetime(processed_df[date_col], errors='coerce')
                
                # 날짜 변환 실패한 행 확인
                invalid_dates = processed_df['date'].isna().sum()
                if invalid_dates > 0:
                    self.logger.warning(f"⚠️ 날짜 변환 실패: {invalid_dates:,}개 행")
                
                # 유효한 날짜가 있는 행만 유지
                before_date_filter = len(processed_df)
                processed_df = processed_df.dropna(subset=['date'])
                after_date_filter = len(processed_df)
                
                self.logger.info(f"📅 날짜 필터링: {before_date_filter:,} → {after_date_filter:,}개")
                
                # 월별 그룹 추가
                processed_df['year_month'] = processed_df['date'].dt.strftime('%Y-%m')
                
                # 날짜 범위 확인
                if len(processed_df) > 0:
                    min_date = processed_df['date'].min()
                    max_date = processed_df['date'].max()
                    self.logger.info(f"📅 날짜 범위: {min_date.strftime('%Y-%m-%d')} ~ {max_date.strftime('%Y-%m-%d')}")
            else:
                self.logger.warning("⚠️ 날짜 컬럼을 찾을 수 없습니다.")
                # 날짜가 없어도 분석할 수 있도록 기본 월 설정
                processed_df['year_month'] = '2024-01'
                processed_df['date'] = pd.to_datetime('2024-01-01')
            
            # 중복 제거
            initial_count = len(processed_df)
            processed_df = processed_df.drop_duplicates(subset=['cleaned_text'])
            final_count = len(processed_df)
            
            self.logger.info(f"🗑️ 중복 제거: {initial_count - final_count:,}개")
            self.logger.info(f"✅ {target_name} 전처리 완료: {final_count:,}개 댓글")
            
            return processed_df
            
        except Exception as e:
            self.logger.error(f"❌ {target_name} 전처리 실패: {str(e)}")
            raise
    
    def group_by_month(self, df: pd.DataFrame, target_name: str) -> Dict[str, pd.DataFrame]:
        """
        월별 데이터 그룹화
        Args:
            df: 전처리된 데이터프레임
            target_name: 분석 대상 이름
        Returns:
            월별 데이터프레임 딕셔너리
        """
        try:
            self.logger.info(f"📅 {target_name} 월별 그룹화 시작")
            
            if 'year_month' not in df.columns:
                raise ValueError("year_month 컬럼이 없습니다. 날짜 전처리를 먼저 수행하세요.")
            
            monthly_data = {}
            
            # 월별 그룹화
            for year_month, group in df.groupby('year_month'):
                if len(group) > 0:  # 빈 그룹 제외
                    monthly_data[year_month] = group.copy()
            
            # 월별 통계
            self.logger.info(f"📊 {target_name} 월별 댓글 분포:")
            for year_month in sorted(monthly_data.keys()):
                count = len(monthly_data[year_month])
                self.logger.info(f"  {year_month}: {count:,}개")
            
            return monthly_data
            
        except Exception as e:
            self.logger.error(f"❌ {target_name} 월별 그룹화 실패: {str(e)}")
            raise
    
    def calculate_video_relevance(self, df: pd.DataFrame, target_name: str) -> pd.DataFrame:
        """
        비디오 관련성 계산
        Args:
            df: 댓글 데이터프레임
            target_name: 분석 대상 이름
        Returns:
            관련성 정보가 추가된 데이터프레임
        """
        try:
            self.logger.info(f"🎬 {target_name} 비디오 관련성 계산 시작")
            
            target_config = self.config.TARGETS[target_name]
            keywords = target_config['keywords']
            
            # 비디오별 그룹화
            video_col = None
            for col in ['video_id', 'video_title', 'title']:
                if col in df.columns:
                    video_col = col
                    break
            
            if video_col is None:
                self.logger.warning("⚠️ 비디오 식별 컬럼을 찾을 수 없습니다.")
                df['video_relevance'] = 'unknown'
                return df
            
            df_with_relevance = df.copy()
            
            # 비디오별 관련성 계산
            for video_id, group in df.groupby(video_col):
                total_comments = len(group)
                
                # 키워드 포함 댓글 수 계산
                keyword_mentions = 0
                for keyword in keywords:
                    keyword_mentions += group['cleaned_text'].str.contains(keyword, case=False, na=False).sum()
                
                # 제목에서 키워드 확인
                title_relevance = False
                if 'video_title' in group.columns or 'title' in group.columns:
                    title_col = 'video_title' if 'video_title' in group.columns else 'title'
                    title = str(group[title_col].iloc[0]) if not group[title_col].empty else ""
                    for keyword in keywords:
                        if keyword.lower() in title.lower():
                            title_relevance = True
                            break
                
                # 관련성 점수 계산
                mention_ratio = keyword_mentions / total_comments if total_comments > 0 else 0
                
                # 관련성 분류
                if title_relevance or mention_ratio > 0.3:
                    relevance = 'high'
                elif mention_ratio > 0.1:
                    relevance = 'medium'
                elif mention_ratio > 0:
                    relevance = 'low'
                else:
                    relevance = 'none'
                
                # 데이터프레임에 관련성 정보 추가
                mask = df_with_relevance[video_col] == video_id
                df_with_relevance.loc[mask, 'video_relevance'] = relevance
                df_with_relevance.loc[mask, 'mention_ratio'] = mention_ratio
                df_with_relevance.loc[mask, 'title_relevance'] = title_relevance
            
            # 관련성 분포 로깅
            relevance_counts = df_with_relevance['video_relevance'].value_counts()
            self.logger.info(f"📊 {target_name} 비디오 관련성 분포:")
            for relevance, count in relevance_counts.items():
                self.logger.info(f"  {relevance}: {count:,}개 ({count/len(df_with_relevance)*100:.1f}%)")
            
            return df_with_relevance
            
        except Exception as e:
            self.logger.error(f"❌ {target_name} 비디오 관련성 계산 실패: {str(e)}")
            raise
    
    def process_target(self, target_name: str) -> Dict:
        """
        특정 대상에 대한 전체 전처리 파이프라인 실행
        Args:
            target_name: 분석 대상 이름
        Returns:
            처리된 데이터 딕셔너리
        """
        try:
            self.logger.info(f"🚀 {target_name} 전체 전처리 시작")
            
            if self.raw_data is None:
                raise ValueError("원본 데이터가 로드되지 않았습니다.")
            
            # 1. 대상 관련 댓글 추출
            target_comments = self.extract_target_comments(self.raw_data, target_name)
            
            # 2. 댓글 전처리
            processed_comments = self.preprocess_comments(target_comments, target_name)
            
            # 3. 비디오 관련성 계산
            comments_with_relevance = self.calculate_video_relevance(processed_comments, target_name)
            
            # 4. 월별 그룹화
            monthly_data = self.group_by_month(comments_with_relevance, target_name)
            
            # 결과 저장
            result = {
                'target_name': target_name,
                'total_comments': len(comments_with_relevance),
                'processed_data': comments_with_relevance,
                'monthly_data': monthly_data,
                'date_range': {
                    'start': comments_with_relevance['date'].min() if 'date' in comments_with_relevance.columns else None,
                    'end': comments_with_relevance['date'].max() if 'date' in comments_with_relevance.columns else None
                }
            }
            
            self.processed_data[target_name] = result
            
            self.logger.info(f"✅ {target_name} 전체 전처리 완료")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ {target_name} 전체 전처리 실패: {str(e)}")
            raise
    
    def save_processed_data(self, file_path: str):
        """
        전처리된 데이터 저장
        Args:
            file_path: 저장할 파일 경로
        """
        try:
            self.logger.info(f"💾 전처리 데이터 저장: {file_path}")
            
            with open(file_path, 'wb') as f:
                pickle.dump(self.processed_data, f)
            
            self.logger.info("✅ 전처리 데이터 저장 완료")
            
        except Exception as e:
            self.logger.error(f"❌ 전처리 데이터 저장 실패: {str(e)}")
            raise
    
    def load_processed_data(self, file_path: str):
        """
        전처리된 데이터 로드
        Args:
            file_path: 로드할 파일 경로
        """
        try:
            self.logger.info(f"📂 전처리 데이터 로드: {file_path}")
            
            with open(file_path, 'rb') as f:
                self.processed_data = pickle.load(f)
            
            self.logger.info("✅ 전처리 데이터 로드 완료")
            
        except Exception as e:
            self.logger.error(f"❌ 전처리 데이터 로드 실패: {str(e)}")
            raise
    
    def group_by_month_single(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        단일 파일 분석용 월별 그룹화
        Args:
            df: 전체 데이터프레임
        Returns:
            월별로 그룹화된 데이터 딕셔너리
        """
        try:
            self.logger.info("📅 단일 파일 월별 그룹화 시작")
            
            # 날짜 컬럼 찾기
            date_columns = ['video_date', 'timestamp', 'published_at', 'date', 'created_at']
            date_col = None
            for col in date_columns:
                if col in df.columns:
                    date_col = col
                    break
            
            if date_col is None:
                self.logger.warning("날짜 컬럼을 찾을 수 없습니다. 전체 데이터를 하나의 그룹으로 처리합니다.")
                return {'전체': df}
            
            # 날짜 변환
            df_copy = df.copy()
            df_copy['date'] = pd.to_datetime(df_copy[date_col], errors='coerce')
            
            # 유효한 날짜만 필터링
            df_copy = df_copy.dropna(subset=['date'])
            
            if len(df_copy) == 0:
                self.logger.warning("유효한 날짜가 있는 데이터가 없습니다.")
                return {'전체': df}
            
            # 월별 그룹화
            df_copy['year_month'] = df_copy['date'].dt.strftime('%Y-%m')
            monthly_groups = {}
            
            for month, group in df_copy.groupby('year_month'):
                monthly_groups[month] = group
                self.logger.info(f"📊 {month}: {len(group):,}개 댓글")
            
            self.logger.info(f"✅ 총 {len(monthly_groups)}개월 데이터 그룹화 완료")
            return monthly_groups
            
        except Exception as e:
            self.logger.error(f"❌ 월별 그룹화 실패: {str(e)}")
            return {'전체': df}
    
    def process_single_file(self, file_path: str) -> Dict:
        """
        단일 파일 전체 처리
        Args:
            file_path: CSV 파일 경로
        Returns:
            처리된 데이터 딕셔너리
        """
        try:
            self.logger.info("🚀 단일 파일 분석 시작")
            
            # 데이터 로드
            df = self.load_data(file_path)
            
            # 기본 전처리
            text_columns = ['comment_text', 'text', 'content', 'comment']
            text_col = None
            for col in text_columns:
                if col in df.columns:
                    text_col = col
                    break
            
            if text_col is None:
                raise ValueError("댓글 텍스트 컬럼을 찾을 수 없습니다.")
            
            # 텍스트 정리 및 불용어 제거
            df['cleaned_text'] = df[text_col].apply(self.clean_text)
            df['filtered_text'] = df['cleaned_text'].apply(self.remove_stopwords)
            
            # 빈 댓글 제거
            df = df[df['cleaned_text'].str.len() > 0]
            df = df[df['filtered_text'].str.len() > 0]
            
            # 길이 필터링
            min_len = self.config.ANALYSIS_PARAMS['min_comment_length']
            max_len = self.config.ANALYSIS_PARAMS['max_comment_length']
            
            df = df[
                (df['cleaned_text'].str.len() >= min_len) &
                (df['cleaned_text'].str.len() <= max_len)
            ]
            
            self.logger.info(f"📊 전처리 후 댓글 수: {len(df):,}")
            
            # 월별 그룹화
            monthly_data = self.group_by_month_single(df)
            
            # 결과 구성
            result = {
                'total_data': df,
                'monthly_data': monthly_data,
                'summary': {
                    'total_comments': len(df),
                    'months_count': len(monthly_data),
                    'date_range': self._get_date_range(df),
                    'avg_comments_per_month': len(df) / len(monthly_data) if monthly_data else 0
                }
            }
            
            self.logger.info("✅ 단일 파일 처리 완료")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ 단일 파일 처리 실패: {str(e)}")
            raise
    
    def _get_date_range(self, df: pd.DataFrame) -> Dict:
        """
        데이터의 날짜 범위 계산
        Args:
            df: 데이터프레임
        Returns:
            날짜 범위 정보
        """
        try:
            date_columns = ['video_date', 'timestamp', 'published_at', 'date', 'created_at']
            date_col = None
            for col in date_columns:
                if col in df.columns:
                    date_col = col
                    break
            
            if date_col is None:
                return {'start': None, 'end': None, 'span_days': 0}
            
            dates = pd.to_datetime(df[date_col], errors='coerce').dropna()
            
            if len(dates) == 0:
                return {'start': None, 'end': None, 'span_days': 0}
            
            start_date = dates.min()
            end_date = dates.max()
            span_days = (end_date - start_date).days
            
            return {
                'start': start_date.strftime('%Y-%m-%d'),
                'end': end_date.strftime('%Y-%m-%d'),
                'span_days': span_days
            }
            
        except Exception as e:
            self.logger.error(f"날짜 범위 계산 실패: {str(e)}")
            return {'start': None, 'end': None, 'span_days': 0} 