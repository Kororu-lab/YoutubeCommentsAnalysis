"""
Data Filter Module
데이터 필터링 모듈 - 댓글 품질 및 관련성 필터링
"""

import pandas as pd
import numpy as np
import logging
import re
from typing import Dict, List, Tuple, Optional
from collections import Counter
import difflib
from tqdm import tqdm

class DataFilter:
    """데이터 필터링 클래스"""
    
    def __init__(self, config):
        """
        초기화
        Args:
            config: AnalysisConfig 객체
        """
        self.config = config
        self.logger = self._setup_logger()
        self.filtering_config = config.COMMENT_FILTERING
        
        # 필터링 통계
        self.filter_stats = {
            'original_count': 0,
            'upvote_filtered': 0,
            'keyword_filtered': 0,
            'quality_filtered': 0,
            'duplicate_filtered': 0,
            'user_filtered': 0,
            'final_count': 0
        }
    
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
    
    def apply_all_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        모든 필터링 적용
        Args:
            df: 원본 데이터프레임
        Returns:
            필터링된 데이터프레임
        """
        try:
            self.logger.info(f"🔍 데이터 필터링 시작: {len(df):,}개 댓글")
            self.filter_stats['original_count'] = len(df)
            
            filtered_df = df.copy()
            
            # 1. 추천수 기반 필터링
            if self.filtering_config['upvote_filtering']['enabled']:
                filtered_df = self._apply_upvote_filter(filtered_df)
                self.filter_stats['upvote_filtered'] = self.filter_stats['original_count'] - len(filtered_df)
                self.logger.info(f"📊 추천수 필터링 후: {len(filtered_df):,}개 댓글 ({self.filter_stats['upvote_filtered']:,}개 제거)")
            
            # 2. 키워드 기반 필터링
            if self.filtering_config['keyword_filtering']['enabled']:
                before_count = len(filtered_df)
                filtered_df = self._apply_keyword_filter(filtered_df)
                self.filter_stats['keyword_filtered'] = before_count - len(filtered_df)
                self.logger.info(f"🔑 키워드 필터링 후: {len(filtered_df):,}개 댓글 ({self.filter_stats['keyword_filtered']:,}개 제거)")
            
            # 3. 품질 기반 필터링
            if self.filtering_config['quality_filtering']['enabled']:
                before_count = len(filtered_df)
                filtered_df = self._apply_quality_filter(filtered_df)
                self.filter_stats['quality_filtered'] = before_count - len(filtered_df)
                self.logger.info(f"✨ 품질 필터링 후: {len(filtered_df):,}개 댓글 ({self.filter_stats['quality_filtered']:,}개 제거)")
            
            # 4. 중복 댓글 필터링
            if self.filtering_config['duplicate_filtering']['enabled']:
                before_count = len(filtered_df)
                filtered_df = self._apply_duplicate_filter(filtered_df)
                self.filter_stats['duplicate_filtered'] = before_count - len(filtered_df)
                self.logger.info(f"🔄 중복 필터링 후: {len(filtered_df):,}개 댓글 ({self.filter_stats['duplicate_filtered']:,}개 제거)")
            
            # 5. 사용자 기반 필터링
            if self.filtering_config['user_filtering']['enabled']:
                before_count = len(filtered_df)
                filtered_df = self._apply_user_filter(filtered_df)
                self.filter_stats['user_filtered'] = before_count - len(filtered_df)
                self.logger.info(f"👤 사용자 필터링 후: {len(filtered_df):,}개 댓글 ({self.filter_stats['user_filtered']:,}개 제거)")
            
            self.filter_stats['final_count'] = len(filtered_df)
            
            # 필터링 결과 요약
            self._log_filtering_summary()
            
            return filtered_df
            
        except Exception as e:
            self.logger.error(f"❌ 데이터 필터링 실패: {str(e)}")
            raise
    
    def _apply_upvote_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        추천수 기반 필터링
        Args:
            df: 데이터프레임
        Returns:
            필터링된 데이터프레임
        """
        try:
            config = self.filtering_config['upvote_filtering']
            
            # upvotes 컬럼이 있는지 확인
            if 'upvotes' not in df.columns:
                self.logger.warning("⚠️ upvotes 컬럼이 없어 추천수 필터링을 건너뜁니다.")
                return df
            
            # NaN 값을 0으로 처리
            df['upvotes'] = df['upvotes'].fillna(0)
            
            # 최소 추천수 필터링
            mask = df['upvotes'] >= config['min_upvotes']
            
            # 순추천수 필터링 (downvotes 컬럼이 있는 경우)
            if 'downvotes' in df.columns and config['min_net_upvotes'] > 0:
                df['downvotes'] = df['downvotes'].fillna(0)
                net_upvotes = df['upvotes'] - df['downvotes']
                
                if config['use_net_upvotes_only_when_positive']:
                    # 순추천수가 양수인 경우만 필터링 적용
                    net_mask = (net_upvotes >= config['min_net_upvotes']) | (net_upvotes <= 0)
                else:
                    # 모든 경우에 순추천수 필터링 적용
                    net_mask = net_upvotes >= config['min_net_upvotes']
                
                mask = mask & net_mask
            
            filtered_df = df[mask].copy()
            
            self.logger.info(f"📊 추천수 필터링: 최소 {config['min_upvotes']}개 추천, "
                           f"최소 순추천 {config['min_net_upvotes']}개")
            
            return filtered_df
            
        except Exception as e:
            self.logger.error(f"❌ 추천수 필터링 실패: {str(e)}")
            return df
    
    def _apply_keyword_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        키워드 기반 필터링 (강화된 모드 지원)
        - basic 모드: 기존 방식 (영상 단위 포함 키워드 + 댓글 단위 제외 키워드)
        - enhanced 모드: 제목 포함 OR 댓글 비율 조건
        Args:
            df: 데이터프레임
        Returns:
            필터링된 데이터프레임
        """
        try:
            config = self.filtering_config['keyword_filtering']
            
            # 텍스트 컬럼 확인
            text_column = None
            for col in ['comment_text', 'cleaned_text', 'text', 'content']:
                if col in df.columns:
                    text_column = col
                    break
            
            if text_column is None:
                self.logger.warning("⚠️ 텍스트 컬럼을 찾을 수 없어 키워드 필터링을 건너뜁니다.")
                return df
            
            # NaN 값 처리
            df[text_column] = df[text_column].fillna('')
            
            # 필터링 모드에 따른 처리
            filtering_mode = config.get('filtering_mode', 'basic')
            
            if filtering_mode == 'enhanced':
                # 🔧 강화된 필터링 모드
                filtered_df = self._apply_enhanced_keyword_filter(df, text_column, config)
            else:
                # 기본 필터링 모드 (기존 방식)
                filtered_df = self._apply_basic_keyword_filter(df, text_column, config)
            
            # 2단계: 댓글 단위 제외 키워드 필터링 (공통)
            if config['excluded_keywords']:
                filtered_df = self._apply_comment_level_exclusion_filter(filtered_df, text_column, config)
                self.logger.info(f"🚫 댓글 단위 제외 키워드 필터링: {len(config['excluded_keywords'])}개 키워드")
            
            return filtered_df
            
        except Exception as e:
            self.logger.error(f"❌ 키워드 필터링 실패: {str(e)}")
            return df
    
    def _apply_video_level_keyword_filter(self, df: pd.DataFrame, text_column: str, config: Dict) -> pd.DataFrame:
        """
        영상 단위 필수 키워드 필터링
        영상 제목이나 해당 영상의 댓글 중 하나라도 키워드를 포함하면 그 영상의 모든 댓글을 포함
        """
        try:
            # 영상 식별 컬럼 확인
            video_id_column = None
            for col in ['video_no', 'video_id', 'video_url', 'url']:
                if col in df.columns:
                    video_id_column = col
                    break
            
            if video_id_column is None:
                self.logger.warning("⚠️ 영상 식별 컬럼을 찾을 수 없어 댓글 단위로 필터링합니다.")
                return self._apply_comment_level_keyword_filter(df, text_column, config)
            
            # 영상 제목 컬럼 확인
            title_column = None
            for col in ['video_title', 'title', 'video_name']:
                if col in df.columns:
                    title_column = col
                    break
            
            valid_videos = set()
            
            # 각 영상별로 키워드 포함 여부 확인
            for video_id, video_group in df.groupby(video_id_column):
                video_has_keyword = False
                
                # 1. 영상 제목에서 키워드 확인
                if title_column and not video_group[title_column].isna().all():
                    video_title = str(video_group[title_column].iloc[0])
                    if self._text_contains_keywords(video_title, config['required_keywords'], config):
                        video_has_keyword = True
                        self.logger.debug(f"📺 영상 제목에서 키워드 발견: {video_id}")
                
                # 2. 해당 영상의 댓글들에서 키워드 확인
                if not video_has_keyword:
                    for comment_text in video_group[text_column]:
                        if self._text_contains_keywords(str(comment_text), config['required_keywords'], config):
                            video_has_keyword = True
                            self.logger.debug(f"💬 영상 댓글에서 키워드 발견: {video_id}")
                            break
                
                # 키워드가 포함된 영상을 유효 목록에 추가
                if video_has_keyword:
                    valid_videos.add(video_id)
            
            # 유효한 영상의 모든 댓글 반환
            filtered_df = df[df[video_id_column].isin(valid_videos)].copy()
            
            self.logger.info(f"🎬 영상 단위 필터링 결과: {len(valid_videos)}개 영상의 {len(filtered_df)}개 댓글")
            
            return filtered_df
            
        except Exception as e:
            self.logger.error(f"❌ 영상 단위 키워드 필터링 실패: {str(e)}")
            return df
    
    def _apply_comment_level_keyword_filter(self, df: pd.DataFrame, text_column: str, config: Dict) -> pd.DataFrame:
        """
        댓글 단위 필수 키워드 필터링 (영상 단위 필터링이 불가능한 경우 대체)
        """
        try:
            required_mask = pd.Series([False] * len(df), index=df.index)
            
            for keyword in config['required_keywords']:
                keyword_mask = self._create_keyword_mask(df[text_column], keyword, config)
                required_mask = required_mask | keyword_mask
            
            filtered_df = df[required_mask].copy()
            self.logger.info(f"💬 댓글 단위 필수 키워드 필터링: {len(filtered_df)}개 댓글")
            
            return filtered_df
            
        except Exception as e:
            self.logger.error(f"❌ 댓글 단위 키워드 필터링 실패: {str(e)}")
            return df
    
    def _apply_comment_level_exclusion_filter(self, df: pd.DataFrame, text_column: str, config: Dict) -> pd.DataFrame:
        """
        댓글 단위 제외 키워드 필터링
        개별 댓글에 제외 키워드가 있으면 그 댓글만 제외
        """
        try:
            mask = pd.Series([True] * len(df), index=df.index)
            
            for keyword in config['excluded_keywords']:
                exclude_mask = ~self._create_keyword_mask(df[text_column], keyword, config)
                mask = mask & exclude_mask
            
            filtered_df = df[mask].copy()
            excluded_count = len(df) - len(filtered_df)
            
            self.logger.info(f"🚫 댓글 단위 제외 키워드 필터링: {excluded_count}개 댓글 제외")
            
            return filtered_df
            
        except Exception as e:
            self.logger.error(f"❌ 댓글 단위 제외 키워드 필터링 실패: {str(e)}")
            return df
    
    def _text_contains_keywords(self, text: str, keywords: List[str], config: Dict) -> bool:
        """
        텍스트에 키워드가 포함되어 있는지 확인
        """
        try:
            if not isinstance(text, str) or not text.strip():
                return False
            
            for keyword in keywords:
                if self._text_contains_keyword(text, keyword, config):
                    return True
            
            return False
            
        except Exception as e:
            return False
    
    def _text_contains_keyword(self, text: str, keyword: str, config: Dict) -> bool:
        """
        텍스트에 특정 키워드가 포함되어 있는지 확인
        """
        try:
            if config['case_sensitive']:
                search_text = text
                search_keyword = keyword
            else:
                search_text = text.lower()
                search_keyword = keyword.lower()
            
            if config['partial_match']:
                return search_keyword in search_text
            else:
                # 완전 단어 매칭
                import re
                pattern = f'\\b{re.escape(search_keyword)}\\b'
                return bool(re.search(pattern, search_text))
                
        except Exception as e:
            return False
    
    def _create_keyword_mask(self, text_series: pd.Series, keyword: str, config: Dict) -> pd.Series:
        """
        키워드에 대한 마스크 생성
        """
        try:
            if config['case_sensitive']:
                if config['partial_match']:
                    return text_series.str.contains(keyword, na=False, regex=False)
                else:
                    return text_series.str.contains(f'\\b{re.escape(keyword)}\\b', na=False, regex=True)
            else:
                if config['partial_match']:
                    return text_series.str.contains(keyword, case=False, na=False, regex=False)
                else:
                    return text_series.str.contains(f'\\b{re.escape(keyword)}\\b', case=False, na=False, regex=True)
                    
        except Exception as e:
            self.logger.error(f"❌ 키워드 마스크 생성 실패: {str(e)}")
            return pd.Series([False] * len(text_series), index=text_series.index)
    
    def _apply_quality_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        품질 기반 필터링
        Args:
            df: 데이터프레임
        Returns:
            필터링된 데이터프레임
        """
        try:
            config = self.filtering_config['quality_filtering']
            
            # 텍스트 컬럼 확인
            text_column = None
            for col in ['comment_text', 'cleaned_text', 'text', 'content']:
                if col in df.columns:
                    text_column = col
                    break
            
            if text_column is None:
                self.logger.warning("⚠️ 텍스트 컬럼을 찾을 수 없어 품질 필터링을 건너뜁니다.")
                return df
            
            # NaN 값 처리
            df[text_column] = df[text_column].fillna('')
            
            mask = pd.Series([True] * len(df), index=df.index)
            
            for idx, text in tqdm(df[text_column].items(), desc="품질 필터링", total=len(df)):
                if not isinstance(text, str) or len(text.strip()) == 0:
                    mask[idx] = False
                    continue
                
                # 한 글자 댓글 제외
                if config['exclude_single_char_comments'] and len(text.strip()) == 1:
                    mask[idx] = False
                    continue
                
                # 한글 비율 확인
                korean_chars = len(re.findall(r'[가-힣]', text))
                total_chars = len(text)
                korean_ratio = korean_chars / total_chars if total_chars > 0 else 0
                
                if korean_ratio < config['min_korean_ratio']:
                    mask[idx] = False
                    continue
                
                # 특수문자 비율 확인
                special_chars = len(re.findall(r'[^\w\s가-힣]', text))
                special_ratio = special_chars / total_chars if total_chars > 0 else 0
                
                if special_ratio > config['max_special_char_ratio']:
                    mask[idx] = False
                    continue
                
                # 반복 문자 비율 확인
                repeated_chars = 0
                for char in set(text):
                    if char != ' ':
                        char_count = text.count(char)
                        if char_count > 3:  # 3번 이상 반복되는 문자
                            repeated_chars += char_count - 3
                
                repeated_ratio = repeated_chars / total_chars if total_chars > 0 else 0
                
                if repeated_ratio > config['max_repeated_char_ratio']:
                    mask[idx] = False
                    continue
                
                # 의미있는 단어 수 확인
                words = re.findall(r'[가-힣a-zA-Z]+', text)
                meaningful_words = [word for word in words if len(word) > 1]
                
                if len(meaningful_words) < config['min_meaningful_words']:
                    mask[idx] = False
                    continue
                
                # 이모지만 있는 댓글 제외
                if config['exclude_emoji_only_comments']:
                    # 이모지 패턴 (간단한 버전)
                    emoji_pattern = re.compile(
                        "["
                        "\U0001F600-\U0001F64F"  # emoticons
                        "\U0001F300-\U0001F5FF"  # symbols & pictographs
                        "\U0001F680-\U0001F6FF"  # transport & map symbols
                        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
                        "\U00002702-\U000027B0"
                        "\U000024C2-\U0001F251"
                        "]+", flags=re.UNICODE
                    )
                    
                    text_without_emoji = emoji_pattern.sub('', text).strip()
                    if len(text_without_emoji) == 0:
                        mask[idx] = False
                        continue
            
            filtered_df = df[mask].copy()
            
            self.logger.info(f"✨ 품질 필터링: 한글비율 {config['min_korean_ratio']:.1%} 이상, "
                           f"특수문자 {config['max_special_char_ratio']:.1%} 이하, "
                           f"의미있는 단어 {config['min_meaningful_words']}개 이상")
            
            return filtered_df
            
        except Exception as e:
            self.logger.error(f"❌ 품질 필터링 실패: {str(e)}")
            return df
    
    def _apply_duplicate_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        중복 댓글 필터링
        Args:
            df: 데이터프레임
        Returns:
            필터링된 데이터프레임
        """
        try:
            config = self.filtering_config['duplicate_filtering']
            
            # 텍스트 컬럼 확인
            text_column = None
            for col in ['comment_text', 'cleaned_text', 'text', 'content']:
                if col in df.columns:
                    text_column = col
                    break
            
            if text_column is None:
                self.logger.warning("⚠️ 텍스트 컬럼을 찾을 수 없어 중복 필터링을 건너뜁니다.")
                return df
            
            # NaN 값 처리
            df[text_column] = df[text_column].fillna('')
            
            if config['exact_match_only']:
                # 완전 일치 중복 제거
                if config['keep_highest_upvotes'] and 'upvotes' in df.columns:
                    # 추천수가 높은 것 유지
                    df_sorted = df.sort_values('upvotes', ascending=False)
                    filtered_df = df_sorted.drop_duplicates(subset=[text_column], keep='first')
                else:
                    # 첫 번째 것 유지
                    filtered_df = df.drop_duplicates(subset=[text_column], keep='first')
            else:
                # 유사도 기반 중복 제거
                filtered_df = self._remove_similar_duplicates(df, text_column, config)
            
            filter_type = '완전일치' if config['exact_match_only'] else f"유사도 {config['similarity_threshold']:.1%}"
            self.logger.info(f"🔄 중복 필터링: {filter_type} 기준")
            
            return filtered_df
            
        except Exception as e:
            self.logger.error(f"❌ 중복 필터링 실패: {str(e)}")
            return df
    
    def _remove_similar_duplicates(self, df: pd.DataFrame, text_column: str, config: Dict) -> pd.DataFrame:
        """
        유사도 기반 중복 제거
        Args:
            df: 데이터프레임
            text_column: 텍스트 컬럼명
            config: 중복 필터링 설정
        Returns:
            중복 제거된 데이터프레임
        """
        try:
            texts = df[text_column].tolist()
            indices_to_keep = []
            processed_indices = set()
            
            for i, text1 in enumerate(tqdm(texts, desc="유사도 기반 중복 제거")):
                if i in processed_indices:
                    continue
                
                # 현재 텍스트와 유사한 텍스트들 찾기
                similar_indices = [i]
                
                for j, text2 in enumerate(texts[i+1:], start=i+1):
                    if j in processed_indices:
                        continue
                    
                    # 유사도 계산
                    similarity = difflib.SequenceMatcher(None, text1, text2).ratio()
                    
                    if similarity >= config['similarity_threshold']:
                        similar_indices.append(j)
                        processed_indices.add(j)
                
                # 유사한 댓글들 중에서 하나 선택
                if config['keep_highest_upvotes'] and 'upvotes' in df.columns:
                    # 추천수가 가장 높은 것 선택
                    best_idx = max(similar_indices, key=lambda x: df.iloc[x]['upvotes'])
                else:
                    # 첫 번째 것 선택
                    best_idx = similar_indices[0]
                
                indices_to_keep.append(best_idx)
                processed_indices.add(i)
            
            filtered_df = df.iloc[indices_to_keep].copy()
            return filtered_df
            
        except Exception as e:
            self.logger.error(f"❌ 유사도 기반 중복 제거 실패: {str(e)}")
            return df
    
    def _apply_user_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        사용자 기반 필터링
        Args:
            df: 데이터프레임
        Returns:
            필터링된 데이터프레임
        """
        try:
            config = self.filtering_config['user_filtering']
            
            # 사용자 컬럼 확인
            user_column = None
            for col in ['author_name', 'user_name', 'username', 'author']:
                if col in df.columns:
                    user_column = col
                    break
            
            if user_column is None:
                self.logger.warning("⚠️ 사용자 컬럼을 찾을 수 없어 사용자 필터링을 건너뜁니다.")
                return df
            
            # NaN 값 처리
            df[user_column] = df[user_column].fillna('Unknown')
            
            mask = pd.Series([True] * len(df), index=df.index)
            
            # 제외할 사용자 필터링
            if config['exclude_users']:
                exclude_mask = ~df[user_column].isin(config['exclude_users'])
                mask = mask & exclude_mask
            
            # 사용자별 댓글 수 제한
            user_counts = df[user_column].value_counts()
            
            # 최소 댓글 수 필터링
            valid_users = user_counts[user_counts >= config['min_user_comments']].index
            min_mask = df[user_column].isin(valid_users)
            mask = mask & min_mask
            
            # 최대 댓글 수 제한 (스팸 방지)
            if config['max_user_comments'] > 0:
                spam_users = user_counts[user_counts > config['max_user_comments']].index
                
                # 스팸 사용자의 댓글 중 일부만 유지 (추천수 기준)
                for user in spam_users:
                    user_comments = df[df[user_column] == user]
                    if 'upvotes' in df.columns:
                        # 추천수 기준으로 상위 댓글만 유지
                        top_comments = user_comments.nlargest(config['max_user_comments'], 'upvotes')
                    else:
                        # 랜덤하게 선택
                        top_comments = user_comments.sample(n=config['max_user_comments'])
                    
                    # 해당 사용자의 다른 댓글들은 제외
                    user_mask = (df[user_column] != user) | df.index.isin(top_comments.index)
                    mask = mask & user_mask
            
            filtered_df = df[mask].copy()
            
            self.logger.info(f"👤 사용자 필터링: 사용자당 {config['min_user_comments']}-{config['max_user_comments']}개 댓글")
            
            return filtered_df
            
        except Exception as e:
            self.logger.error(f"❌ 사용자 필터링 실패: {str(e)}")
            return df
    
    def _log_filtering_summary(self):
        """필터링 결과 요약 로깅"""
        try:
            stats = self.filter_stats
            total_removed = stats['original_count'] - stats['final_count']
            removal_rate = (total_removed / stats['original_count']) * 100 if stats['original_count'] > 0 else 0
            
            self.logger.info("=" * 60)
            self.logger.info("📊 데이터 필터링 결과 요약")
            self.logger.info("=" * 60)
            self.logger.info(f"📥 원본 댓글 수: {stats['original_count']:,}개")
            self.logger.info(f"📊 추천수 필터링: {stats['upvote_filtered']:,}개 제거")
            self.logger.info(f"🔑 키워드 필터링: {stats['keyword_filtered']:,}개 제거")
            self.logger.info(f"✨ 품질 필터링: {stats['quality_filtered']:,}개 제거")
            self.logger.info(f"🔄 중복 필터링: {stats['duplicate_filtered']:,}개 제거")
            self.logger.info(f"👤 사용자 필터링: {stats['user_filtered']:,}개 제거")
            self.logger.info("-" * 60)
            self.logger.info(f"📤 최종 댓글 수: {stats['final_count']:,}개")
            self.logger.info(f"🗑️ 총 제거된 댓글: {total_removed:,}개 ({removal_rate:.1f}%)")
            self.logger.info(f"✅ 데이터 품질 향상률: {100 - removal_rate:.1f}%")
            self.logger.info("=" * 60)
            
        except Exception as e:
            self.logger.error(f"❌ 필터링 요약 로깅 실패: {str(e)}")
    
    def get_filtering_stats(self) -> Dict:
        """
        필터링 통계 반환
        Returns:
            필터링 통계 딕셔너리
        """
        return self.filter_stats.copy()
    
    def validate_filtering_config(self) -> bool:
        """
        필터링 설정 검증
        Returns:
            설정이 유효한지 여부
        """
        try:
            config = self.filtering_config
            self.logger.debug(f"🔍 필터링 설정 검증 시작: {list(config.keys())}")
            
            # 추천수 필터링 검증
            upvote_config = config['upvote_filtering']
            self.logger.debug(f"📊 추천수 필터링 설정: {upvote_config}")
            if upvote_config['min_upvotes'] < 0:
                self.logger.error("❌ 최소 추천수는 0 이상이어야 합니다.")
                return False
            
            # 키워드 필터링 검증
            keyword_config = config['keyword_filtering']
            self.logger.debug(f"🔑 키워드 필터링 설정: {list(keyword_config.keys())}")
            
            # basic_conditions 내의 min_keyword_matches 확인
            basic_conditions = keyword_config.get('basic_conditions', {})
            self.logger.debug(f"🔧 basic_conditions: {basic_conditions}")
            
            if 'min_keyword_matches' in basic_conditions:
                if basic_conditions['min_keyword_matches'] < 1:
                    self.logger.error("❌ 최소 키워드 매칭 수는 1 이상이어야 합니다.")
                    return False
            else:
                self.logger.warning("⚠️ basic_conditions에 min_keyword_matches가 없습니다.")
            
            # 품질 필터링 검증
            quality_config = config['quality_filtering']
            self.logger.debug(f"✨ 품질 필터링 설정: {list(quality_config.keys())}")
            if not (0 <= quality_config['min_korean_ratio'] <= 1):
                self.logger.error("❌ 최소 한글 비율은 0-1 사이여야 합니다.")
                return False
            
            if not (0 <= quality_config['max_special_char_ratio'] <= 1):
                self.logger.error("❌ 최대 특수문자 비율은 0-1 사이여야 합니다.")
                return False
            
            # 중복 필터링 검증
            duplicate_config = config['duplicate_filtering']
            self.logger.debug(f"🔄 중복 필터링 설정: {list(duplicate_config.keys())}")
            if not (0 <= duplicate_config['similarity_threshold'] <= 1):
                self.logger.error("❌ 유사도 임계값은 0-1 사이여야 합니다.")
                return False
            
            self.logger.info("✅ 필터링 설정 검증 완료")
            return True
            
        except KeyError as e:
            self.logger.error(f"❌ 필터링 설정 검증 실패: {str(e)}")
            self.logger.error(f"🔍 사용 가능한 설정 키: {list(self.filtering_config.keys()) if hasattr(self, 'filtering_config') else 'filtering_config 없음'}")
            return False
        except Exception as e:
            self.logger.error(f"❌ 필터링 설정 검증 실패: {str(e)}")
            return False
    
    def _apply_enhanced_keyword_filter(self, df: pd.DataFrame, text_column: str, config: Dict) -> pd.DataFrame:
        """
        강화된 키워드 필터링
        조건 1: 제목에 키워드 포함 OR 조건 2: 댓글들 중 최소 비율 이상이 키워드 포함
        """
        try:
            enhanced_config = config.get('enhanced_conditions', {})
            
            # 영상 식별 컬럼 확인
            video_id_column = None
            for col in ['video_no', 'video_id', 'video_url', 'url']:
                if col in df.columns:
                    video_id_column = col
                    break
            
            if video_id_column is None:
                self.logger.warning("⚠️ 영상 식별 컬럼을 찾을 수 없어 댓글 단위로 필터링합니다.")
                return self._apply_comment_level_keyword_filter(df, text_column, config)
            
            valid_videos = set()
            
            # 각 영상별로 강화된 조건 확인
            for video_id, video_group in df.groupby(video_id_column):
                video_passes = False
                
                # 조건 1: 제목에 키워드 포함 확인
                title_condition = enhanced_config.get('title_contains_keywords', {})
                if title_condition.get('enabled', False):
                    title_column = title_condition.get('title_column', 'video_title')
                    title_keywords = title_condition.get('keywords', config.get('required_keywords', []))
                    
                    if title_column in df.columns and not video_group[title_column].isna().all():
                        video_title = str(video_group[title_column].iloc[0])
                        if self._text_contains_keywords(video_title, title_keywords, config):
                            video_passes = True
                            self.logger.debug(f"📺 영상 제목 조건 통과: {video_id}")
                
                # 조건 2: 댓글 비율 조건 확인 (조건 1이 실패한 경우에만)
                if not video_passes:
                    ratio_condition = enhanced_config.get('comment_ratio_threshold', {})
                    if ratio_condition.get('enabled', False):
                        min_ratio = ratio_condition.get('min_ratio', 0.40)
                        ratio_keywords = ratio_condition.get('keywords', config.get('required_keywords', []))
                        
                        # 해당 영상의 댓글들 중 키워드 포함 비율 계산
                        total_comments = len(video_group)
                        keyword_comments = 0
                        
                        for comment_text in video_group[text_column]:
                            if self._text_contains_keywords(str(comment_text), ratio_keywords, config):
                                keyword_comments += 1
                        
                        if total_comments > 0:
                            keyword_ratio = keyword_comments / total_comments
                            if keyword_ratio >= min_ratio:
                                video_passes = True
                                self.logger.debug(f"💬 댓글 비율 조건 통과: {video_id} ({keyword_ratio:.1%} >= {min_ratio:.1%})")
                
                # 조건을 만족하는 영상을 유효 목록에 추가
                if video_passes:
                    valid_videos.add(video_id)
            
            # 유효한 영상의 모든 댓글 반환
            filtered_df = df[df[video_id_column].isin(valid_videos)].copy()
            
            self.logger.info(f"🔧 강화된 키워드 필터링 결과: {len(valid_videos)}개 영상의 {len(filtered_df)}개 댓글")
            
            # 조건별 통계 로깅
            title_condition = enhanced_config.get('title_contains_keywords', {})
            ratio_condition = enhanced_config.get('comment_ratio_threshold', {})
            
            if title_condition.get('enabled', False):
                self.logger.info(f"  📺 제목 조건: {title_condition.get('title_column', 'video_title')} 컬럼에서 키워드 확인")
            
            if ratio_condition.get('enabled', False):
                min_ratio = ratio_condition.get('min_ratio', 0.40)
                self.logger.info(f"  💬 댓글 비율 조건: 최소 {min_ratio:.1%} 이상의 댓글이 키워드 포함")
            
            return filtered_df
            
        except Exception as e:
            self.logger.error(f"❌ 강화된 키워드 필터링 실패: {str(e)}")
            return df
    
    def _apply_basic_keyword_filter(self, df: pd.DataFrame, text_column: str, config: Dict) -> pd.DataFrame:
        """
        기본 키워드 필터링 (기존 방식)
        """
        try:
            # 1단계: 영상 단위 필수 키워드 필터링
            if config['required_keywords']:
                filtered_df = self._apply_video_level_keyword_filter(df, text_column, config)
                self.logger.info(f"🎬 영상 단위 키워드 필터링: {len(config['required_keywords'])}개 키워드 기준")
            else:
                filtered_df = df.copy()
            
            return filtered_df
            
        except Exception as e:
            self.logger.error(f"❌ 기본 키워드 필터링 실패: {str(e)}")
            return df 