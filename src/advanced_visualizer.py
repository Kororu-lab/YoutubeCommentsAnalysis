"""
Advanced Visualizer Module
고급 프레임 분석 시각화 모듈

이 모듈은 다음 시각화를 제공합니다:
1. 시간 기반 여론 흐름 시각화
2. 토픽 진화 및 비교 시각화
3. 키워드 공출현 네트워크 시각화
4. 언론 프레임 유사도 시각화
5. 변곡점 및 이상치 시각화
6. 인터랙티브 대시보드
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from matplotlib import rcParams
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import networkx as nx
from wordcloud import WordCloud
import logging
from typing import Dict, List, Tuple, Optional, Any
import os
from datetime import datetime
import platform
from collections import Counter

class AdvancedVisualizer:
    """고급 시각화 클래스"""
    
    def __init__(self, config):
        """
        초기화
        Args:
            config: AnalysisConfig 객체
        """
        self.config = config
        self.logger = self._setup_logger()
        self.output_dir = config.OUTPUT_STRUCTURE['visualizations']
        
        # 한국어 폰트 설정
        self._setup_korean_font()
        
        # 시각화 스타일 설정
        self._setup_style()
        
        # 색상 팔레트 설정
        self.colors = {
            'primary': '#2E8B57',
            'secondary': '#DC143C', 
            'accent': '#4169E1',
            'neutral': '#708090',
            'positive': '#32CD32',
            'negative': '#FF6347',
            'warning': '#FFD700',
            'info': '#87CEEB'
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
    
    def _setup_korean_font(self):
        """한글 폰트 설정 (visualizer.py와 동일)"""
        try:
            import matplotlib.font_manager as fm
            
            # 시스템별 한글 폰트 찾기
            korean_fonts = []
            system = platform.system()
            
            if system == "Darwin":  # macOS
                korean_fonts = [
                    "AppleGothic", "Apple SD Gothic Neo", "Noto Sans CJK KR", 
                    "Malgun Gothic", "NanumGothic", "Arial Unicode MS"
                ]
            elif system == "Windows":
                korean_fonts = [
                    "Malgun Gothic", "NanumGothic", "Gulim", "Dotum", 
                    "Batang", "Gungsuh", "Arial Unicode MS"
                ]
            else:  # Linux
                korean_fonts = [
                    "Noto Sans CJK KR", "NanumGothic", "UnDotum", 
                    "Baekmuk Gulim", "Arial Unicode MS"
                ]
            
            # 사용 가능한 폰트 찾기
            available_fonts = [f.name for f in fm.fontManager.ttflist]
            found_font = None
            found_font_path = None
            
            for font in korean_fonts:
                if font in available_fonts:
                    found_font = font
                    for font_obj in fm.fontManager.ttflist:
                        if font_obj.name == font and font_obj.fname.endswith('.ttf'):
                            found_font_path = font_obj.fname
                            break
                    break
            
            if found_font:
                self.korean_font_name = found_font
                self.korean_font_path = found_font_path
                
                # matplotlib 설정
                rcParams['font.family'] = found_font
                rcParams['font.sans-serif'] = [found_font] + ['DejaVu Sans', 'Arial']
                rcParams['axes.unicode_minus'] = False
                rcParams['font.size'] = 12
                
                plt.rcParams['font.family'] = found_font
                plt.rcParams['font.sans-serif'] = [found_font] + ['DejaVu Sans', 'Arial']
                plt.rcParams['axes.unicode_minus'] = False
                plt.rcParams['font.size'] = 12
                
                self.logger.info(f"✅ 한글 폰트 설정 완료: {found_font}")
            else:
                self.korean_font_name = "DejaVu Sans"
                self.korean_font_path = None
                
                rcParams['font.family'] = 'DejaVu Sans'
                rcParams['axes.unicode_minus'] = False
                plt.rcParams['font.family'] = 'DejaVu Sans'
                plt.rcParams['axes.unicode_minus'] = False
                
                self.logger.warning("⚠️ 한글 폰트를 찾을 수 없어 기본 폰트 사용")
                
        except Exception as e:
            self.logger.error(f"폰트 설정 중 오류 발생: {e}")
            self.korean_font_name = "DejaVu Sans"
            self.korean_font_path = None
    
    def _apply_font_settings(self):
        """각 플롯 생성 전에 폰트 설정 재적용"""
        try:
            plt.rcParams['font.family'] = self.korean_font_name
            plt.rcParams['font.sans-serif'] = [self.korean_font_name] + ['DejaVu Sans', 'Arial']
            plt.rcParams['axes.unicode_minus'] = False
            plt.rcParams['font.size'] = 12
        except:
            plt.rcParams['font.family'] = 'DejaVu Sans'
            plt.rcParams['axes.unicode_minus'] = False
    
    def _setup_style(self):
        """시각화 스타일 설정"""
        plt.style.use('default')
        sns.set_palette("Set2")
        
        rcParams['figure.figsize'] = (15, 10)
        rcParams['figure.dpi'] = 300
        rcParams['font.size'] = 12
        rcParams['axes.grid'] = True
        rcParams['grid.alpha'] = 0.3
    
    def visualize_temporal_opinion_flow(self, temporal_results: Dict, case_name: str) -> str:
        """
        시간 기반 여론 흐름 시각화 (프레임 전환 중심으로 개선)
        
        Args:
            temporal_results: 시간별 분석 결과
            case_name: 사건명
        
        Returns:
            저장된 파일 경로
        """
        try:
            self.logger.info(f"📊 {case_name} 시간 기반 여론 흐름 시각화 시작")
            
            self._apply_font_settings()
            
            # 데이터 준비 및 검증
            if 'temporal_results' not in temporal_results:
                self.logger.warning("⚠️ temporal_results 키가 없습니다.")
                return None
                
            periods = sorted(temporal_results['temporal_results'].keys())
            data = temporal_results['temporal_results']
            
            # 최소 데이터 요구사항 확인
            if len(periods) < 2:
                self.logger.warning(f"⚠️ 시간 구간이 부족합니다 ({len(periods)}개). 최소 2개 필요.")
                return None
            
            # 시계열 데이터 추출 및 기본값 설정
            sentiment_scores = []
            comment_counts = []
            volatilities = []
            
            for p in periods:
                period_data = data[p]
                sentiment_scores.append(period_data.get('avg_sentiment', 0.0))
                comment_counts.append(period_data.get('comment_count', 0))
                volatilities.append(period_data.get('sentiment_volatility', 0.0))
            
            # 데이터 유효성 검증
            if len(sentiment_scores) == 0 or all(score == 0 for score in sentiment_scores):
                self.logger.warning("⚠️ 유효한 감성 점수 데이터가 없습니다.")
                return None
            
            # 변곡점 정보
            changepoints = temporal_results.get('changepoints', [])
            changepoint_indices = [cp['index'] for cp in changepoints if cp.get('index', -1) < len(periods)]
            
            # 4개 서브플롯 생성
            fig, axes = plt.subplots(2, 2, figsize=(20, 15))
            fig.suptitle(f'{case_name} 여론 프레임 전환 분석', fontsize=20, fontweight='bold')
            
            # 1. 감성 모멘텀 분석 (감성 점수 + 변화율 + 가속도)
            sentiment_velocity = np.diff(sentiment_scores) if len(sentiment_scores) > 1 else np.array([0])
            sentiment_acceleration = np.diff(sentiment_velocity) if len(sentiment_velocity) > 1 else np.array([0])
            
            # 주축: 감성 점수
            line1 = axes[0, 0].plot(range(len(periods)), sentiment_scores, 
                                   marker='o', linewidth=3, markersize=8, 
                                   color=self.colors['primary'], label='감성 점수')
            
            # 이차 축: 감성 변화율 (차원 맞춤)
            ax_twin = axes[0, 0].twinx()
            if len(sentiment_velocity) > 0:
                # 변화율의 x축은 sentiment_velocity의 길이에 맞춤
                velocity_x = range(1, len(sentiment_velocity) + 1) if len(sentiment_velocity) == len(periods) - 1 else range(len(sentiment_velocity))
                line2 = ax_twin.plot(velocity_x, sentiment_velocity, 
                                   marker='s', linewidth=2, markersize=6, 
                                   color=self.colors['warning'], alpha=0.8, label='변화율')
            else:
                # 변화율 데이터가 없는 경우 빈 플롯
                line2 = ax_twin.plot([], [], marker='s', linewidth=2, markersize=6, 
                                   color=self.colors['warning'], alpha=0.8, label='변화율')
            
            # 변곡점 및 급변 구간 표시
            for cp_idx in changepoint_indices:
                axes[0, 0].axvline(x=cp_idx, color=self.colors['negative'], 
                                  linestyle='--', alpha=0.8, linewidth=3)
                axes[0, 0].text(cp_idx, sentiment_scores[cp_idx], '변곡점', 
                               rotation=90, verticalalignment='bottom', fontweight='bold')
            
            # 급변 구간 하이라이트 (안전한 처리)
            if len(sentiment_velocity) > 1:
                velocity_std = np.std(sentiment_velocity)
                if velocity_std > 0:  # 표준편차가 0이 아닌 경우만
                    for i, vel in enumerate(sentiment_velocity):
                        if abs(vel) > velocity_std * 1.5:  # 1.5 표준편차 이상
                            axes[0, 0].axvspan(i, i+1, alpha=0.3, color='red', 
                                             label='급변구간' if i == 0 else "")
            
            axes[0, 0].set_title('감성 모멘텀 분석 (점수 + 변화율)', fontsize=14, fontweight='bold')
            axes[0, 0].set_xlabel('시간 구간')
            axes[0, 0].set_ylabel('감성 점수', color=self.colors['primary'])
            ax_twin.set_ylabel('변화율', color=self.colors['warning'])
            axes[0, 0].set_xticks(range(len(periods)))
            axes[0, 0].set_xticklabels(periods, rotation=45, ha='right')
            axes[0, 0].grid(True, alpha=0.3)
            
            # 범례 통합 (안전한 처리)
            try:
                lines1, labels1 = axes[0, 0].get_legend_handles_labels()
                lines2, labels2 = ax_twin.get_legend_handles_labels()
                all_lines = lines1 + lines2
                all_labels = labels1 + labels2
                if all_lines and all_labels:
                    axes[0, 0].legend(all_lines, all_labels, loc='upper left')
            except Exception as legend_error:
                self.logger.warning(f"⚠️ 범례 생성 실패: {legend_error}")
                # 기본 범례만 표시
                axes[0, 0].legend(loc='upper left')
            
            # 2. 여론 강도 vs 분극화 지수 (안전한 처리)
            # 여론 강도 = 댓글 수 * (1 + |평균 감성|)
            opinion_intensity = []
            for count, score in zip(comment_counts, sentiment_scores):
                intensity = count * (1 + abs(score)) if count > 0 and score is not None else 0
                opinion_intensity.append(intensity)
            
            # 분극화 지수 = 감성 변동성 * 댓글 수 정규화
            max_comments = max(comment_counts) if comment_counts and max(comment_counts) > 0 else 1
            polarization_index = []
            for vol, count in zip(volatilities, comment_counts):
                pol_idx = vol * (count / max_comments) if vol is not None and count is not None else 0
                polarization_index.append(pol_idx)
            
            # 산점도로 각 시점 표시 (데이터 유효성 확인)
            if len(opinion_intensity) > 0 and len(polarization_index) > 0:
                scatter = axes[0, 1].scatter(opinion_intensity, polarization_index, 
                                           c=range(len(periods)), cmap='viridis', 
                                           s=100, alpha=0.8, edgecolors='black')
                
                # 시간 순서 화살표 (안전한 처리)
                for i in range(len(periods) - 1):
                    if (i < len(opinion_intensity) - 1 and i < len(polarization_index) - 1 and
                        i + 1 < len(opinion_intensity) and i + 1 < len(polarization_index)):
                        try:
                            axes[0, 1].annotate('', xy=(opinion_intensity[i+1], polarization_index[i+1]),
                                               xytext=(opinion_intensity[i], polarization_index[i]),
                                               arrowprops=dict(arrowstyle='->', color='gray', alpha=0.6))
                        except Exception as arrow_error:
                            self.logger.warning(f"⚠️ 화살표 생성 실패 (인덱스 {i}): {arrow_error}")
                
                # 각 점에 기간 라벨 (안전한 처리)
                for i, period in enumerate(periods):
                    if i < len(opinion_intensity) and i < len(polarization_index):
                        try:
                            axes[0, 1].annotate(period[:6], (opinion_intensity[i], polarization_index[i]),
                                               xytext=(5, 5), textcoords='offset points', fontsize=8)
                        except Exception as label_error:
                            self.logger.warning(f"⚠️ 라벨 생성 실패 (인덱스 {i}): {label_error}")
            else:
                # 데이터가 없는 경우 빈 플롯
                scatter = axes[0, 1].scatter([], [], c=[], cmap='viridis', s=100, alpha=0.8, edgecolors='black')
            
            axes[0, 1].set_title('여론 강도 vs 분극화 궤적', fontsize=14, fontweight='bold')
            axes[0, 1].set_xlabel('여론 강도 (댓글수 × 감성강도)')
            axes[0, 1].set_ylabel('분극화 지수 (변동성 × 참여도)')
            axes[0, 1].grid(True, alpha=0.3)
            
            # 컬러바 (안전한 처리)
            try:
                if len(opinion_intensity) > 0 and len(polarization_index) > 0:
                    cbar = plt.colorbar(scatter, ax=axes[0, 1])
                    cbar.set_label('시간 순서')
            except Exception as cbar_error:
                self.logger.warning(f"⚠️ 컬러바 생성 실패: {cbar_error}")
            
            # 3. 프레임 전환 패턴 (키워드 기반) - 안전한 처리
            # 각 기간별 상위 키워드 변화 추적
            frame_transitions = []
            prev_keywords = set()
            
            for period in periods:
                current_keywords = set()
                period_data = data.get(period, {})
                keywords = period_data.get('top_keywords', [])
                
                # 키워드 데이터 안전하게 처리
                if keywords and isinstance(keywords, list):
                    for kw in keywords[:10]:  # 상위 10개
                        if isinstance(kw, (list, tuple)) and len(kw) > 0:
                            current_keywords.add(str(kw[0]))
                        elif isinstance(kw, str):
                            current_keywords.add(kw)
                
                # 자카드 유사도로 프레임 연속성 측정
                if prev_keywords and current_keywords:
                    intersection = len(prev_keywords & current_keywords)
                    union = len(prev_keywords | current_keywords)
                    similarity = intersection / union if union > 0 else 0
                    transition_score = 1 - similarity  # 변화율
                elif prev_keywords or current_keywords:
                    transition_score = 1.0  # 완전 전환
                else:
                    transition_score = 0.0  # 키워드 없음
                
                frame_transitions.append(transition_score)
                prev_keywords = current_keywords
            
            bars = axes[1, 0].bar(range(len(periods)), frame_transitions, 
                                 color=['red' if score > 0.7 else 'orange' if score > 0.4 else 'green' 
                                       for score in frame_transitions], alpha=0.8)
            
            axes[1, 0].set_title('프레임 전환 강도 (키워드 변화율)', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('시간 구간')
            axes[1, 0].set_ylabel('전환 강도 (0=연속, 1=완전전환)')
            axes[1, 0].set_xticks(range(len(periods)))
            axes[1, 0].set_xticklabels(periods, rotation=45, ha='right')
            axes[1, 0].grid(True, alpha=0.3)
            
            # 임계값 선 표시
            axes[1, 0].axhline(y=0.7, color='red', linestyle='--', alpha=0.7, label='급변 임계값')
            axes[1, 0].axhline(y=0.4, color='orange', linestyle='--', alpha=0.7, label='중변 임계값')
            axes[1, 0].legend()
            
            # 값 표시
            for bar, score in zip(bars, frame_transitions):
                height = bar.get_height()
                axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                               f'{score:.2f}', ha='center', va='bottom', fontsize=10)
            
            # 4. 여론 생태계 지도 (감성 vs 참여도 vs 다양성) - 안전한 처리
            # 참여도 = 댓글 수 정규화
            max_comments = max(comment_counts) if comment_counts and max(comment_counts) > 0 else 1
            participation = []
            for count in comment_counts:
                part = count / max_comments if count is not None and max_comments > 0 else 0
                participation.append(part)
            
            # 다양성 = 키워드 수
            diversity = []
            for period in periods:
                period_data = data.get(period, {})
                keywords = period_data.get('top_keywords', [])
                div = len(keywords) if keywords and isinstance(keywords, list) else 0
                diversity.append(div)
            
            # 3D 산점도 효과를 2D로 구현 (안전한 처리)
            bubble_sizes = [max(d * 50, 10) for d in diversity]  # 다양성을 크기로, 최소 크기 보장
            
            if len(sentiment_scores) > 0 and len(participation) > 0:
                scatter = axes[1, 1].scatter(sentiment_scores, participation, 
                                           s=bubble_sizes, c=range(len(periods)), 
                                           cmap='plasma', alpha=0.7, edgecolors='black')
                
                # 시간 순서 연결선 (안전한 처리)
                if len(sentiment_scores) > 1 and len(participation) > 1:
                    try:
                        axes[1, 1].plot(sentiment_scores, participation, 
                                       color='gray', alpha=0.5, linewidth=1, linestyle='--')
                    except Exception as line_error:
                        self.logger.warning(f"⚠️ 연결선 생성 실패: {line_error}")
                
                # 각 점에 기간 라벨 (안전한 처리)
                for i, period in enumerate(periods):
                    if (i < len(sentiment_scores) and i < len(participation) and 
                        sentiment_scores[i] is not None and participation[i] is not None):
                        try:
                            axes[1, 1].annotate(period[:6], (sentiment_scores[i], participation[i]),
                                               xytext=(5, 5), textcoords='offset points', fontsize=8)
                        except Exception as label_error:
                            self.logger.warning(f"⚠️ 라벨 생성 실패 (인덱스 {i}): {label_error}")
            else:
                # 데이터가 없는 경우 빈 플롯
                scatter = axes[1, 1].scatter([], [], s=[], c=[], cmap='plasma', alpha=0.7, edgecolors='black')
            
            axes[1, 1].set_title('여론 생태계 지도 (감성×참여도×다양성)', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('평균 감성 점수')
            axes[1, 1].set_ylabel('참여도 (정규화된 댓글수)')
            axes[1, 1].grid(True, alpha=0.3)
            
            # 사분면 구분선
            axes[1, 1].axhline(y=0.5, color='black', linestyle='-', alpha=0.3)
            axes[1, 1].axvline(x=0, color='black', linestyle='-', alpha=0.3)
            
            # 사분면 라벨
            axes[1, 1].text(0.7, 0.9, '고참여\n긍정', ha='center', va='center', 
                           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
            axes[1, 1].text(-0.7, 0.9, '고참여\n부정', ha='center', va='center',
                           bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
            
            plt.tight_layout()
            
            # 저장
            filename = f'{case_name}_temporal_opinion_flow.png'
            filepath = os.path.join(self.output_dir, filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            self.logger.info(f"✅ {case_name} 시간 기반 여론 흐름 시각화 완료: {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"❌ 시간 기반 여론 흐름 시각화 실패: {str(e)}")
            return None
    
    def visualize_topic_evolution(self, topic_results: Dict, case_name: str) -> str:
        """
        토픽 진화 및 비교 시각화
        
        Args:
            topic_results: 토픽 분석 결과
            case_name: 사건명
        
        Returns:
            저장된 파일 경로
        """
        try:
            self.logger.info(f"🔍 {case_name} 토픽 진화 시각화 시작")
            
            self._apply_font_settings()
            
            # 토픽 결과 데이터 준비
            topic_data = topic_results.get('topic_results', {})
            
            if not topic_data:
                self.logger.warning("⚠️ 토픽 데이터가 없습니다.")
                return None
            
            # 서브플롯 개수 결정
            n_methods = len(topic_data)
            if n_methods == 0:
                return None
            
            fig, axes = plt.subplots(2, max(2, n_methods), figsize=(20, 12))
            if n_methods == 1:
                axes = axes.reshape(2, 1)
            
            fig.suptitle(f'{case_name} 토픽 분석 비교', fontsize=20, fontweight='bold')
            
            method_idx = 0
            
            # 각 토픽 모델링 방법별 시각화
            for method, results in topic_data.items():
                if method_idx >= axes.shape[1]:
                    break
                
                topics = results.get('topics', [])
                if not topics:
                    continue
                
                # 상위 토픽들의 키워드 시각화
                topic_labels = []
                topic_weights = []
                
                for i, topic in enumerate(topics[:5]):  # 상위 5개 토픽
                    if isinstance(topic, dict):
                        keywords = topic.get('keywords', [])
                        weight = topic.get('weight', 0)
                    else:
                        keywords = topic[:3] if len(topic) > 3 else topic
                        weight = 1.0
                    
                    topic_label = ' + '.join([kw[0] if isinstance(kw, tuple) else str(kw) for kw in keywords[:3]])
                    topic_labels.append(f'토픽{i+1}: {topic_label}')
                    topic_weights.append(weight)
                
                # 토픽 가중치 바 차트
                if topic_weights:
                    bars = axes[0, method_idx].barh(range(len(topic_labels)), topic_weights, 
                                                   color=plt.cm.Set3(np.linspace(0, 1, len(topic_labels))))
                    axes[0, method_idx].set_title(f'{method.upper()} 토픽 가중치', fontsize=14, fontweight='bold')
                    axes[0, method_idx].set_xlabel('가중치')
                    axes[0, method_idx].set_yticks(range(len(topic_labels)))
                    axes[0, method_idx].set_yticklabels(topic_labels, fontsize=10)
                    axes[0, method_idx].grid(True, alpha=0.3)
                
                # 토픽 키워드 워드클라우드
                if topics:
                    all_keywords = {}
                    for topic in topics[:3]:  # 상위 3개 토픽
                        if isinstance(topic, dict):
                            keywords = topic.get('keywords', [])
                        else:
                            keywords = topic
                        
                        for kw in keywords[:10]:
                            if isinstance(kw, tuple):
                                word, weight = kw[0], kw[1]
                            else:
                                word, weight = str(kw), 1.0
                            
                            if word in all_keywords:
                                all_keywords[word] += weight
                            else:
                                all_keywords[word] = weight
                    
                    if all_keywords:
                        # 워드클라우드 생성
                        wordcloud_params = {
                            'width': 400,
                            'height': 300,
                            'max_words': 50,
                            'background_color': 'white',
                            'colormap': 'viridis',
                            'relative_scaling': 0.5
                        }
                        
                        if self.korean_font_path:
                            wordcloud_params['font_path'] = self.korean_font_path
                        
                        wordcloud = WordCloud(**wordcloud_params).generate_from_frequencies(all_keywords)
                        
                        axes[1, method_idx].imshow(wordcloud, interpolation='bilinear')
                        axes[1, method_idx].axis('off')
                        axes[1, method_idx].set_title(f'{method.upper()} 주요 키워드', fontsize=14, fontweight='bold')
                
                method_idx += 1
            
            # 빈 서브플롯 숨기기
            for i in range(method_idx, axes.shape[1]):
                axes[0, i].set_visible(False)
                axes[1, i].set_visible(False)
            
            plt.tight_layout()
            
            # 저장
            filename = f'{case_name}_topic_evolution.png'
            filepath = os.path.join(self.output_dir, filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            self.logger.info(f"✅ {case_name} 토픽 진화 시각화 완료: {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"❌ 토픽 진화 시각화 실패: {str(e)}")
            return None
    
    def visualize_keyword_network(self, network_results: Dict, case_name: str) -> str:
        """
        키워드 공출현 네트워크 시각화
        
        Args:
            network_results: 네트워크 분석 결과
            case_name: 사건명
        
        Returns:
            저장된 파일 경로
        """
        try:
            self.logger.info(f"🕸️ {case_name} 키워드 네트워크 시각화 시작")
            
            self._apply_font_settings()
            
            G = network_results.get('network_graph')
            if not G or len(G.nodes()) == 0:
                self.logger.warning("⚠️ 네트워크 그래프가 없습니다.")
                return None
            
            fig, axes = plt.subplots(2, 2, figsize=(20, 16))
            fig.suptitle(f'{case_name} 키워드 공출현 네트워크 분석', fontsize=20, fontweight='bold')
            
            # 1. 전체 네트워크 시각화
            pos = nx.spring_layout(G, k=1, iterations=50)
            
            # 노드 크기 (중심성 기반)
            centrality = nx.degree_centrality(G)
            node_sizes = [centrality[node] * 3000 + 100 for node in G.nodes()]
            
            # 엣지 가중치 기반 색상
            edges = G.edges()
            weights = [G[u][v].get('weight', 1) for u, v in edges]
            
            nx.draw_networkx_nodes(G, pos, ax=axes[0, 0], 
                                 node_size=node_sizes, 
                                 node_color=self.colors['primary'], 
                                 alpha=0.7)
            nx.draw_networkx_edges(G, pos, ax=axes[0, 0], 
                                 width=[w/max(weights)*3 for w in weights],
                                 alpha=0.5, edge_color='gray')
            nx.draw_networkx_labels(G, pos, ax=axes[0, 0], 
                                  font_size=8, font_weight='bold')
            
            axes[0, 0].set_title('키워드 공출현 네트워크', fontsize=14, fontweight='bold')
            axes[0, 0].axis('off')
            
            # 2. 중심성 분석
            centrality_analysis = network_results.get('centrality_analysis', {})
            
            if centrality_analysis:
                centrality_types = ['degree', 'betweenness', 'closeness', 'eigenvector']
                centrality_data = []
                
                for cent_type in centrality_types:
                    if cent_type in centrality_analysis:
                        top_nodes = list(centrality_analysis[cent_type].items())[:10]
                        centrality_data.extend([(node, score, cent_type) for node, score in top_nodes])
                
                if centrality_data:
                    cent_df = pd.DataFrame(centrality_data, columns=['노드', '점수', '중심성유형'])
                    
                    # 중심성 유형별 상위 노드들
                    for i, cent_type in enumerate(centrality_types[:4]):
                        if cent_type in centrality_analysis:
                            top_nodes = list(centrality_analysis[cent_type].items())[:8]
                            nodes, scores = zip(*top_nodes) if top_nodes else ([], [])
                            
                            row, col = (0, 1) if i < 2 else (1, 0)
                            if i % 2 == 1:
                                col = 1
                            
                            if i < 4 and len(nodes) > 0:
                                axes[row, col].barh(range(len(nodes)), scores, 
                                                   color=plt.cm.viridis(np.linspace(0, 1, len(nodes))))
                                axes[row, col].set_title(f'{cent_type.title()} 중심성', fontsize=12, fontweight='bold')
                                axes[row, col].set_xlabel('중심성 점수')
                                axes[row, col].set_yticks(range(len(nodes)))
                                axes[row, col].set_yticklabels(nodes, fontsize=10)
                                axes[row, col].grid(True, alpha=0.3)
            
            # 3. 커뮤니티 탐지 결과
            communities = network_results.get('communities', [])
            if communities and len(communities) > 1:
                # 커뮤니티별 색상 지정
                community_colors = plt.cm.Set3(np.linspace(0, 1, len(communities)))
                node_colors = {}
                
                for i, community in enumerate(communities):
                    for node in community:
                        node_colors[node] = community_colors[i]
                
                # 커뮤니티 네트워크 시각화
                pos_community = nx.spring_layout(G, k=1, iterations=50)
                
                for i, community in enumerate(communities):
                    subgraph = G.subgraph(community)
                    nx.draw_networkx_nodes(subgraph, pos_community, ax=axes[1, 1],
                                         node_color=[community_colors[i]], 
                                         node_size=300, alpha=0.8,
                                         label=f'커뮤니티 {i+1}')
                
                nx.draw_networkx_edges(G, pos_community, ax=axes[1, 1], 
                                     alpha=0.3, edge_color='gray')
                nx.draw_networkx_labels(G, pos_community, ax=axes[1, 1], 
                                      font_size=8)
                
                axes[1, 1].set_title(f'커뮤니티 탐지 결과 ({len(communities)}개)', fontsize=14, fontweight='bold')
                axes[1, 1].legend()
                axes[1, 1].axis('off')
            
            plt.tight_layout()
            
            # 저장
            filename = f'{case_name}_keyword_network.png'
            filepath = os.path.join(self.output_dir, filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            self.logger.info(f"✅ {case_name} 키워드 네트워크 시각화 완료: {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"❌ 키워드 네트워크 시각화 실패: {str(e)}")
            return None
    
    def visualize_media_frame_comparison(self, comparison_results: Dict, case_name: str) -> str:
        """
        언론 프레임 유사도 시각화
        
        Args:
            comparison_results: 프레임 비교 결과
            case_name: 사건명
        
        Returns:
            저장된 파일 경로
        """
        try:
            self.logger.info(f"📰 {case_name} 언론 프레임 유사도 시각화 시작")
            
            self._apply_font_settings()
            
            similarity_results = comparison_results.get('similarity_results', {})
            if not similarity_results:
                self.logger.warning("⚠️ 유사도 결과가 없습니다.")
                return None
            
            fig, axes = plt.subplots(2, 2, figsize=(20, 15))
            fig.suptitle(f'{case_name} 언론 프레임 vs 대중 반응 비교', fontsize=20, fontweight='bold')
            
            # 데이터 준비
            periods = sorted(similarity_results.keys())
            similarities = [similarity_results[p]['cosine_similarity'] for p in periods]
            frame_alignments = [similarity_results[p]['frame_alignment'] for p in periods]
            
            # 1. 시간별 유사도 트렌드
            axes[0, 0].plot(range(len(periods)), similarities, 
                           marker='o', linewidth=3, markersize=8, 
                           color=self.colors['primary'], label='코사인 유사도')
            
            # 임계값 선 표시
            axes[0, 0].axhline(y=0.7, color=self.colors['positive'], 
                              linestyle='--', alpha=0.7, label='높은 일치 (0.7)')
            axes[0, 0].axhline(y=0.4, color=self.colors['warning'], 
                              linestyle='--', alpha=0.7, label='중간 일치 (0.4)')
            
            axes[0, 0].set_title('시간별 언론-대중 프레임 유사도', fontsize=14, fontweight='bold')
            axes[0, 0].set_xlabel('시간 구간')
            axes[0, 0].set_ylabel('코사인 유사도')
            axes[0, 0].set_xticks(range(len(periods)))
            axes[0, 0].set_xticklabels(periods, rotation=45, ha='right')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].set_ylim(0, 1)
            
            # 2. 프레임 일치도 분포
            alignment_counts = Counter(frame_alignments)
            alignment_labels = list(alignment_counts.keys())
            alignment_values = list(alignment_counts.values())
            
            colors_alignment = [self.colors['positive'] if x == 'high' 
                              else self.colors['warning'] if x == 'medium' 
                              else self.colors['negative'] for x in alignment_labels]
            
            axes[0, 1].pie(alignment_values, labels=alignment_labels, autopct='%1.1f%%',
                          colors=colors_alignment, startangle=90)
            axes[0, 1].set_title('프레임 일치도 분포', fontsize=14, fontweight='bold')
            
            # 3. 고유 키워드 분석 (댓글 vs 언론)
            unique_comment_keywords = []
            unique_media_keywords = []
            
            for period in periods:
                period_data = similarity_results[period]
                unique_comment_keywords.extend([kw[0] if isinstance(kw, tuple) else kw 
                                              for kw in period_data.get('unique_comment_keywords', [])])
                unique_media_keywords.extend([kw[0] if isinstance(kw, tuple) else kw 
                                            for kw in period_data.get('unique_media_keywords', [])])
            
            # 상위 고유 키워드들
            comment_counter = Counter(unique_comment_keywords)
            media_counter = Counter(unique_media_keywords)
            
            top_comment_unique = comment_counter.most_common(10)
            top_media_unique = media_counter.most_common(10)
            
            if top_comment_unique:
                words, counts = zip(*top_comment_unique)
                axes[1, 0].barh(range(len(words)), counts, 
                               color=self.colors['accent'], alpha=0.8)
                axes[1, 0].set_title('댓글 고유 키워드 (언론에 없는)', fontsize=14, fontweight='bold')
                axes[1, 0].set_xlabel('빈도')
                axes[1, 0].set_yticks(range(len(words)))
                axes[1, 0].set_yticklabels(words, fontsize=10)
                axes[1, 0].grid(True, alpha=0.3)
            
            if top_media_unique:
                words, counts = zip(*top_media_unique)
                axes[1, 1].barh(range(len(words)), counts, 
                               color=self.colors['secondary'], alpha=0.8)
                axes[1, 1].set_title('언론 고유 키워드 (댓글에 없는)', fontsize=14, fontweight='bold')
                axes[1, 1].set_xlabel('빈도')
                axes[1, 1].set_yticks(range(len(words)))
                axes[1, 1].set_yticklabels(words, fontsize=10)
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 저장
            filename = f'{case_name}_media_frame_comparison.png'
            filepath = os.path.join(self.output_dir, filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            self.logger.info(f"✅ {case_name} 언론 프레임 유사도 시각화 완료: {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"❌ 언론 프레임 유사도 시각화 실패: {str(e)}")
            return None
    
    def create_comprehensive_dashboard(self, all_results: Dict, case_name: str) -> str:
        """
        종합 인터랙티브 대시보드 생성 (Plotly)
        
        Args:
            all_results: 모든 분석 결과
            case_name: 사건명
        
        Returns:
            저장된 HTML 파일 경로
        """
        try:
            self.logger.info(f"📊 {case_name} 종합 대시보드 생성 시작")
            
            # 서브플롯 생성
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=[
                    '시간별 감성 트렌드', '토픽 분포',
                    '키워드 빈도', '프레임 유사도',
                    '댓글 수 변화', '감성 변동성'
                ],
                specs=[[{"secondary_y": False}, {"type": "pie"}],
                       [{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": True}, {"secondary_y": False}]]
            )
            
            # 시간별 데이터 준비
            temporal_results = all_results.get('temporal_flow', {}).get('temporal_results', {})
            if temporal_results:
                periods = sorted(temporal_results.keys())
                sentiment_scores = [temporal_results[p]['avg_sentiment'] for p in periods]
                comment_counts = [temporal_results[p]['comment_count'] for p in periods]
                volatilities = [temporal_results[p]['sentiment_volatility'] for p in periods]
                
                # 1. 감성 트렌드
                fig.add_trace(
                    go.Scatter(x=periods, y=sentiment_scores, 
                             mode='lines+markers', name='감성 점수',
                             line=dict(color=self.colors['primary'], width=3)),
                    row=1, col=1
                )
                
                # 5. 댓글 수 변화
                fig.add_trace(
                    go.Bar(x=periods, y=comment_counts, name='댓글 수',
                          marker_color=self.colors['accent']),
                    row=3, col=1
                )
                
                # 6. 감성 변동성
                fig.add_trace(
                    go.Scatter(x=periods, y=volatilities, 
                             mode='lines+markers', name='감성 변동성',
                             line=dict(color=self.colors['negative'], width=2)),
                    row=3, col=2
                )
            
            # 2. 토픽 분포 (파이 차트)
            topic_results = all_results.get('comprehensive_topics', {}).get('topic_results', {})
            if topic_results and 'lda' in topic_results:
                lda_topics = topic_results['lda'].get('topics', [])
                if lda_topics:
                    topic_labels = [f"토픽 {i+1}" for i in range(len(lda_topics[:5]))]
                    topic_weights = [1] * len(topic_labels)  # 동일 가중치로 설정
                    
                    fig.add_trace(
                        go.Pie(labels=topic_labels, values=topic_weights, name="토픽 분포"),
                        row=1, col=2
                    )
            
            # 3. 키워드 빈도
            if temporal_results:
                all_keywords = {}
                for period_data in temporal_results.values():
                    for keyword, freq in period_data.get('top_keywords', [])[:10]:
                        if keyword in all_keywords:
                            all_keywords[keyword] += freq
                        else:
                            all_keywords[keyword] = freq
                
                top_keywords = sorted(all_keywords.items(), key=lambda x: x[1], reverse=True)[:15]
                if top_keywords:
                    keywords, freqs = zip(*top_keywords)
                    
                    fig.add_trace(
                        go.Bar(x=list(freqs), y=list(keywords), orientation='h',
                              name='키워드 빈도', marker_color=self.colors['info']),
                        row=2, col=1
                    )
            
            # 4. 프레임 유사도
            comparison_results = all_results.get('media_frame_comparison', {})
            if comparison_results:
                similarity_results = comparison_results.get('similarity_results', {})
                if similarity_results:
                    periods = sorted(similarity_results.keys())
                    similarities = [similarity_results[p]['cosine_similarity'] for p in periods]
                    
                    fig.add_trace(
                        go.Scatter(x=periods, y=similarities, 
                                 mode='lines+markers', name='프레임 유사도',
                                 line=dict(color=self.colors['warning'], width=3)),
                        row=2, col=2
                    )
            
            # 레이아웃 설정
            fig.update_layout(
                title=f'{case_name} 종합 분석 대시보드',
                title_font_size=24,
                showlegend=True,
                height=1200,
                font=dict(family="Arial, sans-serif", size=12)
            )
            
            # 축 레이블 설정
            fig.update_xaxes(title_text="시간 구간", row=1, col=1)
            fig.update_yaxes(title_text="감성 점수", row=1, col=1)
            
            fig.update_xaxes(title_text="빈도", row=2, col=1)
            fig.update_yaxes(title_text="키워드", row=2, col=1)
            
            fig.update_xaxes(title_text="시간 구간", row=2, col=2)
            fig.update_yaxes(title_text="유사도", row=2, col=2)
            
            fig.update_xaxes(title_text="시간 구간", row=3, col=1)
            fig.update_yaxes(title_text="댓글 수", row=3, col=1)
            
            fig.update_xaxes(title_text="시간 구간", row=3, col=2)
            fig.update_yaxes(title_text="변동성", row=3, col=2)
            
            # HTML 파일로 저장
            filename = f'{case_name}_comprehensive_dashboard.html'
            filepath = os.path.join(self.output_dir, filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            fig.write_html(filepath)
            
            self.logger.info(f"✅ {case_name} 종합 대시보드 생성 완료: {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"❌ 종합 대시보드 생성 실패: {str(e)}")
            return None 