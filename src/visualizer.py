"""
Visualizer Module
시각화 모듈
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
import logging
from typing import Dict, List, Tuple, Optional
import os
from datetime import datetime

# 워드클라우드
from wordcloud import WordCloud
import networkx as nx
from collections import Counter

# 한국어 폰트 설정
import platform

class Visualizer:
    """시각화 클래스"""
    
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
        """강화된 한글 폰트 설정"""
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
                    # 폰트 경로 찾기
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
                
                # plt.rcParams도 동시 업데이트
                plt.rcParams['font.family'] = found_font
                plt.rcParams['font.sans-serif'] = [found_font] + ['DejaVu Sans', 'Arial']
                plt.rcParams['axes.unicode_minus'] = False
                plt.rcParams['font.size'] = 12
                
                self.logger.info(f"✅ 한글 폰트 설정 완료: {found_font} ({found_font_path})")
            else:
                # 폴백: 기본 폰트 사용
                self.korean_font_name = "DejaVu Sans"
                self.korean_font_path = None
                
                rcParams['font.family'] = 'DejaVu Sans'
                rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
                rcParams['axes.unicode_minus'] = False
                rcParams['font.size'] = 12
                
                plt.rcParams['font.family'] = 'DejaVu Sans'
                plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
                plt.rcParams['axes.unicode_minus'] = False
                plt.rcParams['font.size'] = 12
                
                self.logger.warning("⚠️ 한글 폰트를 찾을 수 없어 기본 폰트 사용: DejaVu Sans")
                
        except Exception as e:
            self.logger.error(f"폰트 설정 중 오류 발생: {e}")
            self.korean_font_name = "DejaVu Sans"
            self.korean_font_path = None
            rcParams['font.family'] = 'DejaVu Sans'
            rcParams['axes.unicode_minus'] = False
    
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
        sns.set_palette(self.config.VISUALIZATION['color_palette'])
        
        # 기본 설정
        rcParams['figure.figsize'] = self.config.VISUALIZATION['figsize']
        rcParams['figure.dpi'] = self.config.VISUALIZATION['dpi']
        rcParams['font.size'] = self.config.VISUALIZATION['font_size']
        rcParams['axes.grid'] = True
        rcParams['grid.alpha'] = 0.3
    
    def plot_sentiment_trends(self, sentiment_trends: Dict, target_name: str) -> str:
        """
        감성 트렌드 시각화 (6개 감정 + 변화 속도 중심으로 개선)
        Args:
            sentiment_trends: 감성 트렌드 데이터
            target_name: 분석 대상 이름
        Returns:
            저장된 파일 경로
        """
        try:
            self.logger.info(f"📊 {target_name} 감성 트렌드 시각화 시작")
            
            # 폰트 설정 재적용
            self._apply_font_settings()
            
            fig, axes = plt.subplots(2, 2, figsize=(20, 15))
            fig.suptitle(f'{target_name} 감성 분석: 시간적 변화 패턴', fontsize=20, fontweight='bold')
            
            months = sentiment_trends['months']
            positive_ratios = sentiment_trends['positive_ratios']
            negative_ratios = sentiment_trends['negative_ratios']
            emotion_trends = sentiment_trends['emotion_trends']
            
            # 1. 긍정/부정 비율 트렌드 + 변화율
            axes[0, 0].plot(months, positive_ratios, marker='o', linewidth=3, 
                           label='긍정', color='#2E8B57', markersize=8)
            axes[0, 0].plot(months, negative_ratios, marker='s', linewidth=3, 
                           label='부정', color='#DC143C', markersize=8)
            
            # 변화율 계산 및 표시
            if len(positive_ratios) > 1:
                pos_changes = np.diff(positive_ratios)
                neg_changes = np.diff(negative_ratios)
                
                # 급격한 변화 구간 하이라이트
                for i, (pos_change, neg_change) in enumerate(zip(pos_changes, neg_changes)):
                    if abs(pos_change) > 0.1 or abs(neg_change) > 0.1:  # 10% 이상 변화
                        axes[0, 0].axvspan(i, i+1, alpha=0.2, color='yellow', 
                                          label='급변구간' if i == 0 else "")
            
            axes[0, 0].set_title('긍정/부정 감성 변화 + 급변구간', fontsize=16, fontweight='bold')
            axes[0, 0].set_ylabel('비율', fontsize=14)
            axes[0, 0].legend(fontsize=12)
            axes[0, 0].tick_params(axis='x', rotation=45)
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. 6개 감정 상세 분석
            emotion_colors = {
                '분노': '#FF4444', '슬픔': '#4444FF', '불안': '#FF8800',
                '상처': '#8800FF', '당황': '#00FF88', '기쁨': '#FFDD00'
            }
            
            # 6개 감정 트렌드 표시
            for emotion, values in emotion_trends.items():
                if emotion in emotion_colors:
                    axes[0, 1].plot(months, values, marker='o', linewidth=2.5, 
                                   label=emotion, color=emotion_colors[emotion], 
                                   markersize=6, alpha=0.8)
            
            axes[0, 1].set_title('6개 감정 상세 변화 추이', fontsize=16, fontweight='bold')
            axes[0, 1].set_ylabel('감정 비율', fontsize=14)
            axes[0, 1].legend(fontsize=11, loc='upper right')
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].grid(True, alpha=0.3)
            
            # 최고점 표시
            for emotion, values in emotion_trends.items():
                if emotion in emotion_colors and max(values) > 0.05:  # 5% 이상인 감정만
                    max_idx = np.argmax(values)
                    max_value = values[max_idx]
                    axes[0, 1].annotate(f'{max_value:.1%}', 
                                       xy=(months[max_idx], max_value),
                                       xytext=(5, 5), textcoords='offset points',
                                       fontsize=9, alpha=0.8,
                                       bbox=dict(boxstyle='round,pad=0.3', 
                                               facecolor=emotion_colors[emotion], 
                                               alpha=0.3))
            
            # 3. 감정 변화 속도 (1차 미분) - 주요 감정만
            emotion_velocities = {}
            significant_emotions = []
            
            # 유의미한 감정 선별 (최대값이 3% 이상)
            for emotion, values in emotion_trends.items():
                if emotion in emotion_colors and max(values) > 0.03:
                    significant_emotions.append((emotion, max(values)))
                    if len(values) > 1:
                        velocity = np.diff(values)
                        emotion_velocities[emotion] = velocity
            
            # 최대값 기준으로 정렬하여 상위 4개만 표시
            significant_emotions.sort(key=lambda x: x[1], reverse=True)
            top_emotions = [emotion for emotion, _ in significant_emotions[:4]]
            
            for emotion in top_emotions:
                if emotion in emotion_velocities:
                    velocity_months = months[1:]  # 미분이므로 길이가 1 줄어듦
                    axes[1, 0].plot(velocity_months, emotion_velocities[emotion], 
                                   marker='o', linewidth=2.5, label=f'{emotion} 변화율',
                                   color=emotion_colors[emotion], alpha=0.8, markersize=6)
            
            axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[1, 0].set_title('주요 감정 변화 속도 (증가/감소 추세)', fontsize=16, fontweight='bold')
            axes[1, 0].set_ylabel('변화율 (월간)', fontsize=14)
            axes[1, 0].legend(fontsize=11, loc='upper right')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True, alpha=0.3)
            
            # 급격한 변화 구간 표시
            for emotion in top_emotions:
                if emotion in emotion_velocities:
                    velocity = emotion_velocities[emotion]
                    threshold = np.std(velocity) * 1.5  # 1.5 표준편차 이상
                    for i, vel in enumerate(velocity):
                        if abs(vel) > threshold:
                            axes[1, 0].scatter(velocity_months[i], vel, 
                                             color=emotion_colors[emotion], 
                                             s=100, alpha=0.7, marker='*')
            
            # 4. 감정 다양성 및 지배적 감정 분석
            emotion_diversity = []
            dominant_emotions = []
            dominant_emotion_values = []
            
            for i in range(len(months)):
                month_emotions = [emotion_trends[emotion][i] for emotion in emotion_trends.keys()]
                # 엔트로피 계산 (감정 다양성)
                month_emotions = np.array(month_emotions)
                month_emotions = month_emotions[month_emotions > 0]  # 0 제거
                if len(month_emotions) > 0:
                    # 정규화
                    month_emotions = month_emotions / np.sum(month_emotions)
                    entropy = -np.sum(month_emotions * np.log(month_emotions + 1e-10))
                    emotion_diversity.append(entropy)
                    
                    # 지배적 감정
                    emotion_values = [emotion_trends[emotion][i] for emotion in emotion_trends.keys()]
                    max_emotion_idx = np.argmax(emotion_values)
                    emotion_names = list(emotion_trends.keys())
                    dominant_emotion = emotion_names[max_emotion_idx]
                    dominant_emotions.append(dominant_emotion)
                    dominant_emotion_values.append(emotion_values[max_emotion_idx])
                else:
                    emotion_diversity.append(0)
                    dominant_emotions.append('중립')
                    dominant_emotion_values.append(0)
            
            # 다양성 지수 라인 플롯
            line = axes[1, 1].plot(months, emotion_diversity, marker='o', linewidth=3, 
                                  color='#4169E1', markersize=8, label='감정 다양성')
            axes[1, 1].set_ylabel('다양성 지수', fontsize=14, color='#4169E1')
            axes[1, 1].tick_params(axis='y', labelcolor='#4169E1')
            
            # 지배적 감정을 색상으로 표시 (우측 y축)
            ax2 = axes[1, 1].twinx()
            for i, (month, emotion, value) in enumerate(zip(months, dominant_emotions, dominant_emotion_values)):
                if emotion in emotion_colors:
                    ax2.bar(month, value, color=emotion_colors[emotion], alpha=0.6, width=0.8)
            
            ax2.set_ylabel('지배적 감정 강도', fontsize=14, color='#DC143C')
            ax2.tick_params(axis='y', labelcolor='#DC143C')
            
            axes[1, 1].set_title('감정 다양성 vs 지배적 감정', fontsize=16, fontweight='bold')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].grid(True, alpha=0.3)
            
            # 범례 추가
            emotion_legend = [plt.Rectangle((0,0),1,1, color=color, alpha=0.6) 
                             for emotion, color in emotion_colors.items()]
            emotion_labels = list(emotion_colors.keys())
            axes[1, 1].legend(emotion_legend, emotion_labels, 
                             loc='upper left', fontsize=10, title='지배적 감정')
            
            plt.tight_layout()
            
            # 파일 저장
            filename = f"{target_name}_sentiment_trends.png"
            filepath = os.path.join(self.output_dir, filename)
            
            # 디렉토리 생성
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            plt.savefig(filepath, dpi=self.config.VISUALIZATION['dpi'], 
                       bbox_inches='tight', facecolor='white')
            plt.close()
            
            self.logger.info(f"✅ 감성 트렌드 시각화 완료: {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"❌ 감성 트렌드 시각화 실패: {str(e)}")
            raise
    
    def _save_period_topic_changes_to_file(self, comprehensive_results: Dict, target_name: str, time_unit: str):
        """
        기간별 토픽 변화를 텍스트 파일로 저장
        Args:
            comprehensive_results: 종합 분석 결과
            target_name: 분석 대상 이름  
            time_unit: 시간 단위
        """
        try:
            # BERTopic 기간별 변화 분석
            bertopic_changes = []
            lda_changes = []
            
            sorted_periods = sorted(comprehensive_results.keys())
            
            for period in sorted_periods:
                result = comprehensive_results[period]
                
                # BERTopic 분석
                if 'bertopic' in result and result['bertopic'].get('topic_labels'):
                    bertopic_labels = result['bertopic']['topic_labels']
                    bertopic_changes.append(f"=== {period} ===")
                    bertopic_changes.append(f"총 댓글 수: {result['total_comments']:,}개")
                    bertopic_changes.append(f"발견된 토픽 수: {len(bertopic_labels)}개")
                    bertopic_changes.append("주요 토픽:")
                    for i, label in enumerate(bertopic_labels[:5], 1):
                        bertopic_changes.append(f"  {i}. {label}")
                    bertopic_changes.append("")
                
                # LDA 분석
                if 'lda' in result and result['lda'].get('topic_labels'):
                    lda_labels = result['lda']['topic_labels']
                    lda_changes.append(f"=== {period} ===")
                    lda_changes.append(f"총 댓글 수: {result['total_comments']:,}개")
                    lda_changes.append(f"발견된 토픽 수: {len(lda_labels)}개")
                    lda_changes.append(f"일관성 점수: {result['lda'].get('coherence_score', 0):.3f}")
                    lda_changes.append("주요 토픽:")
                    for i, label in enumerate(lda_labels[:5], 1):
                        lda_changes.append(f"  {i}. {label}")
                    lda_changes.append("")
            
            # BERTopic 변화 파일 저장
            if bertopic_changes:
                bertopic_file = os.path.join(
                    self.config.OUTPUT_STRUCTURE['data_processed'],
                    f'{target_name}_BERTopic_{time_unit}_변화.txt'
                )
                with open(bertopic_file, 'w', encoding='utf-8') as f:
                    f.write(f"{target_name} BERTopic 기간별 변화 분석\n")
                    f.write(f"시간 단위: {time_unit}\n")
                    f.write(f"분석 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("=" * 60 + "\n\n")
                    f.write("\n".join(bertopic_changes))
                
                self.logger.info(f"📄 BERTopic 기간별 변화 저장: {bertopic_file}")
            
            # LDA 변화 파일 저장
            if lda_changes:
                lda_file = os.path.join(
                    self.config.OUTPUT_STRUCTURE['data_processed'],
                    f'{target_name}_LDA_{time_unit}_변화.txt'
                )
                with open(lda_file, 'w', encoding='utf-8') as f:
                    f.write(f"{target_name} LDA 기간별 변화 분석\n")
                    f.write(f"시간 단위: {time_unit}\n")
                    f.write(f"분석 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("=" * 60 + "\n\n")
                    f.write("\n".join(lda_changes))
                
                self.logger.info(f"📄 LDA 기간별 변화 저장: {lda_file}")
                
        except Exception as e:
            self.logger.error(f"❌ 기간별 토픽 변화 파일 저장 실패: {str(e)}")
    
    def _plot_topic_count_trends(self, filtered_results: Dict, ax, model_name: str):
        """
        토픽 수 변화 트렌드 플롯
        Args:
            filtered_results: 필터링된 결과
            ax: matplotlib axes
            model_name: 모델 이름 ('BERTopic' 또는 'LDA')
        """
        try:
            periods = sorted(filtered_results.keys())
            topic_counts = []
            
            model_key = 'bertopic' if model_name == 'BERTopic' else 'lda'
            
            for period in periods:
                result = filtered_results[period]
                if model_key in result and result[model_key].get('topic_labels'):
                    topic_counts.append(len(result[model_key]['topic_labels']))
                else:
                    topic_counts.append(0)
            
            # 플롯 생성
            ax.plot(range(len(periods)), topic_counts, marker='o', linewidth=2, markersize=8,
                   color='#4CAF50' if model_name == 'BERTopic' else '#FF9800')
            ax.set_title(f'{model_name} 토픽 수 변화', fontsize=14, fontweight='bold')
            ax.set_ylabel('토픽 수', fontsize=12)
            ax.set_xticks(range(len(periods)))
            ax.set_xticklabels([p[:10] + '...' if len(p) > 10 else p for p in periods], 
                                      rotation=45, ha='right')
            ax.grid(True, alpha=0.3)
            
            # 값 표시
            for i, count in enumerate(topic_counts):
                if count > 0:
                    ax.text(i, count + 0.1, str(count), ha='center', va='bottom', fontsize=10)
                    
        except Exception as e:
            self.logger.error(f"❌ {model_name} 토픽 수 트렌드 플롯 실패: {str(e)}")
            ax.text(0.5, 0.5, f'{model_name} 데이터 없음', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)

    def _plot_lda_analysis(self, filtered_results: Dict, ax):
        """
        LDA 분석 결과 플롯 (토픽 수 + 일관성)
        Args:
            filtered_results: 필터링된 결과
            ax: matplotlib axes
        """
        try:
            periods = sorted(filtered_results.keys())
            topic_counts = []
            coherence_scores = []
            
            for period in periods:
                result = filtered_results[period]
                if 'lda' in result and result['lda'].get('topic_labels'):
                    topic_counts.append(len(result['lda']['topic_labels']))
                    coherence_scores.append(result['lda'].get('coherence_score', 0))
                else:
                    topic_counts.append(0)
                    coherence_scores.append(0)
            
            # 이중 y축 설정
            ax2 = ax.twinx()
                
            # 토픽 수 (막대 그래프)
            bars = ax.bar(range(len(periods)), topic_counts, alpha=0.7, color='#FF9800', 
                         label='토픽 수')
            
            # 일관성 점수 (선 그래프)
            line = ax2.plot(range(len(periods)), coherence_scores, 'r-o', linewidth=2, 
                           markersize=6, label='일관성 점수')
            
            ax.set_title('LDA 토픽 수 및 일관성 변화', fontsize=14, fontweight='bold')
            ax.set_ylabel('토픽 수', fontsize=12, color='#FF9800')
            ax2.set_ylabel('일관성 점수', fontsize=12, color='red')
            
            ax.set_xticks(range(len(periods)))
            ax.set_xticklabels([p[:10] + '...' if len(p) > 10 else p for p in periods], 
                              rotation=45, ha='right')
            
            ax.grid(True, alpha=0.3)
            
            # 범례
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
                
            # 값 표시
            for i, (count, score) in enumerate(zip(topic_counts, coherence_scores)):
                if count > 0:
                    ax.text(i, count + 0.1, str(count), ha='center', va='bottom', fontsize=9)
                if score > 0:
                    ax2.text(i, score + 0.01, f'{score:.2f}', ha='center', va='bottom', 
                            fontsize=9, color='red')
                    
        except Exception as e:
            self.logger.error(f"❌ LDA 분석 플롯 실패: {str(e)}")
            ax.text(0.5, 0.5, 'LDA 데이터 없음', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)

    def create_topic_analysis_dashboard(self, comprehensive_results: Dict, target_name: str, 
                                      time_unit: str, min_comments_threshold: int = 50) -> str:
        """
        토픽 분석 대시보드 생성 (1x2 레이아웃)
        Args:
            comprehensive_results: 종합 분석 결과
            target_name: 분석 대상 이름
            time_unit: 시간 단위
            min_comments_threshold: 최소 댓글 수 임계값
        Returns:
            저장된 파일 경로
        """
        try:
            self.logger.info(f"🔍 {target_name} 토픽 분석 대시보드 생성 시작")
            
            # 기간별 토픽 변화를 텍스트 파일로 저장
            self._save_period_topic_changes_to_file(comprehensive_results, target_name, time_unit)
            
            # 충분한 데이터가 있는 기간만 필터링
            filtered_results = {
                period: result for period, result in comprehensive_results.items()
                if result['total_comments'] >= min_comments_threshold
            }
            
            if not filtered_results:
                self.logger.warning(f"⚠️ 최소 댓글 수({min_comments_threshold})를 만족하는 기간이 없습니다.")
                return None
            
            # 1x2 레이아웃 설정
            fig, axes = plt.subplots(1, 2, figsize=(20, 8))
            fig.suptitle(f'{target_name} 토픽 분석 대시보드 ({time_unit})', fontsize=16, fontweight='bold')
            
            # 1. BERTopic 토픽 수 변화
            self._plot_topic_count_trends(filtered_results, axes[0], 'BERTopic')
            
            # 2. LDA 토픽 수 변화 및 일관성
            self._plot_lda_analysis(filtered_results, axes[1])
            
            plt.tight_layout()
            
            # 파일 저장
            filename = f'{target_name}_topic_dashboard_{time_unit}.png'
            filepath = os.path.join(self.config.OUTPUT_STRUCTURE['visualizations'], filename)
            plt.savefig(filepath, dpi=self.config.VISUALIZATION['dpi'], bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"✅ 토픽 분석 대시보드 저장: {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"❌ 토픽 분석 대시보드 생성 실패: {str(e)}")
            return None
    
    def plot_topic_evolution(self, topic_evolution: Dict, target_name: str) -> str:
        """
        토픽 진화 시각화 (프레임 변화 중심으로 개선)
        Args:
            topic_evolution: 토픽 진화 데이터
            target_name: 분석 대상 이름
        Returns:
            저장된 파일 경로
        """
        try:
            self.logger.info(f"🔍 {target_name} 토픽 진화 시각화 시작")
            
            # 폰트 설정 재적용
            self._apply_font_settings()
            plt.rcParams['axes.unicode_minus'] = False
            
            fig, axes = plt.subplots(2, 2, figsize=(20, 15))
            fig.suptitle(f'{target_name} 토픽 프레임 변화 분석', fontsize=20, fontweight='bold')
            
            bertopic_data = topic_evolution['bertopic_evolution']
            lda_data = topic_evolution['lda_evolution']
            
            months = bertopic_data['months']
            
            # 1. 토픽 복잡도 변화 (토픽 수 + 일관성)
            axes[0, 0].plot(months, bertopic_data['topic_counts'], marker='o', 
                           linewidth=3, label='BERTopic 토픽 수', color='#4CAF50', markersize=8)
            
            # 이차 y축으로 일관성 점수 추가
            ax2 = axes[0, 0].twinx()
            if lda_data['coherence_scores']:
                ax2.plot(months, lda_data['coherence_scores'], marker='D', 
                        linewidth=3, color='#9C27B0', markersize=8, label='LDA 일관성')
                ax2.set_ylabel('일관성 점수', fontsize=14, color='#9C27B0')
                ax2.tick_params(axis='y', labelcolor='#9C27B0')
            
            axes[0, 0].set_title('토픽 복잡도 vs 일관성 변화', fontsize=16, fontweight='bold')
            axes[0, 0].set_ylabel('토픽 수', fontsize=14, color='#4CAF50')
            axes[0, 0].tick_params(axis='y', labelcolor='#4CAF50')
            axes[0, 0].tick_params(axis='x', rotation=45)
            axes[0, 0].grid(True, alpha=0.3)
            
            # 범례 통합
            lines1, labels1 = axes[0, 0].get_legend_handles_labels()
            if lda_data['coherence_scores']:
                lines2, labels2 = ax2.get_legend_handles_labels()
                axes[0, 0].legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            else:
                axes[0, 0].legend()
            
            # 2. 토픽 전환 패턴 (주요 토픽의 등장/소멸)
            # 각 기간별 주요 토픽 변화를 추적
            topic_transitions = []
            prev_topics = set()
            
            for i, month in enumerate(months):
                current_topics = set()
                if i < len(bertopic_data['main_topics']):
                    current_topic = bertopic_data['main_topics'][i]
                    if current_topic:
                        current_topics.add(current_topic)
                
                # 새로 등장한 토픽
                new_topics = current_topics - prev_topics
                # 사라진 토픽
                disappeared_topics = prev_topics - current_topics
                
                transition_score = len(new_topics) + len(disappeared_topics)
                topic_transitions.append(transition_score)
                prev_topics = current_topics
            
            bars = axes[0, 1].bar(months, topic_transitions, color='#FF6B6B', alpha=0.7)
            axes[0, 1].set_title('토픽 전환 강도 (프레임 변화)', fontsize=16, fontweight='bold')
            axes[0, 1].set_ylabel('전환 점수', fontsize=14)
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].grid(True, alpha=0.3)
            
            # 높은 전환 구간 하이라이트
            max_transition = max(topic_transitions) if topic_transitions else 0
            for i, (bar, score) in enumerate(zip(bars, topic_transitions)):
                if score > max_transition * 0.7:  # 상위 30% 전환 구간
                    bar.set_color('#FF4444')
                    axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                                   '급변', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            # 3. 토픽 지속성 분석 (연속성 vs 단발성)
            topic_persistence = {}
            for i, topic in enumerate(bertopic_data['main_topics']):
                if topic:
                    if topic not in topic_persistence:
                        topic_persistence[topic] = []
                    topic_persistence[topic].append(i)
            
            # 지속성 점수 계산 (연속된 기간의 길이)
            persistence_scores = []
            for month_idx in range(len(months)):
                current_topic = bertopic_data['main_topics'][month_idx] if month_idx < len(bertopic_data['main_topics']) else None
                if current_topic and current_topic in topic_persistence:
                    appearances = topic_persistence[current_topic]
                    # 현재 시점 기준 연속성 계산
                    consecutive_count = 1
                    for j in range(month_idx - 1, -1, -1):
                        if j in appearances:
                            consecutive_count += 1
                        else:
                            break
                    persistence_scores.append(consecutive_count)
                else:
                    persistence_scores.append(0)
            
            axes[1, 0].plot(months, persistence_scores, marker='o', linewidth=3, 
                           color='#4ECDC4', markersize=8)
            axes[1, 0].set_title('토픽 지속성 (연속 출현 기간)', fontsize=16, fontweight='bold')
            axes[1, 0].set_ylabel('연속 기간', fontsize=14)
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True, alpha=0.3)
            
            # 지속성 구간별 색상 구분
            for i, score in enumerate(persistence_scores):
                if score >= 3:  # 3개월 이상 지속
                    axes[1, 0].scatter(months[i], score, color='green', s=100, alpha=0.7, label='장기 지속' if i == 0 else "")
                elif score == 2:
                    axes[1, 0].scatter(months[i], score, color='orange', s=100, alpha=0.7, label='중기 지속' if i == 0 else "")
                elif score == 1:
                    axes[1, 0].scatter(months[i], score, color='red', s=100, alpha=0.7, label='단발성' if i == 0 else "")
            
            # 4. 토픽 키워드 변화율 (의미적 변화)
            # 각 기간별 주요 키워드의 변화를 측정
            keyword_changes = []
            prev_keywords = set()
            
            for i, month in enumerate(months):
                current_keywords = set()
                
                # BERTopic에서 현재 기간의 주요 키워드 추출
                if 'topic_words' in bertopic_data and i < len(bertopic_data.get('topic_words', [])):
                    topic_words = bertopic_data['topic_words'][i] if isinstance(bertopic_data['topic_words'], list) else {}
                    if isinstance(topic_words, dict):
                        for words in topic_words.values():
                            if isinstance(words, list):
                                current_keywords.update(words[:5])  # 상위 5개 키워드
                
                # 키워드 변화율 계산 (자카드 거리)
                if prev_keywords and current_keywords:
                    intersection = len(prev_keywords & current_keywords)
                    union = len(prev_keywords | current_keywords)
                    jaccard_similarity = intersection / union if union > 0 else 0
                    change_rate = 1 - jaccard_similarity
                else:
                    change_rate = 1.0 if current_keywords else 0.0
                
                keyword_changes.append(change_rate)
                prev_keywords = current_keywords
            
            axes[1, 1].plot(months, keyword_changes, marker='s', linewidth=3, 
                           color='#FF9500', markersize=8)
            axes[1, 1].set_title('토픽 의미 변화율 (키워드 기준)', fontsize=16, fontweight='bold')
            axes[1, 1].set_ylabel('변화율 (0=동일, 1=완전변화)', fontsize=14)
            axes[1, 1].set_ylim(0, 1.1)
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].grid(True, alpha=0.3)
            
            # 급격한 의미 변화 구간 하이라이트
            for i, change in enumerate(keyword_changes):
                if change > 0.7:  # 70% 이상 변화
                    axes[1, 1].scatter(months[i], change, color='red', s=150, alpha=0.8, marker='*')
                    axes[1, 1].text(months[i], change + 0.05, '급변', ha='center', va='bottom', 
                                   fontsize=10, fontweight='bold', color='red')
            
            plt.tight_layout()
            
            # 파일 저장
            filename = f"{target_name}_topic_evolution.png"
            filepath = os.path.join(self.output_dir, filename)
            
            # 디렉토리 생성
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            plt.savefig(filepath, dpi=self.config.VISUALIZATION['dpi'], 
                       bbox_inches='tight', facecolor='white')
            plt.close()
            
            self.logger.info(f"✅ 토픽 진화 시각화 완료: {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"❌ 토픽 진화 시각화 실패: {str(e)}")
            raise
    
    def create_time_grouped_wordclouds(self, time_group_results: Dict, target_name: str, 
                                     min_comments_threshold: int = 50) -> List[str]:
        """
        시간 그룹별 WordCloud 생성 (데이터 부족 구간 필터링)
        Args:
            time_group_results: 시간 그룹별 분석 결과
            target_name: 분석 대상 이름
            min_comments_threshold: 최소 댓글 수 임계값
        Returns:
            저장된 파일 경로 리스트
        """
        try:
            self.logger.info(f"☁️ {target_name} 시간 그룹별 워드클라우드 생성 시작")
            
            saved_files = []
            
            for period, result in time_group_results.items():
                comment_count = result.get('total_comments', 0)
                
                # 데이터 부족 구간 필터링
                if comment_count < min_comments_threshold:
                    self.logger.info(f"⚠️ {period}: 댓글 수 부족({comment_count}개), 워드클라우드 생략")
                    continue
                
                # 키워드 추출 (BERTopic 또는 LDA에서)
                keywords = []
                
                # BERTopic 키워드 우선 사용
                if 'bertopic' in result and result['bertopic']:
                    bertopic_result = result['bertopic']
                    topic_words = bertopic_result.get('topic_words', {})
                    
                    for topic_id, words in topic_words.items():
                        if isinstance(words, list) and len(words) > 0:
                            for word, score in words[:10]:  # 각 토픽에서 상위 10개
                                keywords.append((word, int(score * 100)))  # 점수를 정수로 변환
                
                # BERTopic이 없으면 LDA 사용
                elif 'lda' in result and result['lda']:
                    lda_result = result['lda']
                    topic_words = lda_result.get('topic_words', {})
                    
                    for topic_id, words in topic_words.items():
                        if isinstance(words, list) and len(words) > 0:
                            for word, score in words[:10]:  # 각 토픽에서 상위 10개
                                keywords.append((word, int(score * 100)))  # 점수를 정수로 변환
                
                if not keywords:
                    self.logger.warning(f"⚠️ {period}: 키워드가 없어 워드클라우드 생성 불가")
                    continue
                
                # 중복 키워드 제거 및 빈도 합산
                keyword_dict = {}
                for word, freq in keywords:
                    if word in keyword_dict:
                        keyword_dict[word] += freq
                    else:
                        keyword_dict[word] = freq
                
                # 상위 키워드만 선택
                top_keywords = sorted(keyword_dict.items(), key=lambda x: x[1], reverse=True)[:50]
                
                if len(top_keywords) < 5:
                    self.logger.warning(f"⚠️ {period}: 유효한 키워드가 너무 적음({len(top_keywords)}개)")
                    continue
                
                # 워드클라우드 생성
                filepath = self.create_wordcloud(top_keywords, target_name, period)
                if filepath:
                    saved_files.append(filepath)
                    self.logger.info(f"✅ {period} 워드클라우드 생성: {comment_count}개 댓글")
            
            self.logger.info(f"✅ 시간 그룹별 워드클라우드 생성 완료: {len(saved_files)}개")
            return saved_files
            
        except Exception as e:
            self.logger.error(f"❌ 시간 그룹별 워드클라우드 생성 실패: {str(e)}")
            return []
    
    def create_wordcloud(self, keywords: List[Tuple[str, int]], target_name: str, 
                        period: str = None) -> str:
        """
        워드클라우드 생성
        Args:
            keywords: (키워드, 빈도) 튜플 리스트
            target_name: 분석 대상 이름
            month: 월 (선택사항)
        Returns:
            저장된 파일 경로
        """
        try:
            period_str = f"_{period}" if period else ""
            self.logger.info(f"☁️ {target_name}{period_str} 워드클라우드 생성 시작")
            
            if not keywords:
                self.logger.warning("⚠️ 키워드가 없어 워드클라우드를 생성할 수 없습니다.")
                return None
            
            # 키워드 딕셔너리 생성 (불용어 재필터링)
            filtered_keywords = []
            for word, freq in keywords:
                if (word not in self.config.KOREAN_STOPWORDS and 
                    len(word) > 1 and 
                    not word.isdigit() and
                    freq >= 2):  # 최소 빈도 2 이상
                    filtered_keywords.append((word, freq))
            
            if len(filtered_keywords) < 5:
                self.logger.warning("⚠️ 필터링 후 키워드가 너무 적습니다.")
                return None
            
            # 상위 키워드만 사용
            word_freq = dict(filtered_keywords[:self.config.WORDCLOUD['max_words']])
            
            # 워드클라우드 설정
            wordcloud_config = self.config.WORDCLOUD
            
            # 폰트 경로 검증 및 설정
            font_path_to_use = None
            
            # 1. 기본 설정된 폰트 경로 확인
            if hasattr(self, 'korean_font_path') and self.korean_font_path and os.path.exists(self.korean_font_path):
                if self.korean_font_path.endswith('.ttf'):
                    font_path_to_use = self.korean_font_path
                    self.logger.info(f"📝 워드클라우드 폰트 사용: {font_path_to_use}")
                else:
                    self.logger.warning(f"⚠️ 워드클라우드는 .ttf 파일만 지원합니다: {self.korean_font_path}")
            
            # 2. 시스템 폰트 경로에서 한글 폰트 찾기
            if not font_path_to_use:
                import matplotlib.font_manager as fm
                
                # 한글 폰트 이름 목록
                korean_font_names = [
                    'AppleGothic', 'Apple SD Gothic Neo', 'Noto Sans CJK KR',
                    'Malgun Gothic', 'NanumGothic', 'Arial Unicode MS',
                    'Gulim', 'Dotum', 'Batang'
                ]
                
                # 시스템에서 사용 가능한 .ttf 폰트 찾기
                for font_obj in fm.fontManager.ttflist:
                    if (font_obj.name in korean_font_names and 
                        font_obj.fname.endswith('.ttf') and 
                        os.path.exists(font_obj.fname)):
                        font_path_to_use = font_obj.fname
                        self.logger.info(f"📝 시스템 한글 폰트 발견: {font_obj.name} ({font_path_to_use})")
                        break
            
            # 3. 대체 폰트 경로 시도 (macOS 기준)
            if not font_path_to_use:
                alternative_fonts = [
                    '/System/Library/Fonts/Supplemental/AppleGothic.ttf',
                    '/System/Library/Fonts/Supplemental/AppleMyungjo.ttf',
                    '/System/Library/Fonts/Helvetica.ttc',  # 기본 폰트
                    '/System/Library/Fonts/Arial.ttf'
                ]
                
                for alt_font in alternative_fonts:
                    if os.path.exists(alt_font) and alt_font.endswith('.ttf'):
                        font_path_to_use = alt_font
                        self.logger.info(f"📝 대체 폰트 사용: {font_path_to_use}")
                        break
            
            if not font_path_to_use:
                self.logger.warning("⚠️ 한글 폰트를 찾을 수 없어 기본 폰트를 사용합니다.")
            
            # 워드클라우드 생성 - config 설정 사용
            wordcloud_params = {
                'width': wordcloud_config['width'],
                'height': wordcloud_config['height'],
                'max_words': wordcloud_config['max_words'],
                'min_font_size': wordcloud_config['min_font_size'],
                'max_font_size': wordcloud_config['max_font_size'],
                'background_color': wordcloud_config['background_color'],
                'colormap': wordcloud_config['colormap'],
                'prefer_horizontal': wordcloud_config['prefer_horizontal'],
                'collocations': wordcloud_config['collocations'],
                'relative_scaling': wordcloud_config['relative_scaling']
            }
            
            # 폰트 경로가 있으면 추가
            if font_path_to_use:
                wordcloud_params['font_path'] = font_path_to_use
            
            wordcloud = WordCloud(**wordcloud_params).generate_from_frequencies(word_freq)
            
            # 시각화
            plt.figure(figsize=(15, 10))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            
            title = f'{target_name} 주요 키워드'
            if period:
                title += f' ({period})'
            plt.title(title, fontsize=20, fontweight='bold', pad=20)
            
            # 파일 저장 (디렉토리 생성 포함)
            # 파일명에서 특수문자 제거
            safe_period = period.replace('~', '_').replace(' ', '_').replace(':', '_') if period else ""
            filename = f"{target_name}_{safe_period}_wordcloud.png" if safe_period else f"{target_name}_wordcloud.png"
            filepath = os.path.join(self.output_dir, filename)
            
            # 디렉토리가 없으면 생성
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            plt.savefig(filepath, dpi=self.config.VISUALIZATION['dpi'], 
                       bbox_inches='tight', facecolor='white')
            plt.close()
            
            self.logger.info(f"✅ 워드클라우드 생성 완료: {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"❌ 워드클라우드 생성 실패: {str(e)}")
            # 폰트 없이 재시도
            try:
                self.logger.info("🔄 기본 폰트로 워드클라우드 재시도")
                wordcloud = WordCloud(
                    width=wordcloud_config['width'],
                    height=wordcloud_config['height'],
                    max_words=wordcloud_config['max_words'],
                    background_color=wordcloud_config['background_color'],
                    colormap=wordcloud_config['colormap'],
                    relative_scaling=0.5,
                    min_font_size=10
                ).generate_from_frequencies(word_freq)
                
                plt.figure(figsize=(15, 10))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                
                title = f'{target_name} 주요 키워드'
                if period:
                    title += f' ({period})'
                plt.title(title, fontsize=20, fontweight='bold', pad=20)
                
                # 파일명에서 특수문자 제거
                safe_period = period.replace('~', '_').replace(' ', '_').replace(':', '_') if period else ""
                filename = f"{target_name}_{safe_period}_wordcloud.png" if safe_period else f"{target_name}_wordcloud.png"
                filepath = os.path.join(self.output_dir, filename)
                
                # 디렉토리가 없으면 생성
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                
                plt.savefig(filepath, dpi=self.config.VISUALIZATION['dpi'], 
                           bbox_inches='tight', facecolor='white')
                plt.close()
                
                self.logger.info(f"✅ 기본 폰트로 워드클라우드 생성 완료: {filepath}")
                return filepath
                
            except Exception as e2:
                self.logger.error(f"❌ 기본 폰트로도 워드클라우드 생성 실패: {str(e2)}")
                return None
    
    def plot_time_grouped_comparison(self, time_group_results: Dict, target_name: str, time_unit: str) -> str:
        """
        적응적 시간 그룹별 비교 시각화
        Args:
            time_group_results: 시간 그룹별 분석 결과
            target_name: 분석 대상 이름
            time_unit: 시간 단위 (monthly, weekly, hybrid 등)
        Returns:
            저장된 파일 경로
        """
        try:
            self.logger.info(f"📊 {target_name} 시간 그룹별 비교 시각화 생성 시작")
            
            # 폰트 설정 재적용
            self._apply_font_settings()
            
            # 데이터 준비
            periods = []
            positive_ratios = []
            negative_ratios = []
            comment_counts = []
            dominant_emotions = []
            
            for period, result in time_group_results.items():
                periods.append(period)
                
                binary_sentiment = result.get('binary_sentiment', {})
                positive_ratios.append(binary_sentiment.get('positive_ratio', 0) * 100)
                negative_ratios.append(binary_sentiment.get('negative_ratio', 0) * 100)
                comment_counts.append(result.get('total_comments', 0))
                
                emotion_sentiment = result.get('emotion_sentiment', {})
                dominant_emotions.append(emotion_sentiment.get('dominant_emotion', '없음'))
            
            if not periods:
                self.logger.warning("⚠️ 시각화할 데이터가 없습니다.")
                return None
            
            # 그래프 생성
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'{target_name} - 시간 그룹별 감성 분석 비교 ({time_unit})', 
                        fontsize=16, fontweight='bold')
            
            # 1. 감성 비율 트렌드
            x_pos = range(len(periods))
            width = 0.35
            
            axes[0, 0].bar([x - width/2 for x in x_pos], positive_ratios, width, 
                          label='긍정', color='skyblue', alpha=0.8)
            axes[0, 0].bar([x + width/2 for x in x_pos], negative_ratios, width, 
                          label='부정', color='lightcoral', alpha=0.8)
            
            axes[0, 0].set_title('시간 그룹별 감성 비율')
            axes[0, 0].set_xlabel('시간 그룹')
            axes[0, 0].set_ylabel('비율 (%)')
            axes[0, 0].legend()
            axes[0, 0].set_xticks(x_pos)
            axes[0, 0].set_xticklabels(periods, rotation=45, ha='right')
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. 댓글 수 트렌드
            axes[0, 1].plot(x_pos, comment_counts, marker='o', linewidth=2, markersize=6, color='green')
            axes[0, 1].set_title('시간 그룹별 댓글 수')
            axes[0, 1].set_xlabel('시간 그룹')
            axes[0, 1].set_ylabel('댓글 수')
            axes[0, 1].set_xticks(x_pos)
            axes[0, 1].set_xticklabels(periods, rotation=45, ha='right')
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. 주요 감정 분포
            emotion_counts = Counter(dominant_emotions)
            emotions = list(emotion_counts.keys())
            counts = list(emotion_counts.values())
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(emotions)))
            axes[1, 0].pie(counts, labels=emotions, autopct='%1.1f%%', colors=colors, startangle=90)
            axes[1, 0].set_title('주요 감정 분포')
            
            # 4. 감성 점수 히트맵
            if len(periods) > 1:
                heatmap_data = []
                for period in periods:
                    result = time_group_results[period]
                    binary_sentiment = result.get('binary_sentiment', {})
                    emotion_sentiment = result.get('emotion_sentiment', {})
                    
                    row = [
                        binary_sentiment.get('positive_ratio', 0),
                        binary_sentiment.get('negative_ratio', 0),
                        emotion_sentiment.get('avg_confidence', 0)
                    ]
                    heatmap_data.append(row)
                
                heatmap_df = pd.DataFrame(heatmap_data, 
                                        index=periods, 
                                        columns=['긍정비율', '부정비율', '감정신뢰도'])
                
                im = axes[1, 1].imshow(heatmap_df.T, cmap='RdYlBu_r', aspect='auto')
                axes[1, 1].set_title('감성 점수 히트맵')
                axes[1, 1].set_xticks(range(len(periods)))
                axes[1, 1].set_xticklabels(periods, rotation=45, ha='right')
                axes[1, 1].set_yticks(range(len(heatmap_df.columns)))
                axes[1, 1].set_yticklabels(heatmap_df.columns)
                
                # 컬러바 추가
                plt.colorbar(im, ax=axes[1, 1])
            else:
                axes[1, 1].text(0.5, 0.5, '데이터 부족\n(히트맵 생성 불가)', 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('감성 점수 히트맵')
            
            plt.tight_layout()
            
            # 저장
            filename = f'{target_name}_{time_unit}_comparison.png'
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"✅ 시간 그룹별 비교 시각화 저장: {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"❌ 시간 그룹별 비교 시각화 실패: {str(e)}")
            return None
    
    def plot_monthly_comparison(self, monthly_data: Dict, target_name: str) -> str:
        """
        월별 비교 시각화 (개선된 버전)
        Args:
            monthly_data: 월별 데이터
            target_name: 분석 대상 이름
        Returns:
            저장된 파일 경로
        """
        try:
            self.logger.info(f"📊 {target_name} 월별 비교 시각화 시작")
            
            # 데이터 준비
            months = sorted(monthly_data.keys())
            comment_counts = [monthly_data[month]['total_comments'] for month in months]
            
            # 감성 데이터 추출
            positive_ratios = []
            negative_ratios = []
            dominant_emotions = []
            emotion_distributions = {}  # 6감정 분포 저장
            
            for month in months:
                if 'binary_sentiment' in monthly_data[month]:
                    binary_data = monthly_data[month]['binary_sentiment']
                    positive_ratios.append(binary_data.get('positive_ratio', 0))
                    negative_ratios.append(binary_data.get('negative_ratio', 0))
                else:
                    positive_ratios.append(0)
                    negative_ratios.append(0)
                
                if 'emotion_sentiment' in monthly_data[month]:
                    emotion_data = monthly_data[month]['emotion_sentiment']
                    dominant_emotions.append(emotion_data.get('dominant_emotion', '없음'))
                    
                    # 6감정 분포 수집
                    emotion_dist = emotion_data.get('emotion_distribution', {})
                    for emotion, ratio in emotion_dist.items():
                        if emotion not in emotion_distributions:
                            emotion_distributions[emotion] = []
                        emotion_distributions[emotion].append(ratio)
                else:
                    dominant_emotions.append('없음')
                    for emotion in ['분노', '슬픔', '불안', '상처', '당황', '기쁨']:
                        if emotion not in emotion_distributions:
                            emotion_distributions[emotion] = []
                        emotion_distributions[emotion].append(0)
            
            # 시각화 (2x2 레이아웃)
            fig, axes = plt.subplots(2, 2, figsize=(20, 15))
            fig.suptitle(f'{target_name} 월별 종합 분석', fontsize=20, fontweight='bold')
            
            # 1. 월별 댓글 수 + 급증 시점 표시
            bars1 = axes[0, 0].bar(months, comment_counts, color='skyblue', alpha=0.7)
            axes[0, 0].set_title('월별 댓글 수 및 급증 시점', fontsize=16, fontweight='bold')
            axes[0, 0].set_ylabel('댓글 수', fontsize=14)
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # 급증 시점 표시 (평균의 1.5배 이상)
            avg_comments = np.mean(comment_counts)
            for i, (month, count) in enumerate(zip(months, comment_counts)):
                if count > avg_comments * 1.5:
                    axes[0, 0].annotate('급증', xy=(i, count), xytext=(i, count + max(comment_counts)*0.1),
                                       arrowprops=dict(arrowstyle='->', color='red', lw=2),
                                       fontsize=12, ha='center', color='red', fontweight='bold')
            
            # 막대 위에 값 표시
            for bar, count in zip(bars1, comment_counts):
                axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(comment_counts)*0.01,
                               f'{count:,}', ha='center', va='bottom', fontsize=10)
            
            # 2. 감성 비율 스택 바 (기존 유지)
            width = 0.6
            axes[0, 1].bar(months, positive_ratios, width, label='긍정', 
                          color='#2E8B57', alpha=0.8)
            axes[0, 1].bar(months, negative_ratios, width, bottom=positive_ratios, 
                          label='부정', color='#DC143C', alpha=0.8)
            axes[0, 1].set_title('월별 감성 비율', fontsize=16, fontweight='bold')
            axes[0, 1].set_ylabel('비율', fontsize=14)
            axes[0, 1].legend(fontsize=12)
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # 3. 주요 감정 분포 (기존 유지)
            emotion_counts = Counter(dominant_emotions)
            emotions = list(emotion_counts.keys())
            counts = list(emotion_counts.values())
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(emotions)))
            axes[1, 0].pie(counts, labels=emotions, colors=colors, autopct='%1.1f%%', 
                          startangle=90)
            axes[1, 0].set_title('주요 감정 분포', fontsize=16, fontweight='bold')
            
            # 4. 6감정 월별 변화 추이 (새로 추가)
            emotion_colors = {
                '분노': '#FF4444', '슬픔': '#4444FF', '불안': '#FF8800',
                '상처': '#8800FF', '당황': '#00FF88', '기쁨': '#FFFF00'
            }
            
            for emotion, values in emotion_distributions.items():
                if len(values) == len(months) and max(values) > 0.05:  # 5% 이상인 감정만 표시
                    axes[1, 1].plot(months, values, marker='o', linewidth=2, 
                                   label=emotion, color=emotion_colors.get(emotion, '#888888'),
                                   markersize=6, alpha=0.8)
            
            axes[1, 1].set_title('6감정 월별 변화 추이', fontsize=16, fontweight='bold')
            axes[1, 1].set_ylabel('감정 비율', fontsize=14)
            axes[1, 1].legend(fontsize=10, loc='upper right')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_ylim(0, max([max(values) for values in emotion_distributions.values()]) * 1.1)
            
            plt.tight_layout()
            
            # 파일 저장
            filename = f"{target_name}_monthly_comparison.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=self.config.VISUALIZATION['dpi'], 
                       bbox_inches='tight', facecolor='white')
            plt.close()
            
            self.logger.info(f"✅ 월별 비교 시각화 완료: {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"❌ 월별 비교 시각화 실패: {str(e)}")
            raise
    
    def create_network_graph(self, monthly_keywords: Dict[str, List[Tuple[str, int]]], 
                           target_name: str) -> List[str]:
        """
        월별 키워드 네트워크 그래프 생성 (개선된 버전)
        Args:
            monthly_keywords: 월별 키워드 딕셔너리
            target_name: 분석 대상 이름
        Returns:
            저장된 파일 경로 리스트
        """
        try:
            self.logger.info(f"🕸️ {target_name} 월별 네트워크 그래프 생성 시작")
            
            saved_files = []
            
            for month, keywords in monthly_keywords.items():
                if len(keywords) < 8:  # 최소 키워드 수 증가
                    self.logger.warning(f"⚠️ {month}: 키워드가 너무 적어 네트워크 그래프를 생성할 수 없습니다.")
                    continue
                
                # 상위 키워드만 사용 (수 감소)
                top_keywords = keywords[:12]  # 20개에서 12개로 감소
                
                # 네트워크 그래프 생성
                G = nx.Graph()
                
                # 노드 추가 (키워드)
                for word, freq in top_keywords:
                    # 불용어 재확인
                    if word not in self.config.KOREAN_STOPWORDS and len(word) > 1:
                        G.add_node(word, weight=freq)
                
                # 실제 노드 수 확인
                if len(G.nodes()) < 5:
                    self.logger.warning(f"⚠️ {month}: 유효한 노드가 너무 적습니다.")
                    continue
                
                # 엣지 추가 (빈도 기반 연결)
                nodes = list(G.nodes())
                node_weights = {node: G.nodes[node]['weight'] for node in nodes}
                
                # 빈도가 높은 노드들을 중심으로 연결
                sorted_nodes = sorted(nodes, key=lambda x: node_weights[x], reverse=True)
                
                # 중심 노드들 (상위 3개)과 다른 노드들 연결
                center_nodes = sorted_nodes[:3]
                for center in center_nodes:
                    for other in sorted_nodes[3:]:
                        # 빈도 차이가 크지 않으면 연결
                        if node_weights[other] >= node_weights[center] * 0.3:
                            G.add_edge(center, other)
                
                # 비슷한 빈도의 노드들끼리 연결
                for i in range(len(sorted_nodes)):
                    for j in range(i+1, min(i+3, len(sorted_nodes))):
                        if abs(node_weights[sorted_nodes[i]] - node_weights[sorted_nodes[j]]) <= 2:
                            G.add_edge(sorted_nodes[i], sorted_nodes[j])
                
                # 시각화
                plt.figure(figsize=(12, 10))
                
                # 레이아웃 설정 (더 집중된 레이아웃)
                pos = nx.spring_layout(G, k=2, iterations=100, seed=42)
                
                # 노드 크기 (빈도에 비례, 크기 조정)
                node_sizes = [node_weights[node] * 100 for node in G.nodes()]
                
                # 노드 색상 (빈도에 따라)
                node_colors = [node_weights[node] for node in G.nodes()]
                
                # 노드 그리기
                nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                                             node_color=node_colors, 
                                             cmap=plt.cm.viridis, alpha=0.8)
                
                # 엣지 그리기 (더 얇게)
                nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='gray', width=0.5)
                
                # 라벨 그리기 - 한글 폰트 적용
                label_font = {'size': 9, 'weight': 'bold'}
                if hasattr(self, 'korean_font_name') and self.korean_font_name != "DejaVu Sans":
                    label_font['family'] = self.korean_font_name
                
                nx.draw_networkx_labels(G, pos, font_size=label_font['size'], 
                                      font_weight=label_font['weight'],
                                      font_family=label_font.get('family', 'DejaVu Sans'))
                
                # 컬러바 추가
                if nodes:
                    plt.colorbar(nodes, label='키워드 빈도')
                
                plt.title(f'{target_name} 키워드 네트워크 - {month}', 
                         fontsize=14, fontweight='bold')
                plt.axis('off')
                
                # 파일 저장
                filename = f"{target_name}_network_{month}.png"
                filepath = os.path.join(self.output_dir, filename)
                plt.savefig(filepath, dpi=self.config.VISUALIZATION['dpi'], 
                           bbox_inches='tight', facecolor='white')
                plt.close()
                
                saved_files.append(filepath)
                self.logger.info(f"✅ {month} 네트워크 그래프 생성 완료: {filepath}")
            
            self.logger.info(f"✅ 총 {len(saved_files)}개 월별 네트워크 그래프 생성 완료")
            return saved_files
            
        except Exception as e:
            self.logger.error(f"❌ 네트워크 그래프 생성 실패: {str(e)}")
            return []
    
    def create_interactive_dashboard(self, all_results: Dict, target_name: str) -> str:
        """
        인터랙티브 대시보드 생성 (Plotly)
        Args:
            all_results: 모든 분석 결과
            target_name: 분석 대상 이름
        Returns:
            저장된 HTML 파일 경로
        """
        try:
            self.logger.info(f"📱 {target_name} 인터랙티브 대시보드 생성 시작")
            
            # 서브플롯 생성
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('감성 트렌드', '토픽 수 변화', '월별 댓글 수', '감정 분포'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"type": "pie"}]]
            )
            
            # 데이터 준비 (예시)
            months = list(all_results.keys()) if all_results else []
            
            if months:
                # 감성 트렌드
                positive_ratios = []
                negative_ratios = []
                comment_counts = []
                
                for month in months:
                    if 'binary_sentiment' in all_results[month]:
                        binary_data = all_results[month]['binary_sentiment']
                        positive_ratios.append(binary_data.get('positive_ratio', 0))
                        negative_ratios.append(binary_data.get('negative_ratio', 0))
                    else:
                        positive_ratios.append(0)
                        negative_ratios.append(0)
                    
                    comment_counts.append(all_results[month].get('total_comments', 0))
                
                # 감성 트렌드 추가
                fig.add_trace(
                    go.Scatter(x=months, y=positive_ratios, name='긍정', 
                              line=dict(color='green', width=3)),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(x=months, y=negative_ratios, name='부정', 
                              line=dict(color='red', width=3)),
                    row=1, col=1
                )
                
                # 월별 댓글 수
                fig.add_trace(
                    go.Bar(x=months, y=comment_counts, name='댓글 수', 
                          marker_color='skyblue'),
                    row=2, col=1
                )
            
            # 레이아웃 설정
            fig.update_layout(
                title=f'{target_name} 분석 대시보드',
                height=800,
                showlegend=True
            )
            
            # HTML 파일로 저장
            filename = f"{target_name}_dashboard.html"
            filepath = os.path.join(self.output_dir, filename)
            fig.write_html(filepath)
            
            self.logger.info(f"✅ 인터랙티브 대시보드 생성 완료: {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"❌ 인터랙티브 대시보드 생성 실패: {str(e)}")
            raise
    
    def plot_topic_quality_analysis(self, monthly_data: Dict, target_name: str) -> str:
        """
        토픽 품질 분석 시각화
        Args:
            monthly_data: 월별 데이터
            target_name: 분석 대상 이름
        Returns:
            저장된 파일 경로
        """
        try:
            self.logger.info(f"📊 {target_name} 토픽 품질 분석 시각화 시작")
            
            # 데이터 준비
            months = sorted(monthly_data.keys())
            bertopic_counts = []
            lda_counts = []
            lda_coherence_scores = []
            lda_quality_scores = []
            
            for month in months:
                # BERTopic 토픽 수
                if 'bertopic' in monthly_data[month] and monthly_data[month]['bertopic']:
                    bertopic_result = monthly_data[month]['bertopic']
                    bertopic_counts.append(len(bertopic_result.get('topics', [])))
                else:
                    bertopic_counts.append(0)
                
                # LDA 토픽 수 및 품질 지표
                if 'lda' in monthly_data[month] and monthly_data[month]['lda']:
                    lda_result = monthly_data[month]['lda']
                    lda_counts.append(lda_result.get('optimal_topic_count', 0))
                    lda_coherence_scores.append(lda_result.get('coherence_score', 0))
                    lda_quality_scores.append(lda_result.get('topic_quality', 0))
                else:
                    lda_counts.append(0)
                    lda_coherence_scores.append(0)
                    lda_quality_scores.append(0)
            
            # 시각화 (2x2 레이아웃)
            fig, axes = plt.subplots(2, 2, figsize=(20, 15))
            fig.suptitle(f'{target_name} 토픽 분석 품질 평가', fontsize=20, fontweight='bold')
            
            # 1. 월별 토픽 수 비교 (BERTopic vs LDA)
            x = np.arange(len(months))
            width = 0.35
            
            bars1 = axes[0, 0].bar(x - width/2, bertopic_counts, width, label='BERTopic', 
                                  color='skyblue', alpha=0.8)
            bars2 = axes[0, 0].bar(x + width/2, lda_counts, width, label='LDA (최적화)', 
                                  color='lightcoral', alpha=0.8)
            
            axes[0, 0].set_title('월별 토픽 수 비교', fontsize=16, fontweight='bold')
            axes[0, 0].set_ylabel('토픽 수', fontsize=14)
            axes[0, 0].set_xticks(x)
            axes[0, 0].set_xticklabels(months, rotation=45)
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 막대 위에 값 표시
            for bar in bars1:
                height = bar.get_height()
                if height > 0:
                    axes[0, 0].text(bar.get_x() + bar.get_width()/2, height + 0.1,
                                   f'{int(height)}', ha='center', va='bottom', fontsize=9)
            
            for bar in bars2:
                height = bar.get_height()
                if height > 0:
                    axes[0, 0].text(bar.get_x() + bar.get_width()/2, height + 0.1,
                                   f'{int(height)}', ha='center', va='bottom', fontsize=9)
            
            # 2. LDA 일관성 점수 변화
            valid_coherence = [(month, score) for month, score in zip(months, lda_coherence_scores) if score > 0]
            if valid_coherence:
                valid_months, valid_scores = zip(*valid_coherence)
                axes[0, 1].plot(valid_months, valid_scores, marker='o', linewidth=3, 
                               color='green', markersize=8, alpha=0.8)
                axes[0, 1].set_title('LDA 토픽 일관성 점수 변화', fontsize=16, fontweight='bold')
                axes[0, 1].set_ylabel('일관성 점수', fontsize=14)
                axes[0, 1].tick_params(axis='x', rotation=45)
                axes[0, 1].grid(True, alpha=0.3)
                axes[0, 1].set_ylim(0, max(valid_scores) * 1.1)
            else:
                axes[0, 1].text(0.5, 0.5, '일관성 점수 데이터 없음', 
                               ha='center', va='center', transform=axes[0, 1].transAxes,
                               fontsize=14)
                axes[0, 1].set_title('LDA 토픽 일관성 점수 변화', fontsize=16, fontweight='bold')
            
            # 3. LDA 토픽 품질 점수 변화
            valid_quality = [(month, score) for month, score in zip(months, lda_quality_scores) if score > 0]
            if valid_quality:
                valid_months, valid_scores = zip(*valid_quality)
                axes[1, 0].plot(valid_months, valid_scores, marker='s', linewidth=3, 
                               color='purple', markersize=8, alpha=0.8)
                axes[1, 0].set_title('LDA 토픽 품질 점수 변화', fontsize=16, fontweight='bold')
                axes[1, 0].set_ylabel('품질 점수', fontsize=14)
                axes[1, 0].tick_params(axis='x', rotation=45)
                axes[1, 0].grid(True, alpha=0.3)
                axes[1, 0].set_ylim(0, 1)
            else:
                axes[1, 0].text(0.5, 0.5, '품질 점수 데이터 없음', 
                               ha='center', va='center', transform=axes[1, 0].transAxes,
                               fontsize=14)
                axes[1, 0].set_title('LDA 토픽 품질 점수 변화', fontsize=16, fontweight='bold')
            
            # 4. 토픽 분석 효율성 (댓글 수 대비 토픽 수)
            comment_counts = [monthly_data[month]['total_comments'] for month in months]
            bertopic_efficiency = [topics/max(comments, 1) * 1000 for topics, comments in zip(bertopic_counts, comment_counts)]
            lda_efficiency = [topics/max(comments, 1) * 1000 for topics, comments in zip(lda_counts, comment_counts)]
            
            axes[1, 1].plot(months, bertopic_efficiency, marker='o', linewidth=2, 
                           label='BERTopic', color='skyblue', markersize=6)
            axes[1, 1].plot(months, lda_efficiency, marker='s', linewidth=2, 
                           label='LDA', color='lightcoral', markersize=6)
            axes[1, 1].set_title('토픽 분석 효율성 (토픽수/댓글수×1000)', fontsize=16, fontweight='bold')
            axes[1, 1].set_ylabel('효율성 지수', fontsize=14)
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 파일 저장
            filename = f"{target_name}_topic_quality_analysis.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=self.config.VISUALIZATION['dpi'], 
                       bbox_inches='tight', facecolor='white')
            plt.close()
            
            self.logger.info(f"✅ 토픽 품질 분석 시각화 완료: {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"❌ 토픽 품질 분석 시각화 실패: {str(e)}")
            raise
    
    def plot_emotion_intensity_analysis(self, monthly_data: Dict, target_name: str) -> str:
        """
        감정 강도 분석 시각화
        Args:
            monthly_data: 월별 데이터
            target_name: 분석 대상 이름
        Returns:
            저장된 파일 경로
        """
        try:
            self.logger.info(f"📊 {target_name} 감정 강도 분석 시각화 시작")
            
            # 데이터 준비
            months = sorted(monthly_data.keys())
            emotion_intensity_data = {}
            emotion_trigger_data = {}
            
            # 6감정별 데이터 수집
            emotions = ['분노', '슬픔', '불안', '상처', '당황', '기쁨']
            for emotion in emotions:
                emotion_intensity_data[emotion] = []
                emotion_trigger_data[emotion] = []
            
            for month in months:
                if 'emotion_sentiment' in monthly_data[month]:
                    emotion_data = monthly_data[month]['emotion_sentiment']
                    
                    # 감정 강도 데이터
                    intensity_data = emotion_data.get('emotion_intensity', {})
                    for emotion in emotions:
                        if emotion in intensity_data:
                            avg_intensity = intensity_data[emotion].get('average_intensity', 0)
                            emotion_intensity_data[emotion].append(avg_intensity)
                        else:
                            emotion_intensity_data[emotion].append(0)
                    
                    # 감정 트리거 데이터 (상위 키워드 수)
                    trigger_data = emotion_data.get('emotion_triggers', {})
                    for emotion in emotions:
                        if emotion in trigger_data:
                            trigger_count = len(trigger_data[emotion])
                            emotion_trigger_data[emotion].append(trigger_count)
                        else:
                            emotion_trigger_data[emotion].append(0)
                else:
                    for emotion in emotions:
                        emotion_intensity_data[emotion].append(0)
                        emotion_trigger_data[emotion].append(0)
            
            # 시각화 (2x2 레이아웃)
            fig, axes = plt.subplots(2, 2, figsize=(20, 15))
            fig.suptitle(f'{target_name} 감정 강도 및 트리거 분석', fontsize=20, fontweight='bold')
            
            # 감정별 색상
            emotion_colors = {
                '분노': '#FF4444', '슬픔': '#4444FF', '불안': '#FF8800',
                '상처': '#8800FF', '당황': '#00FF88', '기쁨': '#FFFF00'
            }
            
            # 1. 감정 강도 변화 (부정 감정)
            negative_emotions = ['분노', '슬픔', '불안', '상처', '당황']
            for emotion in negative_emotions:
                if max(emotion_intensity_data[emotion]) > 0:
                    axes[0, 0].plot(months, emotion_intensity_data[emotion], 
                                   marker='o', linewidth=2, label=emotion,
                                   color=emotion_colors[emotion], markersize=6, alpha=0.8)
            
            axes[0, 0].set_title('부정 감정 강도 변화', fontsize=16, fontweight='bold')
            axes[0, 0].set_ylabel('평균 강도', fontsize=14)
            axes[0, 0].tick_params(axis='x', rotation=45)
            axes[0, 0].legend(fontsize=10)
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. 기쁨 감정 강도 변화
            if max(emotion_intensity_data['기쁨']) > 0:
                axes[0, 1].plot(months, emotion_intensity_data['기쁨'], 
                               marker='o', linewidth=3, color=emotion_colors['기쁨'],
                               markersize=8, alpha=0.8)
                axes[0, 1].fill_between(months, emotion_intensity_data['기쁨'], 
                                       alpha=0.3, color=emotion_colors['기쁨'])
            
            axes[0, 1].set_title('기쁨 감정 강도 변화', fontsize=16, fontweight='bold')
            axes[0, 1].set_ylabel('평균 강도', fontsize=14)
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. 감정별 트리거 키워드 수
            emotion_names = list(emotion_trigger_data.keys())
            avg_triggers = [np.mean(emotion_trigger_data[emotion]) for emotion in emotion_names]
            
            bars = axes[1, 0].bar(emotion_names, avg_triggers, 
                                 color=[emotion_colors[emotion] for emotion in emotion_names],
                                 alpha=0.8)
            axes[1, 0].set_title('감정별 평균 트리거 키워드 수', fontsize=16, fontweight='bold')
            axes[1, 0].set_ylabel('평균 키워드 수', fontsize=14)
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # 막대 위에 값 표시
            for bar, value in zip(bars, avg_triggers):
                if value > 0:
                    axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                                   f'{value:.1f}', ha='center', va='bottom', fontsize=10)
            
            # 4. 감정 강도 히트맵
            intensity_matrix = []
            for emotion in emotions:
                intensity_matrix.append(emotion_intensity_data[emotion])
            
            intensity_matrix = np.array(intensity_matrix)
            
            if intensity_matrix.max() > 0:
                im = axes[1, 1].imshow(intensity_matrix, cmap='YlOrRd', aspect='auto')
                axes[1, 1].set_title('감정 강도 히트맵 (월별)', fontsize=16, fontweight='bold')
                axes[1, 1].set_ylabel('감정', fontsize=14)
                axes[1, 1].set_xlabel('월', fontsize=14)
                axes[1, 1].set_yticks(range(len(emotions)))
                axes[1, 1].set_yticklabels(emotions)
                axes[1, 1].set_xticks(range(0, len(months), max(1, len(months)//6)))
                axes[1, 1].set_xticklabels([months[i] for i in range(0, len(months), max(1, len(months)//6))], 
                                          rotation=45)
                
                # 컬러바 추가
                plt.colorbar(im, ax=axes[1, 1], label='강도')
            else:
                axes[1, 1].text(0.5, 0.5, '감정 강도 데이터 없음', 
                               ha='center', va='center', transform=axes[1, 1].transAxes,
                               fontsize=14)
                axes[1, 1].set_title('감정 강도 히트맵 (월별)', fontsize=16, fontweight='bold')
            
            plt.tight_layout()
            
            # 파일 저장
            filename = f"{target_name}_emotion_intensity_analysis.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=self.config.VISUALIZATION['dpi'], 
                       bbox_inches='tight', facecolor='white')
            plt.close()
            
            self.logger.info(f"✅ 감정 강도 분석 시각화 완료: {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"❌ 감정 강도 분석 시각화 실패: {str(e)}")
            raise 
    
    def plot_frame_velocity_analysis(self, comprehensive_results: Dict, target_name: str) -> str:
        """
        프레임 변화 속도 분석 시각화 (연구 목적: 뉴스 vs 대중 반응 차이)
        Args:
            comprehensive_results: 종합 분석 결과
            target_name: 분석 대상 이름
        Returns:
            저장된 파일 경로
        """
        try:
            self.logger.info(f"🚀 {target_name} 프레임 변화 속도 분석 시각화 시작")
            
            # 폰트 설정 재적용
            self._apply_font_settings()
            
            fig, axes = plt.subplots(2, 2, figsize=(20, 15))
            fig.suptitle(f'{target_name} 프레임 변화 속도 분석: 뉴스 vs 대중 담론', fontsize=20, fontweight='bold')
            
            periods = sorted(comprehensive_results.keys())
            
            # 1. 토픽 키워드 변화율 (의미적 프레임 변화)
            keyword_change_rates = []
            sentiment_change_rates = []
            comment_velocity = []
            topic_diversity_changes = []
            
            prev_keywords = set()
            prev_sentiment = None
            prev_comment_count = 0
            prev_topic_diversity = 0
            
            for i, period in enumerate(periods):
                result = comprehensive_results[period]
                
                # 현재 기간의 키워드 추출
                current_keywords = set()
                if 'bertopic' in result and result['bertopic'].get('topic_words'):
                    for topic_words in result['bertopic']['topic_words'].values():
                        if isinstance(topic_words, list):
                            current_keywords.update([word for word, _ in topic_words[:5]])
                
                # 키워드 변화율 계산 (자카드 거리)
                if prev_keywords and current_keywords:
                    intersection = len(prev_keywords & current_keywords)
                    union = len(prev_keywords | current_keywords)
                    jaccard_similarity = intersection / union if union > 0 else 0
                    change_rate = 1 - jaccard_similarity
                else:
                    change_rate = 1.0 if current_keywords else 0.0
                
                keyword_change_rates.append(change_rate)
                
                # 감정 변화율 계산
                current_sentiment = result.get('binary_sentiment', {}).get('positive_ratio', 0.5)
                if prev_sentiment is not None:
                    sentiment_change = abs(current_sentiment - prev_sentiment)
                else:
                    sentiment_change = 0
                sentiment_change_rates.append(sentiment_change)
                
                # 댓글 수 변화율 (관심도 급변)
                current_comment_count = result.get('total_comments', 0)
                if prev_comment_count > 0:
                    comment_change = abs(current_comment_count - prev_comment_count) / prev_comment_count
                else:
                    comment_change = 0
                comment_velocity.append(comment_change)
                
                # 토픽 다양성 변화
                current_topic_count = 0
                if 'bertopic' in result and result['bertopic'].get('topic_labels'):
                    current_topic_count = len(result['bertopic']['topic_labels'])
                
                if prev_topic_diversity > 0:
                    diversity_change = abs(current_topic_count - prev_topic_diversity) / max(prev_topic_diversity, 1)
                else:
                    diversity_change = 0
                topic_diversity_changes.append(diversity_change)
                
                # 이전 값 업데이트
                prev_keywords = current_keywords
                prev_sentiment = current_sentiment
                prev_comment_count = current_comment_count
                prev_topic_diversity = current_topic_count
            
            # 1. 프레임 변화 속도 (키워드 기준)
            axes[0, 0].plot(periods, keyword_change_rates, marker='o', linewidth=3, 
                           color='#FF6B6B', markersize=8, alpha=0.8)
            axes[0, 0].set_title('의미적 프레임 변화 속도\n(키워드 변화율)', fontsize=16, fontweight='bold')
            axes[0, 0].set_ylabel('변화율 (0=동일, 1=완전변화)', fontsize=14)
            axes[0, 0].tick_params(axis='x', rotation=45)
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].set_ylim(0, 1.1)
            
            # 급변점 표시
            threshold = np.mean(keyword_change_rates) + np.std(keyword_change_rates)
            for i, rate in enumerate(keyword_change_rates):
                if rate > threshold:
                    axes[0, 0].scatter(periods[i], rate, color='red', s=150, alpha=0.8, marker='*')
                    axes[0, 0].text(periods[i], rate + 0.05, '급변', ha='center', va='bottom', 
                                   fontsize=10, fontweight='bold', color='red')
            
            # 2. 감정 변화 속도 vs 관심도 변화
            ax2 = axes[0, 1].twinx()
            
            line1 = axes[0, 1].plot(periods, sentiment_change_rates, marker='s', linewidth=3, 
                                   color='#4ECDC4', markersize=8, alpha=0.8, label='감정 변화율')
            line2 = ax2.plot(periods, comment_velocity, marker='^', linewidth=3, 
                            color='#45B7D1', markersize=8, alpha=0.8, label='관심도 변화율')
            
            axes[0, 1].set_title('감정 vs 관심도 변화 속도', fontsize=16, fontweight='bold')
            axes[0, 1].set_ylabel('감정 변화율', fontsize=14, color='#4ECDC4')
            ax2.set_ylabel('관심도 변화율', fontsize=14, color='#45B7D1')
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].tick_params(axis='y', labelcolor='#4ECDC4')
            ax2.tick_params(axis='y', labelcolor='#45B7D1')
            axes[0, 1].grid(True, alpha=0.3)
            
            # 범례 통합
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            axes[0, 1].legend(lines, labels, loc='upper left')
            
            # 3. 프레임 변화 패턴 분류
            # 변화 패턴을 4가지로 분류: 안정, 점진적 변화, 급변, 혼란
            pattern_labels = []
            pattern_colors = []
            
            for i in range(len(periods)):
                keyword_rate = keyword_change_rates[i] if i < len(keyword_change_rates) else 0
                sentiment_rate = sentiment_change_rates[i] if i < len(sentiment_change_rates) else 0
                
                if keyword_rate < 0.3 and sentiment_rate < 0.1:
                    pattern = '안정'
                    color = '#2ECC71'
                elif keyword_rate < 0.6 and sentiment_rate < 0.2:
                    pattern = '점진적 변화'
                    color = '#F39C12'
                elif keyword_rate >= 0.6 or sentiment_rate >= 0.2:
                    pattern = '급변'
                    color = '#E74C3C'
                else:
                    pattern = '혼란'
                    color = '#9B59B6'
                
                pattern_labels.append(pattern)
                pattern_colors.append(color)
            
            # 패턴별 색상으로 막대 그래프
            bars = axes[1, 0].bar(periods, [1] * len(periods), color=pattern_colors, alpha=0.8)
            axes[1, 0].set_title('프레임 변화 패턴 분류', fontsize=16, fontweight='bold')
            axes[1, 0].set_ylabel('패턴 강도', fontsize=14)
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].set_ylim(0, 1.2)
            
            # 패턴 라벨 표시
            for bar, pattern in zip(bars, pattern_labels):
                axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                               pattern, ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            # 범례 추가
            unique_patterns = list(set(pattern_labels))
            unique_colors = [pattern_colors[pattern_labels.index(p)] for p in unique_patterns]
            legend_elements = [plt.Rectangle((0,0),1,1, color=color, alpha=0.8) 
                             for color in unique_colors]
            axes[1, 0].legend(legend_elements, unique_patterns, loc='upper right')
            
            # 4. 변화 속도 종합 지수
            # 키워드, 감정, 관심도 변화를 종합한 지수
            composite_velocity = []
            for i in range(len(periods)):
                keyword_rate = keyword_change_rates[i] if i < len(keyword_change_rates) else 0
                sentiment_rate = sentiment_change_rates[i] if i < len(sentiment_change_rates) else 0
                comment_rate = comment_velocity[i] if i < len(comment_velocity) else 0
                
                # 가중 평균 (키워드 변화에 더 높은 가중치)
                composite = (keyword_rate * 0.5 + sentiment_rate * 0.3 + comment_rate * 0.2)
                composite_velocity.append(composite)
            
            axes[1, 1].plot(periods, composite_velocity, marker='o', linewidth=4, 
                           color='#8E44AD', markersize=10, alpha=0.9)
            axes[1, 1].fill_between(periods, composite_velocity, alpha=0.3, color='#8E44AD')
            axes[1, 1].set_title('종합 프레임 변화 지수\n(뉴스 분석으로는 포착 불가능)', fontsize=16, fontweight='bold')
            axes[1, 1].set_ylabel('종합 변화 지수', fontsize=14)
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].grid(True, alpha=0.3)
            
            # 평균선 표시
            avg_velocity = np.mean(composite_velocity)
            axes[1, 1].axhline(y=avg_velocity, color='red', linestyle='--', alpha=0.7, 
                              label=f'평균: {avg_velocity:.3f}')
            axes[1, 1].legend()
            
            # 최고점 표시
            max_idx = np.argmax(composite_velocity)
            max_value = composite_velocity[max_idx]
            axes[1, 1].annotate(f'최대 변화\n{periods[max_idx]}\n({max_value:.3f})', 
                               xy=(periods[max_idx], max_value),
                               xytext=(10, 20), textcoords='offset points',
                               bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8),
                               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
            
            plt.tight_layout()
            
            # 파일 저장
            filename = f"{target_name}_frame_velocity_analysis.png"
            filepath = os.path.join(self.output_dir, filename)
            
            # 디렉토리 생성
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            plt.savefig(filepath, dpi=self.config.VISUALIZATION['dpi'], 
                       bbox_inches='tight', facecolor='white')
            plt.close()
            
            self.logger.info(f"✅ 프레임 변화 속도 분석 시각화 완료: {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"❌ 프레임 변화 속도 분석 시각화 실패: {str(e)}")
            raise
    
    def plot_hidden_discourse_analysis(self, comprehensive_results: Dict, target_name: str) -> str:
        """
        숨겨진 하위 담론 발굴 분석 시각화 (연구 목적: 주류 언론에서 다루지 않는 대중 관심사)
        Args:
            comprehensive_results: 종합 분석 결과
            target_name: 분석 대상 이름
        Returns:
            저장된 파일 경로
        """
        try:
            self.logger.info(f"🔍 {target_name} 숨겨진 하위 담론 발굴 분석 시각화 시작")
            
            # 폰트 설정 재적용
            self._apply_font_settings()
            
            fig, axes = plt.subplots(2, 2, figsize=(20, 15))
            fig.suptitle(f'{target_name} 숨겨진 하위 담론 발굴: 언론이 놓친 대중의 목소리', fontsize=20, fontweight='bold')
            
            periods = sorted(comprehensive_results.keys())
            
            # 1. 소수 의견이지만 강한 감정을 동반하는 토픽 발굴
            minority_topics = {}  # {topic: {periods: [], emotion_intensity: [], comment_ratio: []}}
            emotion_intensity_by_topic = {}
            topic_persistence = {}
            
            for period in periods:
                result = comprehensive_results[period]
                total_comments = result.get('total_comments', 0)
                
                # BERTopic에서 소수 토픽 추출
                if 'bertopic' in result and result['bertopic'].get('topic_words'):
                    topic_words = result['bertopic']['topic_words']
                    topic_sizes = result['bertopic'].get('topic_sizes', {})
                    
                    for topic_id, words in topic_words.items():
                        if isinstance(words, list) and len(words) > 0:
                            # 토픽 대표 키워드 (첫 번째 키워드)
                            topic_key = words[0][0] if isinstance(words[0], tuple) else str(words[0])
                            
                            # 토픽 크기 (전체 댓글 대비 비율)
                            topic_size = topic_sizes.get(topic_id, 0)
                            topic_ratio = topic_size / max(total_comments, 1)
                            
                            # 소수 토픽 기준: 전체의 5% 미만이지만 존재하는 토픽
                            if 0.01 < topic_ratio < 0.05:
                                if topic_key not in minority_topics:
                                    minority_topics[topic_key] = {
                                        'periods': [],
                                        'ratios': [],
                                        'emotion_scores': [],
                                        'keywords': []
                                    }
                                
                                minority_topics[topic_key]['periods'].append(period)
                                minority_topics[topic_key]['ratios'].append(topic_ratio)
                                
                                # 감정 강도 계산 (임의로 설정, 실제로는 해당 토픽 댓글의 감정 분석 필요)
                                emotion_score = topic_ratio * 10  # 비율에 비례한 감정 강도
                                minority_topics[topic_key]['emotion_scores'].append(emotion_score)
                                
                                # 키워드 저장
                                topic_keywords = [w[0] if isinstance(w, tuple) else str(w) for w in words[:3]]
                                minority_topics[topic_key]['keywords'] = topic_keywords
            
            # 소수 의견 토픽들의 감정 강도 vs 지속성
            if minority_topics:
                topic_names = list(minority_topics.keys())[:8]  # 상위 8개만 표시
                avg_emotions = []
                persistence_scores = []
                avg_ratios = []
                
                for topic in topic_names:
                    data = minority_topics[topic]
                    avg_emotions.append(np.mean(data['emotion_scores']))
                    persistence_scores.append(len(data['periods']))  # 등장 기간 수
                    avg_ratios.append(np.mean(data['ratios']) * 100)  # 백분율로 변환
                
                # 버블 차트
                scatter = axes[0, 0].scatter(persistence_scores, avg_emotions, s=np.array(avg_ratios)*200, 
                                           alpha=0.6, c=range(len(topic_names)), cmap='viridis')
                
                axes[0, 0].set_title('소수 의견 토픽의 지속성 vs 감정 강도', fontsize=16, fontweight='bold')
                axes[0, 0].set_xlabel('지속성 (등장 기간 수)', fontsize=14)
                axes[0, 0].set_ylabel('평균 감정 강도', fontsize=14)
                axes[0, 0].grid(True, alpha=0.3)
                
                # 토픽 라벨 표시
                for i, topic in enumerate(topic_names):
                    axes[0, 0].annotate(topic[:8] + '...', 
                                       (persistence_scores[i], avg_emotions[i]),
                                       xytext=(5, 5), textcoords='offset points',
                                       fontsize=9, alpha=0.8)
                
                # 컬러바
                plt.colorbar(scatter, ax=axes[0, 0], label='토픽 인덱스')
            else:
                axes[0, 0].text(0.5, 0.5, '소수 의견 토픽 없음', ha='center', va='center', 
                               transform=axes[0, 0].transAxes, fontsize=14)
                axes[0, 0].set_title('소수 의견 토픽의 지속성 vs 감정 강도', fontsize=16, fontweight='bold')
            
            # 2. 시간대별 숨겨진 관심사 등장 패턴
            hidden_interests = {}
            for period in periods:
                result = comprehensive_results[period]
                
                # 각 기간별로 특이한 키워드 추출 (다른 기간에는 없는 키워드)
                current_keywords = set()
                if 'bertopic' in result and result['bertopic'].get('topic_words'):
                    for topic_words in result['bertopic']['topic_words'].values():
                        if isinstance(topic_words, list):
                            current_keywords.update([w[0] if isinstance(w, tuple) else str(w) 
                                                   for w in topic_words[:5]])
                
                # 다른 기간들의 키워드 수집
                other_keywords = set()
                for other_period in periods:
                    if other_period != period:
                        other_result = comprehensive_results[other_period]
                        if 'bertopic' in other_result and other_result['bertopic'].get('topic_words'):
                            for topic_words in other_result['bertopic']['topic_words'].values():
                                if isinstance(topic_words, list):
                                    other_keywords.update([w[0] if isinstance(w, tuple) else str(w) 
                                                         for w in topic_words[:5]])
                
                # 현재 기간에만 등장하는 키워드 (숨겨진 관심사)
                unique_keywords = current_keywords - other_keywords
                hidden_interests[period] = len(unique_keywords)
            
            # 숨겨진 관심사 등장 패턴
            periods_list = list(hidden_interests.keys())
            hidden_counts = list(hidden_interests.values())
            
            bars = axes[0, 1].bar(periods_list, hidden_counts, color='#E67E22', alpha=0.8)
            axes[0, 1].set_title('기간별 숨겨진 관심사 등장 수\n(해당 기간에만 나타나는 토픽)', fontsize=16, fontweight='bold')
            axes[0, 1].set_ylabel('고유 키워드 수', fontsize=14)
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].grid(True, alpha=0.3)
            
            # 값 표시
            for bar, count in zip(bars, hidden_counts):
                if count > 0:
                    axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                                   str(count), ha='center', va='bottom', fontsize=10)
            
            # 3. 감정 극성별 하위 담론 분포
            emotion_discourse_map = {
                '분노': [], '슬픔': [], '불안': [], '상처': [], '당황': [], '기쁨': []
            }
            
            for period in periods:
                result = comprehensive_results[period]
                
                # 감정별 토픽 분포 (가상 데이터 - 실제로는 감정별 토픽 분석 필요)
                if 'emotion_sentiment' in result:
                    emotion_dist = result['emotion_sentiment'].get('emotion_distribution', {})
                    
                    for emotion, ratio in emotion_dist.items():
                        if emotion in emotion_discourse_map:
                            emotion_discourse_map[emotion].append(ratio)
                        else:
                            emotion_discourse_map[emotion] = [ratio]
            
            # 감정별 평균 비율 계산
            emotion_names = []
            emotion_ratios = []
            emotion_colors = ['#FF4444', '#4444FF', '#FF8800', '#8800FF', '#00FF88', '#FFFF00']
            
            for emotion, ratios in emotion_discourse_map.items():
                if ratios:
                    emotion_names.append(emotion)
                    emotion_ratios.append(np.mean(ratios))
            
            if emotion_names:
                # 도넛 차트
                wedges, texts, autotexts = axes[1, 0].pie(emotion_ratios, labels=emotion_names, 
                                                         colors=emotion_colors[:len(emotion_names)],
                                                         autopct='%1.1f%%', startangle=90,
                                                         wedgeprops=dict(width=0.5))
                axes[1, 0].set_title('감정별 하위 담론 분포\n(언론에서 간과된 감정적 반응)', fontsize=16, fontweight='bold')
            else:
                axes[1, 0].text(0.5, 0.5, '감정 데이터 없음', ha='center', va='center', 
                               transform=axes[1, 0].transAxes, fontsize=14)
                axes[1, 0].set_title('감정별 하위 담론 분포', fontsize=16, fontweight='bold')
            
            # 4. 담론 다양성 지수 (엔트로피 기반)
            discourse_diversity = []
            mainstream_concentration = []
            
            for period in periods:
                result = comprehensive_results[period]
                
                # 토픽 분포의 엔트로피 계산 (다양성 지수)
                if 'bertopic' in result and result['bertopic'].get('topic_sizes'):
                    topic_sizes = list(result['bertopic']['topic_sizes'].values())
                    if topic_sizes:
                        # 정규화
                        total_size = sum(topic_sizes)
                        if total_size > 0:
                            probabilities = [size / total_size for size in topic_sizes]
                            # 엔트로피 계산
                            entropy = -sum(p * np.log(p + 1e-10) for p in probabilities if p > 0)
                            discourse_diversity.append(entropy)
                            
                            # 주류 담론 집중도 (상위 3개 토픽의 비율)
                            sorted_probs = sorted(probabilities, reverse=True)
                            mainstream_ratio = sum(sorted_probs[:3])
                            mainstream_concentration.append(mainstream_ratio)
                        else:
                            discourse_diversity.append(0)
                            mainstream_concentration.append(0)
                    else:
                        discourse_diversity.append(0)
                        mainstream_concentration.append(0)
                else:
                    discourse_diversity.append(0)
                    mainstream_concentration.append(0)
            
            # 이중 y축 그래프
            ax2 = axes[1, 1].twinx()
            
            line1 = axes[1, 1].plot(periods, discourse_diversity, marker='o', linewidth=3, 
                                   color='#27AE60', markersize=8, alpha=0.8, label='담론 다양성')
            line2 = ax2.plot(periods, mainstream_concentration, marker='s', linewidth=3, 
                            color='#E74C3C', markersize=8, alpha=0.8, label='주류 집중도')
            
            axes[1, 1].set_title('담론 다양성 vs 주류 집중도\n(언론 프레임의 한계 지점)', fontsize=16, fontweight='bold')
            axes[1, 1].set_ylabel('다양성 지수 (엔트로피)', fontsize=14, color='#27AE60')
            ax2.set_ylabel('주류 집중도', fontsize=14, color='#E74C3C')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].tick_params(axis='y', labelcolor='#27AE60')
            ax2.tick_params(axis='y', labelcolor='#E74C3C')
            axes[1, 1].grid(True, alpha=0.3)
            
            # 범례 통합
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            axes[1, 1].legend(lines, labels, loc='upper left')
            
            # 다양성이 높은 구간 하이라이트
            if discourse_diversity:
                high_diversity_threshold = np.mean(discourse_diversity) + np.std(discourse_diversity)
                for i, diversity in enumerate(discourse_diversity):
                    if diversity > high_diversity_threshold:
                        axes[1, 1].axvspan(i-0.4, i+0.4, alpha=0.2, color='green', 
                                          label='고다양성' if i == 0 else "")
            
            plt.tight_layout()
            
            # 파일 저장
            filename = f"{target_name}_hidden_discourse_analysis.png"
            filepath = os.path.join(self.output_dir, filename)
            
            # 디렉토리 생성
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            plt.savefig(filepath, dpi=self.config.VISUALIZATION['dpi'], 
                       bbox_inches='tight', facecolor='white')
            plt.close()
            
            self.logger.info(f"✅ 숨겨진 하위 담론 발굴 분석 시각화 완료: {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"❌ 숨겨진 하위 담론 발굴 분석 시각화 실패: {str(e)}")
            raise 