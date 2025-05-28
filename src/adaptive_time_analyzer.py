"""
Adaptive Time Analyzer Module
적응적 시간 분석 모듈 - 데이터 분포에 따라 월별/주간 분석 자동 선택
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import os

class AdaptiveTimeAnalyzer:
    """적응적 시간 분석 클래스"""
    
    def __init__(self, config):
        """
        초기화
        Args:
            config: AnalysisConfig 객체
        """
        self.config = config
        self.logger = self._setup_logger()
        
        # config.py의 적응적 시간 분할 설정 사용
        adaptive_config = config.ADAPTIVE_TIME_ANALYSIS
        self.adaptive_enabled = adaptive_config['enabled']
        self.high_ratio_threshold = adaptive_config['high_ratio_threshold']
        self.subdivision_levels = adaptive_config['subdivision_levels']
        self.min_comments_per_period = adaptive_config['min_comments_per_period']
        
        # 🔧 최소 시간 단위 설정 (하이퍼파라미터)
        self.min_time_unit = adaptive_config['min_time_unit']
        time_unit_config = adaptive_config['time_unit_configs'][self.min_time_unit]
        self.max_subdivision_depth = time_unit_config['max_subdivision_depth']
        self.subdivision_chain = time_unit_config['subdivision_chain']
        self.initial_level = time_unit_config['initial_level']
        
        self.logger.info(f"🔧 시간 간격 최소 단위: {self.min_time_unit}")
        self.logger.info(f"🔧 세분화 체인: {' → '.join(self.subdivision_chain)}")
        self.logger.info(f"🔧 최대 세분화 깊이: {self.max_subdivision_depth}")
        
        # 기존 분석 기준 설정 (호환성 유지)
        self.min_periods = 5  # 최소 분석 기간 수
        self.outlier_threshold = 0.1  # 아웃라이어 임계값 (10분위수)
        self.weekly_merge_threshold = 50  # 주간 분석에서 병합할 최소 댓글 수
        self.absolute_min_threshold = 30  # 절대 최소 댓글 수 (이 미만은 아예 제외)
        
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
    
    def _setup_korean_font_for_plot(self):
        """플롯용 한글 폰트 설정 강화"""
        try:
            import matplotlib.font_manager as fm
            from matplotlib import rcParams
            import platform
            
            # 1. 시스템별 한국어 폰트 경로 확장
            font_paths = []
            
            system = platform.system()
            if system == "Darwin":  # macOS
                font_paths.extend([
                    '/System/Library/Fonts/AppleGothic.ttf',
                    '/System/Library/Fonts/AppleSDGothicNeo.ttc',
                    '/Library/Fonts/NanumGothic.ttf',
                    '/Library/Fonts/NanumBarunGothic.ttf',
                    os.path.join(self.config.BASE_DIR, 'fonts', 'AppleGothic.ttf')
                ])
            elif system == "Linux":  # Ubuntu/Linux
                font_paths.extend([
                    '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
                    '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf',
                    '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
                    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                    os.path.join(self.config.BASE_DIR, 'fonts', 'NanumGothic.ttf')
                ])
            elif system == "Windows":  # Windows
                font_paths.extend([
                    'C:/Windows/Fonts/malgun.ttf',
                    'C:/Windows/Fonts/gulim.ttc',
                    'C:/Windows/Fonts/batang.ttc',
                    os.path.join(self.config.BASE_DIR, 'fonts', 'malgun.ttf')
                ])
            
            # 2. 사용 가능한 폰트 파일 찾기
            available_font_path = None
            for path in font_paths:
                if os.path.exists(path):
                    available_font_path = path
                    break
            
            # 3. 폰트 설정
            if available_font_path:
                try:
                    # matplotlib 캐시 정리
                    import shutil
                    cache_dir = os.path.expanduser('~/.cache/matplotlib')
                    if os.path.exists(cache_dir):
                        try:
                            shutil.rmtree(cache_dir)
                        except:
                            pass
                    
                    # 폰트 등록
                    try:
                        fm.fontManager.addfont(available_font_path)
                    except:
                        pass
                    
                    # 폰트 속성 가져오기
                    font_prop = fm.FontProperties(fname=available_font_path)
                    font_name = font_prop.get_name()
                    
                    # matplotlib 설정 초기화 후 재설정
                    plt.rcdefaults()
                    
                    # 강력한 폰트 설정
                    rcParams['font.family'] = font_name
                    rcParams['font.sans-serif'] = [font_name, 'DejaVu Sans', 'Arial Unicode MS']
                    rcParams['axes.unicode_minus'] = False
                    rcParams['font.size'] = 10
                    
                    plt.rcParams['font.family'] = font_name
                    plt.rcParams['font.sans-serif'] = [font_name, 'DejaVu Sans', 'Arial Unicode MS']
                    plt.rcParams['axes.unicode_minus'] = False
                    plt.rcParams['font.size'] = 10
                    
                    # 폰트 캐시 재구성
                    fm._rebuild()
                    
                    self.logger.info(f"✅ 한글 폰트 파일 설정 완료: {font_name} ({available_font_path})")
                    return
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ 폰트 파일 설정 실패: {e}")
            
            # 4. 시스템 폰트 목록에서 한국어 폰트 찾기
            available_fonts = [f.name for f in fm.fontManager.ttflist]
            korean_fonts = [
                'AppleGothic', 'Apple Gothic', 'AppleSDGothicNeo-Regular',
                'NanumGothic', 'Nanum Gothic', 'NanumBarunGothic',
                'Malgun Gothic', 'Microsoft YaHei', 'SimHei',
                'Gulim', 'Batang', 'Dotum'
            ]
                
            found_font = None
            for korean_font in korean_fonts:
                if korean_font in available_fonts:
                    found_font = korean_font
                    break
            
            if found_font:
                plt.rcdefaults()
                rcParams['font.family'] = found_font
                rcParams['font.sans-serif'] = [found_font, 'DejaVu Sans']
                rcParams['axes.unicode_minus'] = False
                rcParams['font.size'] = 10
                
                plt.rcParams['font.family'] = found_font
                plt.rcParams['font.sans-serif'] = [found_font, 'DejaVu Sans']
                plt.rcParams['axes.unicode_minus'] = False
                plt.rcParams['font.size'] = 10
                
                self.logger.info(f"✅ 시스템 한글 폰트 사용: {found_font}")
            else:
                # 5. 최후 수단: 유니코드 지원 폰트
                plt.rcdefaults()
                rcParams['font.family'] = 'DejaVu Sans'
                rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'Liberation Sans']
                rcParams['axes.unicode_minus'] = False
                rcParams['font.size'] = 10
                
                plt.rcParams['font.family'] = 'DejaVu Sans'
                plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'Liberation Sans']
                plt.rcParams['axes.unicode_minus'] = False
                plt.rcParams['font.size'] = 10
                
                self.logger.warning("⚠️ 한글 폰트를 찾을 수 없어 DejaVu Sans 사용")
                    
        except Exception as e:
            self.logger.error(f"❌ 플롯용 폰트 설정 실패: {str(e)}")
            # 안전한 기본 설정
            plt.rcParams['font.family'] = 'DejaVu Sans'
            plt.rcParams['axes.unicode_minus'] = False
    
    def analyze_time_distribution(self, df: pd.DataFrame, date_column: str = 'date') -> Dict:
        """
        시간 분포 분석
        Args:
            df: 데이터프레임
            date_column: 날짜 컬럼명
        Returns:
            분포 분석 결과
        """
        try:
            self.logger.info("📊 시간 분포 분석 시작")
            
            # 날짜 컬럼 확인 및 변환
            if date_column not in df.columns:
                self.logger.error(f"❌ 날짜 컬럼 '{date_column}'을 찾을 수 없습니다.")
                return None
            
            # 날짜 형식 통일
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
            df = df.dropna(subset=[date_column])
            
            if len(df) == 0:
                self.logger.error("❌ 유효한 날짜 데이터가 없습니다.")
                return None
            
            # 기본 통계
            date_range = {
                'start': df[date_column].min(),
                'end': df[date_column].max(),
                'total_days': (df[date_column].max() - df[date_column].min()).days + 1
            }
            
            self.logger.info(f"📅 분석 기간: {date_range['start'].strftime('%Y-%m-%d')} ~ {date_range['end'].strftime('%Y-%m-%d')} ({date_range['total_days']}일)")
            
            # 월별 분포 분석
            monthly_dist = self._analyze_monthly_distribution(df, date_column)
            
            # 주간 분포 분석
            weekly_dist = self._analyze_weekly_distribution(df, date_column)
            
            # 일별 분포 분석
            daily_dist = self._analyze_daily_distribution(df, date_column)
            
            # 최적 분석 단위 결정
            optimal_unit = self._determine_optimal_time_unit(monthly_dist, weekly_dist, daily_dist)
            
            result = {
                'date_range': date_range,
                'total_comments': len(df),
                'monthly_distribution': monthly_dist,
                'weekly_distribution': weekly_dist,
                'daily_distribution': daily_dist,
                'optimal_time_unit': optimal_unit,
                'recommendation': self._generate_recommendation(optimal_unit, monthly_dist, weekly_dist)
            }
            
            self.logger.info(f"✅ 시간 분포 분석 완료: 최적 단위 = {optimal_unit}")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ 시간 분포 분석 실패: {str(e)}")
            return None
    
    def _analyze_monthly_distribution(self, df: pd.DataFrame, date_column: str) -> Dict:
        """월별 분포 분석"""
        try:
            # 월별 그룹화
            df['year_month'] = df[date_column].dt.to_period('M')
            monthly_counts = df.groupby('year_month').size()
            
            if len(monthly_counts) == 0:
                return {'valid': False, 'reason': '월별 데이터 없음'}
            
            # 통계 계산
            stats = {
                'count': len(monthly_counts),
                'mean': monthly_counts.mean(),
                'median': monthly_counts.median(),
                'std': monthly_counts.std(),
                'min': monthly_counts.min(),
                'max': monthly_counts.max(),
                'q10': monthly_counts.quantile(0.1),
                'q90': monthly_counts.quantile(0.9)
            }
            
            # 분포 품질 평가
            cv = stats['std'] / stats['mean'] if stats['mean'] > 0 else float('inf')  # 변동계수
            outlier_ratio = len(monthly_counts[monthly_counts < stats['q10']]) / len(monthly_counts)
            
            # 유효성 검사
            valid = (
                stats['count'] >= self.min_periods and
                stats['min'] >= self.min_comments_per_period and
                cv < 2.0 and  # 변동계수가 너무 크지 않음
                outlier_ratio < 0.3  # 아웃라이어가 30% 미만
            )
            
            return {
                'valid': valid,
                'stats': stats,
                'coefficient_of_variation': cv,
                'outlier_ratio': outlier_ratio,
                'data': monthly_counts.to_dict(),
                'periods': [str(period) for period in monthly_counts.index]
            }
            
        except Exception as e:
            self.logger.error(f"❌ 월별 분포 분석 실패: {str(e)}")
            return {'valid': False, 'reason': str(e)}
    
    def _analyze_weekly_distribution(self, df: pd.DataFrame, date_column: str) -> Dict:
        """주간 분포 분석"""
        try:
            # 주간 그룹화 (ISO 주차)
            df['year_week'] = df[date_column].dt.to_period('W')
            weekly_counts = df.groupby('year_week').size()
            
            if len(weekly_counts) == 0:
                return {'valid': False, 'reason': '주간 데이터 없음'}
            
            # 통계 계산
            stats = {
                'count': len(weekly_counts),
                'mean': weekly_counts.mean(),
                'median': weekly_counts.median(),
                'std': weekly_counts.std(),
                'min': weekly_counts.min(),
                'max': weekly_counts.max(),
                'q10': weekly_counts.quantile(0.1),
                'q90': weekly_counts.quantile(0.9)
            }
            
            # 분포 품질 평가
            cv = stats['std'] / stats['mean'] if stats['mean'] > 0 else float('inf')
            outlier_ratio = len(weekly_counts[weekly_counts < stats['q10']]) / len(weekly_counts)
            low_data_weeks = len(weekly_counts[weekly_counts < self.weekly_merge_threshold])
            
            # 유효성 검사
            valid = (
                stats['count'] >= self.min_periods * 4 and  # 월별보다 4배 많은 기간 필요
                stats['median'] >= self.min_comments_per_period / 4 and  # 주간은 월별의 1/4 수준
                cv < 3.0 and  # 주간은 변동이 더 클 수 있음
                outlier_ratio < 0.4 and  # 아웃라이어 허용도 높임
                low_data_weeks / len(weekly_counts) < 0.5  # 저데이터 주간이 50% 미만
            )
            
            return {
                'valid': valid,
                'stats': stats,
                'coefficient_of_variation': cv,
                'outlier_ratio': outlier_ratio,
                'low_data_weeks': low_data_weeks,
                'low_data_ratio': low_data_weeks / len(weekly_counts),
                'data': weekly_counts.to_dict(),
                'periods': [str(period) for period in weekly_counts.index]
            }
            
        except Exception as e:
            self.logger.error(f"❌ 주간 분포 분석 실패: {str(e)}")
            return {'valid': False, 'reason': str(e)}
    
    def _analyze_daily_distribution(self, df: pd.DataFrame, date_column: str) -> Dict:
        """일별 분포 분석 (참고용)"""
        try:
            # 일별 그룹화
            df['date_only'] = df[date_column].dt.date
            daily_counts = df.groupby('date_only').size()
            
            if len(daily_counts) == 0:
                return {'valid': False, 'reason': '일별 데이터 없음'}
            
            # 기본 통계만 계산 (일별 분석은 너무 세분화되어 주로 참고용)
            stats = {
                'count': len(daily_counts),
                'mean': daily_counts.mean(),
                'median': daily_counts.median(),
                'std': daily_counts.std(),
                'min': daily_counts.min(),
                'max': daily_counts.max()
            }
            
            return {
                'valid': False,  # 일별 분석은 기본적으로 사용하지 않음
                'stats': stats,
                'reason': '일별 분석은 너무 세분화됨'
            }
            
        except Exception as e:
            self.logger.error(f"❌ 일별 분포 분석 실패: {str(e)}")
            return {'valid': False, 'reason': str(e)}
    
    def _determine_optimal_time_unit(self, monthly_dist: Dict, weekly_dist: Dict, daily_dist: Dict) -> str:
        """최적 시간 단위 결정 (최소 시간 단위 제약 적용)"""
        try:
            # 🔧 최소 시간 단위 제약 적용
            self.logger.info(f"🔧 최소 시간 단위 제약: {self.min_time_unit}")
            
            # 유효성 검사
            monthly_valid = monthly_dist.get('valid', False)
            weekly_valid = weekly_dist.get('valid', False)
            daily_valid = daily_dist.get('valid', False)
            
            # 데이터 품질 점수 계산
            monthly_score = self._calculate_distribution_quality_score(monthly_dist) if monthly_valid else 0
            weekly_score = self._calculate_distribution_quality_score(weekly_dist) if weekly_valid else 0
            daily_score = self._calculate_distribution_quality_score(daily_dist) if daily_valid else 0
            
            self.logger.info(f"📊 분포 품질 점수 - 월별: {monthly_score:.2f}, 주간: {weekly_score:.2f}, 일별: {daily_score:.2f}")
            
            # 데이터 밀도 분석
            monthly_density = self._analyze_data_density(monthly_dist)
            weekly_density = self._analyze_data_density(weekly_dist)
            
            # 🔧 최소 시간 단위에 따른 결정 로직
            if self.min_time_unit == 'monthly':
                # 월별만 허용
                if monthly_valid and monthly_score >= 0.4:
                    self.logger.info("📅 월별 분석 적용 (최소 단위 제약)")
                    return 'monthly'
                else:
                    self.logger.warning("⚠️ 월별 분석이 부적절하지만 최소 단위 제약으로 강제 적용")
                    return 'monthly'
                    
            elif self.min_time_unit == 'weekly':
                # 월별 또는 주간 허용
                if monthly_valid and monthly_score >= 0.7:
                    # 월별 분석이 충분히 좋은 경우
                    if monthly_density['high_density_ratio'] > 0.3:
                        # 고밀도 구간이 30% 이상이면 주간으로 세분화
                        if weekly_valid and weekly_score >= 0.5:
                            self.logger.info("📅 주간 분석 적용 (고밀도 구간 세분화)")
                            return 'weekly'
                        else:
                            self.logger.info("📅 월별 분석 적용 (주간 분석 불가)")
                            return 'monthly'
                    else:
                        self.logger.info("📅 월별 분석 적용")
                        return 'monthly'
                elif weekly_valid and weekly_score >= 0.6:
                    # 주간 분석이 적절한 경우
                    self.logger.info("📅 주간 분석 적용")
                    return 'weekly'
                elif monthly_valid and monthly_score >= 0.4:
                    # 월별이 최소한 사용 가능한 경우
                    self.logger.info("📅 월별 분석 적용")
                    return 'monthly'
                elif weekly_valid:
                    # 주간이라도 사용 가능한 경우
                    self.logger.info("📅 주간 분석 적용")
                    return 'weekly'
                else:
                    # 최소 단위 제약으로 주간 강제 적용
                    self.logger.warning("⚠️ 주간 분석이 부적절하지만 최소 단위 제약으로 강제 적용")
                    return 'weekly'
                    
            elif self.min_time_unit == 'daily':
                # 모든 단위 허용 (기존 로직)
                if monthly_valid and monthly_score >= 0.7:
                    # 월별 분석이 충분히 좋은 경우
                    if monthly_density['high_density_ratio'] > 0.3:
                        # 고밀도 구간이 30% 이상이면 혼합 방식 (주간/일별 세분화)
                        self.logger.info("📅 월별 분석이 우수하나 고밀도 구간이 많아 혼합 분석을 적용합니다.")
                        return 'hybrid'
                    else:
                        self.logger.info("📅 월별 분석이 적합합니다.")
                    return 'monthly'
                elif weekly_valid and weekly_score >= 0.6:
                    # 주간 분석이 적절한 경우
                    self.logger.info("📅 주간 분석이 적합합니다.")
                    return 'weekly'
                elif monthly_valid and monthly_score >= 0.4:
                    # 월별이 최소한 사용 가능한 경우
                    self.logger.info("📅 월별 분석이 적절합니다.")
                    return 'monthly'
                elif weekly_valid:
                    # 주간이라도 사용 가능한 경우
                    self.logger.info("📅 주간 분석을 적용합니다.")
                    return 'weekly'
                elif daily_valid:
                    # 일별만 가능한 경우
                    self.logger.info("📅 일별 분석을 적용합니다.")
                    return 'daily'
                else:
                    # 모든 단위가 부적절한 경우 혼합 방식으로 최대한 활용
                    self.logger.warning("⚠️ 모든 단위가 부적절하여 혼합 분석을 강제 적용합니다.")
                    return 'hybrid'
            
            # 기본값 (설정 오류 시)
            self.logger.warning(f"⚠️ 알 수 없는 최소 시간 단위: {self.min_time_unit}, 기본값 적용")
            return self.min_time_unit if self.min_time_unit in ['monthly', 'weekly', 'daily'] else 'weekly'
            
        except Exception as e:
            self.logger.error(f"❌ 최적 시간 단위 결정 실패: {str(e)}")
            return self.min_time_unit if hasattr(self, 'min_time_unit') else 'weekly'  # 안전한 기본값
    
    def _calculate_distribution_quality_score(self, dist: Dict) -> float:
        """분포 품질 점수 계산 (0~1)"""
        try:
            if not dist.get('valid', False):
                return 0.0
            
            stats = dist.get('stats', {})
            if not stats:
                return 0.0
            
            # 기본 점수 (유효한 분포면 0.5)
            score = 0.5
            
            # 1. 기간 수 점수 (더 많은 기간이 좋음)
            period_count = stats.get('count', 0)
            if period_count >= 6:
                score += 0.2
            elif period_count >= 3:
                score += 0.1
            
            # 2. 변동계수 점수 (적당한 변동이 좋음)
            cv = dist.get('coefficient_of_variation', float('inf'))
            if 0.3 <= cv <= 1.5:
                score += 0.2
            elif cv < 2.0:
                score += 0.1
            
            # 3. 아웃라이어 비율 점수 (적은 아웃라이어가 좋음)
            outlier_ratio = dist.get('outlier_ratio', 1.0)
            if outlier_ratio < 0.2:
                score += 0.1
            elif outlier_ratio < 0.4:
                score += 0.05
            
            # 4. 최소값 점수 (충분한 데이터가 좋음)
            min_count = stats.get('min', 0)
            if min_count >= self.min_comments_per_period * 2:
                score += 0.1
            elif min_count >= self.min_comments_per_period:
                score += 0.05
            
            return min(score, 1.0)
            
        except Exception as e:
            self.logger.warning(f"⚠️ 분포 품질 점수 계산 실패: {e}")
            return 0.0
    
    def _analyze_data_density(self, dist: Dict) -> Dict:
        """데이터 밀도 분석 (비율 기반)"""
        try:
            if not dist.get('valid', False):
                return {'high_density_ratio': 0, 'density_analysis': 'invalid'}
            
            stats = dist.get('stats', {})
            data = dist.get('data', {})
            
            if not stats or not data:
                return {'high_density_ratio': 0, 'density_analysis': 'no_data'}
            
            # 전체 댓글 수 계산
            total_comments = sum(data.values())
            
            # 비율 기반 고밀도 임계값 (전체의 20% 이상을 차지하는 기간)
            high_density_comment_threshold = total_comments * 0.20
            
            # 또는 평균의 2배 이상인 기간 (둘 중 더 관대한 기준 사용)
            mean_based_threshold = stats.get('mean', 0) * 2.0
            high_density_threshold = min(high_density_comment_threshold, mean_based_threshold)
            
            # 고밀도 구간 비율 계산
            high_density_periods = [period for period, count in data.items() if count >= high_density_threshold]
            high_density_count = len(high_density_periods)
            total_periods = len(data)
            high_density_ratio = high_density_count / total_periods if total_periods > 0 else 0
            
            # 고밀도 구간의 댓글 비율 계산
            high_density_comments = sum(data[period] for period in high_density_periods)
            high_density_comment_ratio = high_density_comments / total_comments if total_comments > 0 else 0
            
            # 밀도 분석 결과 (기간 비율과 댓글 비율 모두 고려)
            if high_density_ratio > 0.3 or high_density_comment_ratio > 0.5:
                density_analysis = 'high_concentration'  # 고집중
            elif high_density_ratio > 0.15 or high_density_comment_ratio > 0.3:
                density_analysis = 'moderate_concentration'  # 중간집중
            else:
                density_analysis = 'low_concentration'  # 저집중
            
            return {
                'high_density_ratio': high_density_ratio,
                'high_density_comment_ratio': high_density_comment_ratio,
                'high_density_threshold': high_density_threshold,
                'high_density_periods': high_density_count,
                'high_density_periods_list': high_density_periods,
                'total_periods': total_periods,
                'total_comments': total_comments,
                'density_analysis': density_analysis
            }
                
        except Exception as e:
            self.logger.warning(f"⚠️ 데이터 밀도 분석 실패: {e}")
            return {'high_density_ratio': 0, 'density_analysis': 'error'}
    
    def _generate_recommendation(self, optimal_unit: str, monthly_dist: Dict, weekly_dist: Dict) -> Dict:
        """분석 권장사항 생성"""
        try:
            recommendations = {
                'time_unit': optimal_unit,
                'reasons': [],
                'warnings': [],
                'suggestions': []
            }
            
            if optimal_unit == 'monthly':
                recommendations['reasons'].append("월별 데이터 분포가 안정적이고 충분합니다.")
                if monthly_dist.get('stats', {}).get('count', 0) < 12:
                    recommendations['warnings'].append("분석 기간이 1년 미만으로 계절성 분석에 제한이 있을 수 있습니다.")
            
            elif optimal_unit == 'weekly':
                recommendations['reasons'].append("주간 데이터 분포가 월별보다 더 세밀한 분석을 가능하게 합니다.")
                recommendations['warnings'].append("주간 분석은 노이즈가 많을 수 있으니 트렌드 해석에 주의하세요.")
                
                low_data_ratio = weekly_dist.get('low_data_ratio', 0)
                if low_data_ratio > 0.3:
                    recommendations['suggestions'].append(f"데이터가 적은 주간({low_data_ratio:.1%})은 인접 주간과 병합을 고려하세요.")
            
            elif optimal_unit == 'hybrid':
                recommendations['reasons'].append("데이터 분포가 불균등하여 혼합 분석이 필요합니다.")
                recommendations['suggestions'].append("데이터가 충분한 기간은 주간 분석, 부족한 기간은 월별 분석을 적용합니다.")
            
            elif optimal_unit == 'monthly_forced':
                recommendations['warnings'].append("데이터 분포가 이상적이지 않지만 월별 분석을 강제 적용합니다.")
                recommendations['suggestions'].append("결과 해석 시 데이터 품질 한계를 고려하세요.")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"❌ 권장사항 생성 실패: {str(e)}")
            return {'time_unit': optimal_unit, 'reasons': [], 'warnings': [], 'suggestions': []}
    
    def create_adaptive_subdivided_groups(self, df: pd.DataFrame, date_column: str = 'date') -> Dict:
        """
        적응적 세분화 그룹 생성 (config.py 설정 기반)
        Args:
            df: 데이터프레임
            date_column: 날짜 컬럼명
        Returns:
            세분화된 시간 그룹 정보
        """
        try:
            if not self.adaptive_enabled:
                return self.create_adaptive_time_groups(df, 'monthly', date_column)
            
            self.logger.info("🔄 적응적 시간 세분화 분석 시작")
            
            # 1. 월별 기본 분석
            monthly_groups = self._create_monthly_groups(df, date_column)
            
            # 2. 비율 기반 세분화 필요성 검토
            total_comments = len(df)
            subdivisions_needed = []
            
            for period, group_data in monthly_groups.items():
                period_ratio = len(group_data) / total_comments
                if period_ratio >= self.high_ratio_threshold:
                    subdivisions_needed.append({
                        'period': period,
                        'ratio': period_ratio,
                        'count': len(group_data),
                        'data': group_data
                    })
            
            if not subdivisions_needed:
                self.logger.info("✅ 세분화 불필요 - 월별 분석 사용")
                return {'groups': monthly_groups, 'metadata': {'period_format': 'monthly'}}
            
            # 3. 세분화 실행
            final_groups = {'groups': {}, 'metadata': {}}
            
            for period, group_data in monthly_groups.items():
                period_ratio = len(group_data) / total_comments
                
                if period_ratio >= self.high_ratio_threshold:
                    # 세분화 필요
                    subdivided = self._subdivide_period(group_data, period, date_column)
                    final_groups['groups'].update(subdivided['groups'])
                    self.logger.info(f"📊 {period} 세분화: {len(subdivided['groups'])}개 구간")
                else:
                    # 세분화 불필요
                    start_date = group_data[date_column].min().strftime('%Y-%m-%d')
                    end_date = group_data[date_column].max().strftime('%Y-%m-%d')
                    period_key = f"{start_date} ~ {end_date}"
                    final_groups['groups'][period_key] = group_data
            
            # 4. 메타데이터 생성
            final_groups['metadata'] = {
                'total_periods': len(final_groups['groups']),
                'subdivisions_applied': len(subdivisions_needed),
                'high_ratio_threshold': self.high_ratio_threshold,
                'period_format': 'adaptive_subdivided'
            }
            
            self.logger.info(f"✅ 적응적 세분화 완료: {len(final_groups['groups'])}개 구간")
            return final_groups
            
        except Exception as e:
            self.logger.error(f"❌ 적응적 세분화 실패: {str(e)}")
            import traceback
            self.logger.error(f"상세 오류: {traceback.format_exc()}")
            return self.create_adaptive_time_groups(df, 'monthly', date_column)
    
    def _subdivide_period(self, period_data: pd.DataFrame, period_name: str, date_column: str) -> Dict:
        """
        특정 기간을 세분화 (최소 시간 단위 제약 적용)
        Args:
            period_data: 기간 데이터
            period_name: 기간명
            date_column: 날짜 컬럼명
        Returns:
            세분화된 그룹
        """
        try:
            # 🔧 최소 시간 단위 제약에 따른 세분화 로직
            if self.min_time_unit == 'monthly':
                # 월별만 허용 - 세분화 안함
                start_date = period_data[date_column].min().strftime('%Y-%m-%d')
                end_date = period_data[date_column].max().strftime('%Y-%m-%d')
                period_key = f"{start_date} ~ {end_date}"
                self.logger.info(f"🔧 월별 최소 단위 제약으로 세분화 생략: {period_name}")
                return {'groups': {period_key: period_data}}
                
            elif self.min_time_unit == 'weekly':
                # 주별까지만 허용 - 주별 세분화만 시도
                weekly_groups = self._create_weekly_groups_for_period(period_data, date_column)
                
                # 주별 세분화가 효과적인지 확인
                if len(weekly_groups['groups']) > 1:
                    # 각 주의 댓글 수가 충분한지 확인
                    valid_weeks = {}
                    for week_key, week_data in weekly_groups['groups'].items():
                        if len(week_data) >= self.min_comments_per_period:
                            valid_weeks[week_key] = week_data
                    
                    if len(valid_weeks) > 1:
                        self.logger.info(f"🔧 주별 세분화 적용: {period_name} → {len(valid_weeks)}개 주간")
                        return {'groups': valid_weeks}
                
                # 주별 세분화가 효과적이지 않으면 원본 반환
                start_date = period_data[date_column].min().strftime('%Y-%m-%d')
                end_date = period_data[date_column].max().strftime('%Y-%m-%d')
                period_key = f"{start_date} ~ {end_date}"
                self.logger.info(f"🔧 주별 세분화 불가로 월별 유지: {period_name}")
                return {'groups': {period_key: period_data}}
                
            elif self.min_time_unit == 'daily':
                # 모든 단위 허용 - 기존 로직
                # 주별 세분화 시도
                weekly_groups = self._create_weekly_groups_for_period(period_data, date_column)
                
                # 주별 세분화가 효과적인지 확인
                if len(weekly_groups['groups']) > 1:
                    # 각 주의 댓글 수가 충분한지 확인
                    valid_weeks = {}
                    for week_key, week_data in weekly_groups['groups'].items():
                        if len(week_data) >= self.min_comments_per_period:
                            valid_weeks[week_key] = week_data
                    
                    if len(valid_weeks) > 1:
                        self.logger.info(f"🔧 주별 세분화 적용: {period_name} → {len(valid_weeks)}개 주간")
                        return {'groups': valid_weeks}
                
                # 주별 세분화가 효과적이지 않으면 일별 시도
                daily_groups = self._create_daily_groups_for_period(period_data, date_column)
                
                if len(daily_groups['groups']) > 1:
                    valid_days = {}
                    for day_key, day_data in daily_groups['groups'].items():
                        if len(day_data) >= self.min_comments_per_period:
                            valid_days[day_key] = day_data
                    
                    if len(valid_days) > 1:
                        self.logger.info(f"🔧 일별 세분화 적용: {period_name} → {len(valid_days)}개 일간")
                        return {'groups': valid_days}
                
                # 세분화가 불가능하면 원본 반환
                start_date = period_data[date_column].min().strftime('%Y-%m-%d')
                end_date = period_data[date_column].max().strftime('%Y-%m-%d')
                period_key = f"{start_date} ~ {end_date}"
                self.logger.info(f"🔧 세분화 불가로 월별 유지: {period_name}")
                return {'groups': {period_key: period_data}}
            
            # 기본값 (설정 오류 시)
            start_date = period_data[date_column].min().strftime('%Y-%m-%d')
            end_date = period_data[date_column].max().strftime('%Y-%m-%d')
            period_key = f"{start_date} ~ {end_date}"
            self.logger.warning(f"⚠️ 알 수 없는 최소 시간 단위: {self.min_time_unit}, 월별 유지")
            return {'groups': {period_key: period_data}}
            
        except Exception as e:
            self.logger.error(f"❌ 기간 세분화 실패: {str(e)}")
            return {'groups': {period_name: period_data}}
    
    def _create_weekly_groups_for_period(self, period_data: pd.DataFrame, date_column: str) -> Dict:
        """특정 기간의 주별 그룹 생성"""
        try:
            period_data = period_data.copy()
            period_data['week'] = period_data[date_column].dt.strftime('%Y-W%U')
            
            groups = {}
            for week, group in period_data.groupby('week'):
                start_date = group[date_column].min().strftime('%Y-%m-%d')
                end_date = group[date_column].max().strftime('%Y-%m-%d')
                week_key = f"{start_date} ~ {end_date}"
                groups[week_key] = group.drop('week', axis=1)
            
            return {'groups': groups}
            
        except Exception as e:
            self.logger.error(f"❌ 주별 그룹 생성 실패: {str(e)}")
            return {'groups': {}}
    
    def _create_daily_groups_for_period(self, period_data: pd.DataFrame, date_column: str) -> Dict:
        """특정 기간의 일별 그룹 생성"""
        try:
            period_data = period_data.copy()
            period_data['day'] = period_data[date_column].dt.strftime('%Y-%m-%d')
            
            groups = {}
            for day, group in period_data.groupby('day'):
                groups[day] = group.drop('day', axis=1)
            
            return {'groups': groups}
            
        except Exception as e:
            self.logger.error(f"❌ 일별 그룹 생성 실패: {str(e)}")
            return {'groups': {}}

    def create_adaptive_time_groups(self, df: pd.DataFrame, optimal_unit: str, date_column: str = 'date') -> Dict:
        """
        적응적 시간 그룹 생성
        Args:
            df: 데이터프레임
            optimal_unit: 최적 시간 단위
            date_column: 날짜 컬럼명
        Returns:
            시간 그룹별 데이터
        """
        try:
            self.logger.info(f"🔄 {optimal_unit} 기준으로 시간 그룹 생성 시작")
            
            # 날짜 컬럼 확인 및 변환
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
            df = df.dropna(subset=[date_column])
            
            if optimal_unit == 'monthly' or optimal_unit == 'monthly_forced':
                groups = self._create_monthly_groups(df, date_column)
                return {'groups': groups, 'metadata': {'period_format': 'monthly'}}
            
            elif optimal_unit == 'weekly':
                groups = self._create_weekly_groups(df, date_column)
                return {'groups': groups, 'metadata': {'period_format': 'weekly'}}
            
            elif optimal_unit == 'hybrid':
                groups = self._create_adaptive_hybrid_groups(df, date_column)
                return {'groups': groups, 'metadata': {'period_format': 'hybrid'}}
            
            else:
                self.logger.warning(f"⚠️ 알 수 없는 시간 단위 '{optimal_unit}', 월별로 대체합니다.")
                groups = self._create_monthly_groups(df, date_column)
                return {'groups': groups, 'metadata': {'period_format': 'monthly'}}
                
        except Exception as e:
            self.logger.error(f"❌ 적응적 시간 그룹 생성 실패: {str(e)}")
            return {'groups': {}, 'metadata': {'period_format': 'error'}}
    
    def _create_monthly_groups(self, df: pd.DataFrame, date_column: str) -> Dict:
        """월별 그룹 생성 (최소 댓글 수 필터링 적용)"""
        try:
            df['year_month'] = df[date_column].dt.to_period('M')
            monthly_groups = {}
            filtered_count = 0
            
            for period, group_df in df.groupby('year_month'):
                period_str = str(period)
                
                # 최소 댓글 수 확인
                if len(group_df) >= self.absolute_min_threshold:
                    monthly_groups[period_str] = group_df.drop('year_month', axis=1).copy()
                    self.logger.info(f"📅 {period_str}: {len(group_df)}개 댓글 포함")
                else:
                    filtered_count += 1
                    self.logger.info(f"⚠️ {period_str}: 댓글 수 부족({len(group_df)}개 < {self.absolute_min_threshold}개), 제외")
                
            if filtered_count > 0:
                self.logger.info(f"📊 월별 그룹 필터링: {filtered_count}개 월 제외됨")
                
            self.logger.info(f"📅 월별 그룹 생성 완료: {len(monthly_groups)}개 월 (필터링 적용)")
            return monthly_groups
            
        except Exception as e:
            self.logger.error(f"❌ 월별 그룹 생성 실패: {str(e)}")
            return {}
    
    def _create_weekly_groups(self, df: pd.DataFrame, date_column: str) -> Dict:
        """주간 그룹 생성 (저데이터 주간 병합 및 절대 최소값 필터링)"""
        try:
            df['year_week'] = df[date_column].dt.to_period('W')
            weekly_counts = df.groupby('year_week').size()
            
            # 저데이터 주간 식별
            low_data_weeks = weekly_counts[weekly_counts < self.weekly_merge_threshold].index
            
            weekly_groups = {}
            merge_buffer = []
            
            for period, group_df in df.groupby('year_week'):
                period_str = str(period)
                
                if period in low_data_weeks:
                    # 저데이터 주간은 버퍼에 추가 (절대 최소값 확인)
                    if len(group_df) >= self.absolute_min_threshold:
                        merge_buffer.append((period_str, group_df.drop('year_week', axis=1).copy()))
                    else:
                        self.logger.info(f"⚠️ 주간 {period_str}: 데이터 너무 부족({len(group_df)}개), 완전 제외")
                else:
                    # 정상 데이터 주간
                    if merge_buffer:
                        # 이전 저데이터 주간들과 병합
                        merged_df = pd.concat([item[1] for item in merge_buffer] + [group_df.drop('year_week', axis=1).copy()])
                        
                        # 병합 후에도 최소 임계값 확인
                        if len(merged_df) >= self.absolute_min_threshold:
                            merged_period = f"{merge_buffer[0][0]}_to_{period_str}"
                            weekly_groups[merged_period] = merged_df
                            self.logger.info(f"📅 {merged_period}: 저데이터 주간 병합 ({len(merged_df)}개 댓글)")
                        else:
                            self.logger.info(f"⚠️ 병합 주간 {merge_buffer[0][0]}_to_{period_str}: 병합 후에도 데이터 부족({len(merged_df)}개), 제외")
                        
                        merge_buffer = []
                    else:
                        # 단독 정상 주간도 최소 임계값 확인
                        if len(group_df) >= self.absolute_min_threshold:
                            weekly_groups[period_str] = group_df.drop('year_week', axis=1).copy()
                        else:
                            self.logger.info(f"⚠️ 주간 {period_str}: 데이터 부족({len(group_df)}개), 제외")
            
            # 마지막에 남은 저데이터 주간들 처리
            if merge_buffer:
                if len(weekly_groups) > 0:
                    # 마지막 정상 주간과 병합
                    last_key = list(weekly_groups.keys())[-1]
                    last_df = weekly_groups[last_key]
                    merged_df = pd.concat([last_df] + [item[1] for item in merge_buffer])
                    
                    # 병합 후 최소 임계값 확인
                    if len(merged_df) >= self.absolute_min_threshold:
                        weekly_groups[f"{last_key}_extended"] = merged_df
                        del weekly_groups[last_key]
                        self.logger.info(f"📅 {last_key}_extended: 마지막 저데이터 주간 병합 ({len(merged_df)}개 댓글)")
                    else:
                        self.logger.info(f"⚠️ 마지막 병합 주간: 데이터 부족({len(merged_df)}개), 기존 유지")
                else:
                    # 모든 주간이 저데이터인 경우 전체 병합
                    merged_df = pd.concat([item[1] for item in merge_buffer])
                    if len(merged_df) >= self.absolute_min_threshold:
                        weekly_groups['all_weeks_merged'] = merged_df
                        self.logger.info(f"📅 all_weeks_merged: 전체 저데이터 주간 병합 ({len(merged_df)}개 댓글)")
                    else:
                        self.logger.info(f"⚠️ 전체 주간 병합: 데이터 부족({len(merged_df)}개), 주간 분석 불가")
            
            self.logger.info(f"📅 주간 그룹 생성 완료: {len(weekly_groups)}개 주간 (필터링 강화)")
            return weekly_groups
            
        except Exception as e:
            self.logger.error(f"❌ 주간 그룹 생성 실패: {str(e)}")
            return {}
    
    def _create_adaptive_hybrid_groups(self, df: pd.DataFrame, date_column: str) -> Dict:
        """
        적응형 혼합 그룹 생성 (데이터 밀도에 따라 월별/주간 혼합)
        고밀도 구간은 주간으로 세분화, 저밀도 구간은 월별 유지
        """
        try:
            self.logger.info("🔄 적응형 혼합 그룹 생성 시작")
            
            # 먼저 월별 분포 분석
            monthly_groups = self._create_monthly_groups(df, date_column)
            
            adaptive_groups = {}
            
            # 전체 댓글 수 계산
            total_comments = sum(len(month_df) for month_df in monthly_groups.values())
            
            # 비율 기반 임계값 설정
            high_density_ratio = 0.25  # 전체의 25% 이상이면 주간 세분화
            medium_density_ratio = 0.05  # 전체의 5% 이상이면 월별 유지
            min_ratio = 0.01  # 전체의 1% 이상이면 최소 유지
            
            high_density_threshold = max(total_comments * high_density_ratio, self.min_comments_per_period)
            medium_density_threshold = max(total_comments * medium_density_ratio, self.absolute_min_threshold)
            min_threshold = max(total_comments * min_ratio, self.absolute_min_threshold)
            
            self.logger.info(f"📊 비율 기반 임계값 설정 (총 {total_comments:,}개 댓글)")
            self.logger.info(f"  🔥 고밀도 임계값: {high_density_threshold:.0f}개 ({high_density_ratio:.1%})")
            self.logger.info(f"  📊 중밀도 임계값: {medium_density_threshold:.0f}개 ({medium_density_ratio:.1%})")
            self.logger.info(f"  📉 최소 임계값: {min_threshold:.0f}개 ({min_ratio:.1%})")
            
            for month_key, month_df in monthly_groups.items():
                comment_count = len(month_df)
                comment_ratio = comment_count / total_comments
                
                if comment_count >= high_density_threshold:
                    # 🔧 최소 시간 단위 제약 확인
                    if self.min_time_unit == 'monthly':
                        # 월별만 허용 - 세분화 안함
                        self.logger.info(f"🔧 {month_key}: {comment_count:,}개 댓글 ({comment_ratio:.1%}) → 월별 유지 (최소 단위 제약)")
                        adaptive_groups[month_key] = month_df.copy()
                    elif self.min_time_unit in ['weekly', 'daily']:
                        # 주간 세분화 허용
                        self.logger.info(f"🔥 {month_key}: {comment_count:,}개 댓글 ({comment_ratio:.1%}) → 주간 세분화")
                        
                        # 올바른 주간 그룹화 사용
                        month_df_copy = month_df.copy()
                        month_df_copy['year_week'] = month_df_copy[date_column].dt.to_period('W')
                        
                        week_groups = {}
                        for week_period, week_df in month_df_copy.groupby('year_week'):
                            if len(week_df) >= self.absolute_min_threshold:
                                # 주간 키 생성 (시작일~종료일 형식)
                                week_start = week_df[date_column].min().strftime('%Y-%m-%d')
                                week_end = week_df[date_column].max().strftime('%Y-%m-%d')
                                week_key = f"{week_start} ~ {week_end}"
                                
                                week_groups[week_key] = week_df.drop('year_week', axis=1).copy()
                                week_ratio = len(week_df) / total_comments
                                self.logger.info(f"  📅 {week_key}: {len(week_df):,}개 댓글 ({week_ratio:.1%})")
                        
                        # 주간 그룹이 여러 개 생성되었으면 추가, 아니면 월별 유지
                        if len(week_groups) > 1:
                            adaptive_groups.update(week_groups)
                        else:
                            self.logger.info(f"  📅 주간 세분화 효과 없음, 월별 유지: {month_key}")
                            adaptive_groups[month_key] = month_df.copy()
                    else:
                        # 알 수 없는 설정 - 월별 유지
                        self.logger.warning(f"⚠️ 알 수 없는 최소 시간 단위: {self.min_time_unit}, 월별 유지")
                        adaptive_groups[month_key] = month_df.copy()
                
                elif comment_count >= medium_density_threshold:
                    # 중밀도 구간: 월별 유지
                    self.logger.info(f"📊 {month_key}: {comment_count:,}개 댓글 ({comment_ratio:.1%}) → 월별 유지")
                    adaptive_groups[month_key] = month_df.copy()
                
                elif comment_count >= min_threshold:
                    # 저밀도 구간: 월별 유지 (최소 기준 충족)
                    self.logger.info(f"📉 {month_key}: {comment_count:,}개 댓글 ({comment_ratio:.1%}) → 월별 유지 (저밀도)")
                    adaptive_groups[month_key] = month_df.copy()
                
                else:
                    # 매우 저밀도: 제외
                    self.logger.warning(f"⚠️ {month_key}: {comment_count:,}개 댓글 ({comment_ratio:.1%}) → 제외 (최소 기준 미달)")
            
            self.logger.info(f"✅ 적응형 혼합 그룹 생성 완료: {len(adaptive_groups)}개 그룹")
            
            # 그룹 정보 요약
            total_comments = sum(len(group_df) for group_df in adaptive_groups.values())
            weekly_groups = [k for k in adaptive_groups.keys() if '_W' in k]
            monthly_groups_kept = [k for k in adaptive_groups.keys() if '_W' not in k]
            
            self.logger.info(f"  📊 총 댓글 수: {total_comments}개")
            self.logger.info(f"  🗓️ 주간 세분화 그룹: {len(weekly_groups)}개")
            self.logger.info(f"  📅 월별 유지 그룹: {len(monthly_groups_kept)}개")
            
            return adaptive_groups
            
        except Exception as e:
            self.logger.error(f"❌ 적응형 혼합 그룹 생성 실패: {str(e)}")
            # 실패 시 기본 월별 그룹으로 폴백
            return self._create_monthly_groups(df, date_column)
    
    def _create_hybrid_groups(self, df: pd.DataFrame, date_column: str) -> Dict:
        """혼합 그룹 생성 (월별 + 주간) - 주간 우선 정책"""
        try:
            # 먼저 월별 분포 확인
            df['year_month'] = df[date_column].dt.to_period('M')
            monthly_counts = df.groupby('year_month').size()
            
            # 데이터가 충분한 월과 부족한 월 구분
            sufficient_months = monthly_counts[monthly_counts >= self.min_comments_per_period].index
            insufficient_months = monthly_counts[monthly_counts < self.min_comments_per_period].index
            
            hybrid_groups = {}
            
            # 데이터가 충분한 월은 주간 분석 (월간 구간 제거)
            for month in sufficient_months:
                month_df = df[df['year_month'] == month].copy()
                month_df['year_week'] = month_df[date_column].dt.to_period('W')
                
                week_groups = {}
                for week, week_df in month_df.groupby('year_week'):
                    if len(week_df) >= self.absolute_min_threshold:  # 주간도 최소 임계값 확인
                        week_str = str(week)
                        week_groups[week_str] = week_df.drop(['year_month', 'year_week'], axis=1).copy()
                
                # 주간 그룹이 있으면 추가 (월간 구간은 제외)
                if week_groups:
                    hybrid_groups.update(week_groups)
                    self.logger.info(f"📅 {month}: 주간 분석 적용 ({len(week_groups)}개 주간)")
            
            # 데이터가 부족한 월들은 병합하여 월별 분석 (최소 임계값 확인)
            if len(insufficient_months) > 0:
                insufficient_df = df[df['year_month'].isin(insufficient_months)].copy()
                
                # 연속된 부족한 월들을 그룹화
                month_groups = []
                current_group = []
                
                for month in sorted(insufficient_months):
                    if not current_group or (month - current_group[-1]).n == 1:
                        current_group.append(month)
                    else:
                        month_groups.append(current_group)
                        current_group = [month]
                
                if current_group:
                    month_groups.append(current_group)
                
                # 각 그룹을 하나의 기간으로 처리 (최소 임계값 확인)
                for i, month_group in enumerate(month_groups):
                    group_df = insufficient_df[insufficient_df['year_month'].isin(month_group)].copy()
                    
                    # 최소 임계값 확인
                    if len(group_df) >= self.absolute_min_threshold:
                        if len(month_group) == 1:
                            group_key = str(month_group[0])
                        else:
                            group_key = f"{month_group[0]}_to_{month_group[-1]}"
                        
                        hybrid_groups[group_key] = group_df.drop('year_month', axis=1).copy()
                        self.logger.info(f"📅 {group_key}: 월간 분석 적용 ({len(group_df)}개 댓글)")
                    else:
                        self.logger.info(f"⚠️ 월간 그룹 {month_group}: 데이터 부족({len(group_df)}개), 제외")
            
            self.logger.info(f"📅 혼합 그룹 생성 완료: {len(hybrid_groups)}개 기간 (주간 우선 정책)")
            return hybrid_groups
            
        except Exception as e:
            self.logger.error(f"❌ 혼합 그룹 생성 실패: {str(e)}")
            return {}
    
    def visualize_time_distribution(self, distribution_result: Dict, df: pd.DataFrame = None, 
                                  date_column: str = 'date', save_path: str = None) -> str:
        """
        시간 분포 시각화 (최소 시간 단위에 따른 강제 표기)
        Args:
            distribution_result: analyze_time_distribution 결과
            df: 원본 데이터프레임 (절대 시간 계산용)
            date_column: 날짜 컬럼명
            save_path: 저장 경로
        Returns:
            저장된 파일 경로
        """
        try:
            if not distribution_result:
                self.logger.error("❌ 분포 결과가 없어 시각화할 수 없습니다.")
                return None
            
            # 한글 폰트 설정 강화
            self._setup_korean_font_for_plot()
            
            # 1x2 레이아웃으로 변경 (절대 시간 기준만 표시)
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            fig.suptitle(f'시간별 댓글 분포 분석 (최소 단위: {self.min_time_unit})', fontsize=16, fontweight='bold')
            
            # 🔧 최소 시간 단위에 따른 강제 표기
            if df is not None and date_column in df.columns:
                df_copy = df.copy()
                df_copy[date_column] = pd.to_datetime(df_copy[date_column])
                
                # 1. 최소 시간 단위에 따른 첫 번째 플롯
                if self.min_time_unit == 'monthly':
                    # 월별 표기 강제
                    df_copy['time_group'] = df_copy[date_column].dt.to_period('M')
                    time_counts = df_copy.groupby('time_group').size().sort_index()
                    time_labels = [pd.to_datetime(str(period)).strftime('%Y-%m') for period in time_counts.index]
                    plot_title = '월별 댓글 분포'
                    xlabel = '년-월'
                    
                elif self.min_time_unit == 'weekly':
                    # 주별 표기 강제
                    df_copy['time_group'] = df_copy[date_column].dt.to_period('W')
                    time_counts = df_copy.groupby('time_group').size().sort_index()
                    # 주별 라벨을 "YYYY-WXX" 형식으로 표시
                    time_labels = []
                    for period in time_counts.index:
                        week_start = pd.to_datetime(str(period).split('/')[0])
                        week_num = week_start.isocalendar()[1]
                        time_labels.append(f"{week_start.year}-W{week_num:02d}")
                    plot_title = '주별 댓글 분포'
                    xlabel = '년-주차'
                    
                elif self.min_time_unit == 'daily':
                    # 일별 표기 강제
                    df_copy['time_group'] = df_copy[date_column].dt.date
                    time_counts = df_copy.groupby('time_group').size().sort_index()
                    time_labels = [date.strftime('%m-%d') for date in time_counts.index]
                    plot_title = '일별 댓글 분포'
                    xlabel = '월-일'
                else:
                    # 기본값: 월별
                    df_copy['time_group'] = df_copy[date_column].dt.to_period('M')
                    time_counts = df_copy.groupby('time_group').size().sort_index()
                    time_labels = [pd.to_datetime(str(period)).strftime('%Y-%m') for period in time_counts.index]
                    plot_title = '월별 댓글 분포 (기본값)'
                    xlabel = '년-월'
                
                # 첫 번째 플롯: 최소 시간 단위 기준
                bars1 = axes[0].bar(range(len(time_counts)), time_counts.values, 
                                      color='skyblue', alpha=0.7, edgecolor='navy', linewidth=1)
                axes[0].set_title(f'{plot_title} (강제 표기)', fontsize=14, fontweight='bold')
                axes[0].set_xlabel(xlabel, fontsize=12)
                axes[0].set_ylabel('댓글 수', fontsize=12)
                
                # x축 라벨 설정 (너무 많으면 샘플링)
                if len(time_labels) > 15:
                    # 15개 이상이면 균등하게 샘플링
                    sample_indices = np.linspace(0, len(time_labels)-1, 15, dtype=int)
                    axes[0].set_xticks(sample_indices)
                    axes[0].set_xticklabels([time_labels[i] for i in sample_indices], rotation=45, ha='right')
                else:
                    axes[0].set_xticks(range(len(time_labels)))
                    axes[0].set_xticklabels(time_labels, rotation=45, ha='right')
                
                axes[0].grid(True, alpha=0.3)
                
                # 값 표시
                for i, (bar, count) in enumerate(zip(bars1, time_counts.values)):
                    if i < len(bars1):  # 인덱스 범위 확인
                        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(time_counts.values) * 0.01,
                                       f'{count:,}', ha='center', va='bottom', fontsize=9, fontweight='bold')
                
                # 2. 실제 데이터 분포 (참고용)
                # 실제 데이터가 어떻게 분포되어 있는지 보여주는 플롯
                df_copy['actual_date'] = df_copy[date_column].dt.date
                daily_counts = df_copy.groupby('actual_date').size().sort_index()
                
                # 일별 데이터가 너무 많으면 주별로 집계
                if len(daily_counts) > 30:
                    df_copy['week_group'] = df_copy[date_column].dt.to_period('W')
                    weekly_counts = df_copy.groupby('week_group').size().sort_index()
                    
                    # 주별 라벨 생성
                    week_dates = [pd.to_datetime(str(period).split('/')[0]) for period in weekly_counts.index]
                    week_labels = [date.strftime('%m/%d') for date in week_dates]
                    
                    axes[1].plot(range(len(weekly_counts)), weekly_counts.values, 
                                   marker='o', markersize=6, linewidth=2, color='orange', alpha=0.8)
                    axes[1].set_title('실제 데이터 분포 (주별 집계)', fontsize=14, fontweight='bold')
                    axes[1].set_xlabel('주 시작일 (월/일)', fontsize=12)
                    
                    # x축 라벨 설정
                    if len(week_labels) > 15:
                        sample_indices = np.linspace(0, len(week_labels)-1, 15, dtype=int)
                        axes[1].set_xticks(sample_indices)
                        axes[1].set_xticklabels([week_labels[i] for i in sample_indices], rotation=45, ha='right')
                    else:
                        axes[1].set_xticks(range(len(week_labels)))
                        axes[1].set_xticklabels(week_labels, rotation=45, ha='right')
                    
                    # 피크 포인트 표시
                    if len(weekly_counts) > 0:
                        max_idx = weekly_counts.values.argmax()
                        max_value = weekly_counts.values[max_idx]
                        axes[1].annotate(f'최대: {max_value:,}개', 
                                           xy=(max_idx, max_value), 
                                           xytext=(max_idx, max_value + max_value * 0.1),
                                           arrowprops=dict(arrowstyle='->', color='red', lw=2),
                                           fontsize=10, ha='center', color='red', fontweight='bold')
                else:
                    # 일별 데이터가 적으면 그대로 표시
                    daily_labels = [date.strftime('%m-%d') for date in daily_counts.index]
                    
                    axes[1].plot(range(len(daily_counts)), daily_counts.values, 
                                   marker='o', markersize=6, linewidth=2, color='green', alpha=0.8)
                    axes[1].set_title('실제 데이터 분포 (일별)', fontsize=14, fontweight='bold')
                    axes[1].set_xlabel('월-일', fontsize=12)
                    
                    # x축 라벨 설정
                    if len(daily_labels) > 15:
                        sample_indices = np.linspace(0, len(daily_labels)-1, 15, dtype=int)
                        axes[1].set_xticks(sample_indices)
                        axes[1].set_xticklabels([daily_labels[i] for i in sample_indices], rotation=45, ha='right')
                    else:
                        axes[1].set_xticks(range(len(daily_labels)))
                        axes[1].set_xticklabels(daily_labels, rotation=45, ha='right')
                    
                    # 피크 포인트 표시
                    if len(daily_counts) > 0:
                        max_idx = daily_counts.values.argmax()
                        max_value = daily_counts.values[max_idx]
                        axes[1].annotate(f'최대: {max_value:,}개', 
                                           xy=(max_idx, max_value), 
                                           xytext=(max_idx, max_value + max_value * 0.1),
                                           arrowprops=dict(arrowstyle='->', color='red', lw=2),
                                           fontsize=10, ha='center', color='red', fontweight='bold')
                
                axes[1].set_ylabel('댓글 수', fontsize=12)
                axes[1].grid(True, alpha=0.3)
            
            else:
                # 데이터가 없는 경우 기본 메시지 표시
                for ax in axes:
                    ax.text(0.5, 0.5, '데이터 없음', ha='center', va='center', 
                           transform=ax.transAxes, fontsize=14, color='gray')
                    ax.set_title('데이터 없음', fontsize=14)
            
            # Time Interval 정보를 별도 텍스트 파일로 저장
            self._save_time_interval_info(distribution_result, df, date_column, save_path)
            
            plt.tight_layout()
            
            # 저장
            if not save_path:
                save_path = os.path.join(self.config.OUTPUT_STRUCTURE['visualizations'], 'time_distribution_analysis.png')
            
            # 디렉토리 생성
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            self.logger.info(f"📊 시간 분포 시각화 저장: {save_path}")
            self.logger.info(f"🔧 최소 시간 단위 '{self.min_time_unit}'에 따른 강제 표기 적용")
            return save_path
            
        except Exception as e:
            self.logger.error(f"❌ 시간 분포 시각화 실패: {str(e)}")
            import traceback
            self.logger.error(f"상세 오류: {traceback.format_exc()}")
            return None
    
    def _save_time_interval_info(self, distribution_result: Dict, df: pd.DataFrame = None, 
                                date_column: str = 'date', save_path: str = None):
        """
        Time Interval 정보를 별도 텍스트 파일로 저장
        Args:
            distribution_result: analyze_time_distribution 결과
            df: 원본 데이터프레임
            date_column: 날짜 컬럼명
            save_path: 이미지 저장 경로 (텍스트 파일 경로 생성용)
        """
        try:
            # 텍스트 파일 경로 생성
            if save_path:
                text_path = save_path.replace('.png', '_time_interval_info.txt')
            else:
                text_path = os.path.join(self.config.OUTPUT_STRUCTURE['visualizations'], 'time_distribution_interval_info.txt')
            
            # 디렉토리 생성
            os.makedirs(os.path.dirname(text_path), exist_ok=True)
            
            # Time Interval 정보 생성
            interval_info_lines = []
            interval_info_lines.append("=" * 80)
            interval_info_lines.append("시간 분포 분석 - Time Interval 정보")
            interval_info_lines.append("=" * 80)
            interval_info_lines.append(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            interval_info_lines.append("")
            
            # 1. 기본 권장사항 정보
            optimal_unit = distribution_result.get('optimal_time_unit', 'monthly')
            recommendation = distribution_result.get('recommendation', {})
            
            interval_info_lines.append("📊 권장 분석 단위 정보")
            interval_info_lines.append("-" * 40)
            interval_info_lines.append(f"권장 분석 단위: {optimal_unit}")
            
            if recommendation:
                interval_info_lines.append(f"분석 기간 수: {recommendation.get('total_periods', 'N/A')}개")
                avg_comments = recommendation.get('avg_comments_per_period', 'N/A')
                if isinstance(avg_comments, (int, float)):
                    interval_info_lines.append(f"평균 댓글/기간: {avg_comments:,.0f}개")
                else:
                    interval_info_lines.append(f"평균 댓글/기간: {avg_comments}개")
                interval_info_lines.append(f"데이터 품질: {recommendation.get('data_quality', 'N/A')}")
                
                # 권장사항 이유
                if 'reasons' in recommendation:
                    interval_info_lines.append("\n권장 이유:")
                    for reason in recommendation['reasons']:
                        interval_info_lines.append(f"  • {reason}")
                
                # 경고사항
                if 'warnings' in recommendation:
                    interval_info_lines.append("\n주의사항:")
                    for warning in recommendation['warnings']:
                        interval_info_lines.append(f"  ⚠️ {warning}")
                
                # 제안사항
                if 'suggestions' in recommendation:
                    interval_info_lines.append("\n제안사항:")
                    for suggestion in recommendation['suggestions']:
                        interval_info_lines.append(f"  💡 {suggestion}")
            
            interval_info_lines.append("")
            
            # 2. 상세 Time Interval 통계 (원본 데이터가 있는 경우)
            if df is not None and date_column in df.columns:
                interval_info_lines.append("📅 상세 Time Interval 통계")
                interval_info_lines.append("-" * 40)
                
                # 최적 단위에 따른 실제 그룹 생성
                df_detail = df.copy()
                df_detail[date_column] = pd.to_datetime(df_detail[date_column])
                
                if optimal_unit in ['monthly', 'monthly_forced']:
                    df_detail['time_group'] = df_detail[date_column].dt.to_period('M')
                elif optimal_unit == 'weekly':
                    df_detail['time_group'] = df_detail[date_column].dt.to_period('W')
                else:  # hybrid
                    df_detail['time_group'] = df_detail[date_column].dt.to_period('M')
                
                # 실제 time interval별 통계
                if 'comment_text' in df_detail.columns:
                    time_interval_stats = df_detail.groupby('time_group').agg({
                        'comment_text': 'count',
                        date_column: ['min', 'max']
                    }).round(0)
                    time_interval_stats.columns = ['comment_count', 'start_date', 'end_date']
                else:
                    # comment_text 컬럼이 없는 경우 첫 번째 컬럼 사용
                    first_col = df_detail.columns[0]
                    time_interval_stats = df_detail.groupby('time_group').agg({
                        first_col: 'count',
                        date_column: ['min', 'max']
                    }).round(0)
                    time_interval_stats.columns = ['comment_count', 'start_date', 'end_date']
                
                # 전체 통계 요약
                interval_info_lines.append(f"총 분석 기간 수: {len(time_interval_stats)}개")
                interval_info_lines.append(f"평균 댓글/기간: {time_interval_stats['comment_count'].mean():.0f}개")
                interval_info_lines.append(f"중앙값 댓글/기간: {time_interval_stats['comment_count'].median():.0f}개")
                interval_info_lines.append(f"최대 댓글/기간: {time_interval_stats['comment_count'].max():,}개")
                interval_info_lines.append(f"최소 댓글/기간: {time_interval_stats['comment_count'].min():,}개")
                
                # 데이터 품질 평가
                cv = time_interval_stats['comment_count'].std() / time_interval_stats['comment_count'].mean()
                if cv < 0.5:
                    quality = "매우 좋음"
                elif cv < 1.0:
                    quality = "좋음"
                elif cv < 1.5:
                    quality = "보통"
                else:
                    quality = "불균등"
                
                interval_info_lines.append(f"데이터 품질 평가: {quality}")
                interval_info_lines.append(f"변동계수(CV): {cv:.2f}")
                
                # 시간 범위 정보
                total_start = pd.to_datetime(time_interval_stats['start_date'].min()).strftime('%Y-%m-%d')
                total_end = pd.to_datetime(time_interval_stats['end_date'].max()).strftime('%Y-%m-%d')
                total_days = (pd.to_datetime(time_interval_stats['end_date'].max()) - 
                             pd.to_datetime(time_interval_stats['start_date'].min())).days + 1
                
                interval_info_lines.append(f"\n전체 분석 기간:")
                interval_info_lines.append(f"  시작: {total_start}")
                interval_info_lines.append(f"  종료: {total_end}")
                interval_info_lines.append(f"  총 일수: {total_days:,}일")
                
                # 모든 Time Interval 상세 정보
                interval_info_lines.append(f"\n📋 전체 Time Interval 상세 정보 ({len(time_interval_stats)}개)")
                interval_info_lines.append("-" * 60)
                interval_info_lines.append(f"{'순위':<4} {'기간':<20} {'시작일':<12} {'종료일':<12} {'댓글수':<10}")
                interval_info_lines.append("-" * 60)
                
                # 댓글 수 기준으로 정렬
                sorted_intervals = time_interval_stats.sort_values('comment_count', ascending=False)
                
                for i, (period, row) in enumerate(sorted_intervals.iterrows(), 1):
                    start_date = pd.to_datetime(row['start_date']).strftime('%Y-%m-%d')
                    end_date = pd.to_datetime(row['end_date']).strftime('%Y-%m-%d')
                    comment_count = int(row['comment_count'])
                    
                    period_str = str(period)
                    if len(period_str) > 18:
                        period_str = period_str[:15] + "..."
                    
                    interval_info_lines.append(f"{i:<4} {period_str:<20} {start_date:<12} {end_date:<12} {comment_count:,}")
                
            else:
                interval_info_lines.append("📅 상세 Time Interval 통계")
                interval_info_lines.append("-" * 40)
                interval_info_lines.append("⚠️ 상세 정보를 위해서는 원본 데이터가 필요합니다.")
            
            interval_info_lines.append("")
            interval_info_lines.append("=" * 80)
            interval_info_lines.append("분석 완료")
            interval_info_lines.append("=" * 80)
            
            # 텍스트 파일로 저장
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(interval_info_lines))
            
            self.logger.info(f"📄 Time Interval 정보 저장: {text_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Time Interval 정보 저장 실패: {str(e)}")
            import traceback
            self.logger.error(f"상세 오류: {traceback.format_exc()}") 