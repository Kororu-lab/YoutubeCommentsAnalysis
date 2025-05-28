"""
Sentiment Analyzer Module
감성 분석 모듈
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class SentimentAnalyzer:
    """감성 분석 클래스"""
    
    def __init__(self, config):
        """
        초기화
        Args:
            config: AnalysisConfig 객체
        """
        self.config = config
        self.logger = self._setup_logger()
        self.device = config.DEVICE
        
        # 모델 초기화
        self.binary_model = None
        self.emotion_model = None
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
        """감성 분석 모델 초기화"""
        try:
            self.logger.info("🤖 감성 분석 모델 초기화 시작")
            
            # 이진 감성 분석 모델 (긍정/부정)
            binary_config = self.config.SENTIMENT_MODELS['binary']
            self.logger.info(f"📥 이진 감성 모델 로드: {binary_config['model_name']}")
            
            self.binary_model = pipeline(
                "sentiment-analysis",
                model=binary_config['model_name'],
                device=0 if self.device.type == 'cuda' else -1,
                return_all_scores=True
            )
            
            # 6가지 감정 분석 모델
            emotion_config = self.config.SENTIMENT_MODELS['emotion_6']
            self.logger.info(f"📥 감정 분석 모델 로드: {emotion_config['model_name']}")
            
            try:
                self.emotion_model = pipeline(
                    "text-classification",
                    model=emotion_config['model_name'],
                    device=0 if self.device.type == 'cuda' else -1,
                    return_all_scores=True
                )
            except Exception as e:
                self.logger.warning(f"⚠️ 6감정 모델 로드 실패, 대체 모델 사용: {str(e)}")
                # 대체 모델 사용
                self.emotion_model = pipeline(
                    "sentiment-analysis",
                    model="monologg/koelectra-base-v3-goemotions",
                    device=0 if self.device.type == 'cuda' else -1,
                    return_all_scores=True
                )
            
            self.logger.info("✅ 감성 분석 모델 초기화 완료")
            
        except Exception as e:
            self.logger.error(f"❌ 감성 분석 모델 초기화 실패: {str(e)}")
            raise
    
    def analyze_binary_sentiment(self, texts: List[str], batch_size: int = 32) -> List[Dict]:
        """
        이진 감성 분석 (긍정/부정)
        Args:
            texts: 분석할 텍스트 리스트
            batch_size: 배치 크기
        Returns:
            감성 분석 결과 리스트
        """
        try:
            self.logger.info(f"😊 이진 감성 분석 시작: {len(texts):,}개 텍스트")
            
            results = []
            
            # 배치 단위로 처리
            for i in tqdm(range(0, len(texts), batch_size), desc="이진 감성 분석"):
                batch_texts = texts[i:i+batch_size]
                
                # 빈 텍스트 필터링
                valid_texts = [text for text in batch_texts if text and len(text.strip()) > 0]
                
                if not valid_texts:
                    # 빈 배치인 경우 기본값 추가
                    for _ in batch_texts:
                        results.append({
                            'label': '중립',
                            'score': 0.5,
                            'positive_score': 0.5,
                            'negative_score': 0.5
                        })
                    continue
                
                try:
                    # 모델 예측
                    batch_results = self.binary_model(valid_texts)
                    
                    # 결과 처리 - valid_texts와 batch_results는 1:1 대응
                    valid_idx = 0
                    for j, text in enumerate(batch_texts):
                        if text and len(text.strip()) > 0:
                            result = batch_results[valid_idx]
                            valid_idx += 1
                            
                            # 점수 정리
                            if isinstance(result, list):
                                scores = {item['label']: item['score'] for item in result}
                                positive_score = scores.get('positive', scores.get('POSITIVE', scores.get('긍정', 0.0)))
                                negative_score = scores.get('negative', scores.get('NEGATIVE', scores.get('부정', 0.0)))
                                
                                # 점수가 없는 경우 기본값 설정
                                if positive_score == 0.0 and negative_score == 0.0:
                                    positive_score = 0.5
                                    negative_score = 0.5
                            else:
                                if result['label'] in ['positive', 'POSITIVE', '긍정']:
                                    positive_score = result['score']
                                    negative_score = 1 - result['score']
                                else:
                                    negative_score = result['score']
                                    positive_score = 1 - result['score']
                            
                            # 최종 라벨 결정
                            if positive_score > negative_score:
                                label = '긍정'
                                score = positive_score
                            else:
                                label = '부정'
                                score = negative_score
                            
                            results.append({
                                'label': label,
                                'score': score,
                                'positive_score': positive_score,
                                'negative_score': negative_score
                            })
                        else:
                            # 빈 텍스트인 경우
                            results.append({
                                'label': '중립',
                                'score': 0.5,
                                'positive_score': 0.5,
                                'negative_score': 0.5
                            })
                
                except Exception as e:
                    self.logger.warning(f"⚠️ 배치 처리 실패: {str(e)}")
                    # 실패한 배치는 중립으로 처리
                    for _ in batch_texts:
                        results.append({
                            'label': '중립',
                            'score': 0.5,
                            'positive_score': 0.5,
                            'negative_score': 0.5
                        })
            
            self.logger.info("✅ 이진 감성 분석 완료")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ 이진 감성 분석 실패: {str(e)}")
            raise
    
    def analyze_emotion_sentiment(self, texts: List[str], batch_size: int = 32) -> List[Dict]:
        """
        6가지 감정 분석
        Args:
            texts: 분석할 텍스트 리스트
            batch_size: 배치 크기
        Returns:
            감정 분석 결과 리스트
        """
        try:
            self.logger.info(f"😍 6감정 분석 시작: {len(texts):,}개 텍스트")
            
            results = []
            emotion_labels = self.config.SENTIMENT_MODELS['emotion_6']['labels']
            
            # 배치 단위로 처리
            for i in tqdm(range(0, len(texts), batch_size), desc="6감정 분석"):
                batch_texts = texts[i:i+batch_size]
                
                # 빈 텍스트 필터링
                valid_texts = [text for text in batch_texts if text and len(text.strip()) > 0]
                
                if not valid_texts:
                    # 빈 배치인 경우 기본값 추가
                    for _ in batch_texts:
                        emotion_scores = {emotion: 1.0/len(emotion_labels) for emotion in emotion_labels}
                        results.append({
                            'dominant_emotion': '중립',
                            'emotion_scores': emotion_scores,
                            'confidence': 1.0/len(emotion_labels)
                        })
                    continue
                
                try:
                    # 모델 예측
                    batch_results = self.emotion_model(valid_texts)
                    
                    # 결과 처리 - valid_texts와 batch_results는 1:1 대응
                    valid_idx = 0
                    for j, text in enumerate(batch_texts):
                        if text and len(text.strip()) > 0:
                            result = batch_results[valid_idx]
                            valid_idx += 1
                            
                            if isinstance(result, list):
                                # 모든 감정 점수를 6가지 기본 감정으로 매핑
                                emotion_scores = {emotion: 0.0 for emotion in emotion_labels}
                                
                                for item in result:
                                    label = item['label']
                                    score = item['score']
                                    
                                    # 세분화된 감정을 6가지 기본 감정으로 매핑 (정정된 그룹 기준)
                                    # 분노 그룹
                                    if label in ['툴툴대는', '좌절한', '짜증나는', '방어적인', '악의적인', '안달하는', '구역질 나는', '노여워하는', '성가신', '분노']:
                                        emotion_scores['분노'] += score
                                    # 슬픔 그룹
                                    elif label in ['실망한', '비통한', '후회되는', '우울한', '마비된', '염세적인', '눈물이 나는', '낙담한', '환멸을 느끼는', '슬픔']:
                                        emotion_scores['슬픔'] += score
                                    # 불안 그룹
                                    elif label in ['두려운', '스트레스 받는', '취약한', '혼란스러운', '당혹스러운', '회의적인', '걱정스러운', '조심스러운', '초조한', '불안']:
                                        emotion_scores['불안'] += score
                                    # 상처 그룹
                                    elif label in ['질투하는', '배신당한', '고립된', '충격 받은', '불우한', '희생된', '억울한', '괴로워하는', '버려진', '상처']:
                                        emotion_scores['상처'] += score
                                    # 기쁨 그룹
                                    elif label in ['감사하는', '사랑하는', '편안한', '만족스러운', '흥분되는', '느긋한', '안도하는', '신이 난', '자신하는', '기쁨']:
                                        emotion_scores['기쁨'] += score
                                    # 당황 그룹 (나머지 모든 감정)
                                    else:
                                        emotion_scores['당황'] += score
                                
                                # 정규화
                                total_score = sum(emotion_scores.values())
                                if total_score > 0:
                                    emotion_scores = {k: v/total_score for k, v in emotion_scores.items()}
                                else:
                                    # 모든 점수가 0인 경우 균등 분배
                                    emotion_scores = {emotion: 1.0/len(emotion_labels) for emotion in emotion_labels}
                                
                                # 주요 감정 결정
                                dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])
                                
                                results.append({
                                    'dominant_emotion': dominant_emotion[0],
                                    'emotion_scores': emotion_scores,
                                    'confidence': dominant_emotion[1]
                                })
                            else:
                                # 단일 결과인 경우
                                emotion_scores = {emotion: 1.0/len(emotion_labels) for emotion in emotion_labels}
                                results.append({
                                    'dominant_emotion': '당황',
                                    'emotion_scores': emotion_scores,
                                    'confidence': 1.0/len(emotion_labels)
                                })
                        else:
                            # 빈 텍스트인 경우
                            emotion_scores = {emotion: 1.0/len(emotion_labels) for emotion in emotion_labels}
                            results.append({
                                'dominant_emotion': '중립',
                                'emotion_scores': emotion_scores,
                                'confidence': 1.0/len(emotion_labels)
                            })
                
                except Exception as e:
                    self.logger.warning(f"⚠️ 감정 분석 배치 처리 실패: {str(e)}")
                    # 실패한 배치는 균등 분포로 처리
                    for _ in batch_texts:
                        emotion_scores = {emotion: 1.0/len(emotion_labels) for emotion in emotion_labels}
                        results.append({
                            'dominant_emotion': '중립',
                            'emotion_scores': emotion_scores,
                            'confidence': 1.0/len(emotion_labels)
                        })
            
            self.logger.info("✅ 6감정 분석 완료")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ 6감정 분석 실패: {str(e)}")
            raise
    
    def analyze_monthly_sentiment(self, monthly_data: Dict[str, pd.DataFrame], target_name: str) -> Dict:
        """
        월별 감성 분석
        Args:
            monthly_data: 월별 데이터 딕셔너리
            target_name: 분석 대상 이름
        Returns:
            월별 감성 분석 결과
        """
        try:
            self.logger.info(f"📅 {target_name} 월별 감성 분석 시작")
            
            monthly_results = {}
            
            for year_month, df in monthly_data.items():
                self.logger.info(f"📊 {year_month} 감성 분석 중... ({len(df):,}개 댓글)")
                
                if len(df) == 0:
                    continue
                
                texts = df['cleaned_text'].tolist()
                
                # 이진 감성 분석
                binary_results = self.analyze_binary_sentiment(texts)
                
                # 6감정 분석
                emotion_results = self.analyze_emotion_sentiment(texts)
                
                # 결과 집계
                binary_summary = self._summarize_binary_results(binary_results)
                emotion_summary = self._summarize_emotion_results(emotion_results)
                
                monthly_results[year_month] = {
                    'total_comments': len(df),
                    'binary_sentiment': binary_summary,
                    'emotion_sentiment': emotion_summary,
                    'detailed_results': {
                        'binary': binary_results,
                        'emotion': emotion_results
                    }
                }
                
                self.logger.info(f"✅ {year_month} 완료 - 긍정: {binary_summary['positive_ratio']:.1%}, "
                               f"주요 감정: {emotion_summary['dominant_emotion']}")
            
            self.results[target_name] = monthly_results
            
            self.logger.info(f"✅ {target_name} 월별 감성 분석 완료")
            return monthly_results
            
        except Exception as e:
            self.logger.error(f"❌ {target_name} 월별 감성 분석 실패: {str(e)}")
            raise
    
    def analyze_time_grouped_sentiment(self, time_groups: Dict[str, pd.DataFrame], target_name: str) -> Dict:
        """
        적응적 시간 그룹별 감성 분석
        Args:
            time_groups: 시간 그룹별 데이터 딕셔너리
            target_name: 분석 대상 이름
        Returns:
            시간 그룹별 감성 분석 결과
        """
        try:
            self.logger.info(f"📅 {target_name} 시간 그룹별 감성 분석 시작")
            
            time_group_results = {}
            
            for group_name, df in time_groups.items():
                self.logger.info(f"📊 {group_name} 감성 분석 중... ({len(df):,}개 댓글)")
                
                if len(df) == 0:
                    continue
                
                texts = df['cleaned_text'].tolist()
                
                # 이진 감성 분석
                binary_results = self.analyze_binary_sentiment(texts)
                
                # 6감정 분석
                emotion_results = self.analyze_emotion_sentiment(texts)
                
                # 결과 집계
                binary_summary = self._summarize_binary_results(binary_results)
                emotion_summary = self._summarize_emotion_results(emotion_results)
                
                time_group_results[group_name] = {
                    'total_comments': len(df),
                    'binary_sentiment': binary_summary,
                    'emotion_sentiment': emotion_summary,
                    'detailed_results': {
                        'binary': binary_results,
                        'emotion': emotion_results
                    }
                }
                
                self.logger.info(f"✅ {group_name} 완료 - 긍정: {binary_summary['positive_ratio']:.1%}, "
                               f"주요 감정: {emotion_summary['dominant_emotion']}")
            
            self.results[target_name] = time_group_results
            
            self.logger.info(f"✅ {target_name} 시간 그룹별 감성 분석 완료")
            return time_group_results
            
        except Exception as e:
            self.logger.error(f"❌ {target_name} 시간 그룹별 감성 분석 실패: {str(e)}")
            raise
    
    def _summarize_binary_results(self, results: List[Dict]) -> Dict:
        """이진 감성 분석 결과 요약"""
        if not results:
            return {
                'positive_count': 0,
                'negative_count': 0,
                'positive_ratio': 0.0,
                'negative_ratio': 0.0,
                'avg_positive_score': 0.0,
                'avg_negative_score': 0.0
            }
        
        positive_count = sum(1 for r in results if r['label'] == '긍정')
        negative_count = sum(1 for r in results if r['label'] == '부정')
        total_count = len(results)
        
        avg_positive_score = np.mean([r['positive_score'] for r in results])
        avg_negative_score = np.mean([r['negative_score'] for r in results])
        
        return {
            'positive_count': positive_count,
            'negative_count': negative_count,
            'positive_ratio': positive_count / total_count if total_count > 0 else 0.0,
            'negative_ratio': negative_count / total_count if total_count > 0 else 0.0,
            'avg_positive_score': avg_positive_score,
            'avg_negative_score': avg_negative_score
        }
    
    def _summarize_emotion_results(self, results: List[Dict]) -> Dict:
        """6감정 분석 결과 요약"""
        if not results:
            emotion_labels = self.config.SENTIMENT_MODELS['emotion_6']['labels']
            return {
                'emotion_distribution': {emotion: 0.0 for emotion in emotion_labels},
                'dominant_emotion': '중립',
                'avg_confidence': 0.0
            }
        
        emotion_labels = self.config.SENTIMENT_MODELS['emotion_6']['labels']
        emotion_counts = {emotion: 0 for emotion in emotion_labels}
        
        # 감정별 카운트
        for result in results:
            emotion = result['dominant_emotion']
            if emotion in emotion_counts:
                emotion_counts[emotion] += 1
        
        total_count = len(results)
        emotion_distribution = {emotion: count/total_count for emotion, count in emotion_counts.items()}
        
        # 주요 감정
        dominant_emotion = max(emotion_distribution.items(), key=lambda x: x[1])[0]
        
        # 평균 신뢰도
        avg_confidence = np.mean([r['confidence'] for r in results])
        
        return {
            'emotion_distribution': emotion_distribution,
            'dominant_emotion': dominant_emotion,
            'avg_confidence': avg_confidence
        }
    
    def get_sentiment_trends(self, target_name: str) -> Dict:
        """
        감성 트렌드 분석
        Args:
            target_name: 분석 대상 이름
        Returns:
            트렌드 분석 결과
        """
        try:
            if target_name not in self.results:
                raise ValueError(f"{target_name}의 감성 분석 결과가 없습니다.")
            
            monthly_results = self.results[target_name]
            
            # 시간순 정렬
            sorted_months = sorted(monthly_results.keys())
            
            # 트렌드 데이터 추출
            trends = {
                'months': sorted_months,
                'positive_ratios': [],
                'negative_ratios': [],
                'emotion_trends': {emotion: [] for emotion in self.config.SENTIMENT_MODELS['emotion_6']['labels']}
            }
            
            for month in sorted_months:
                result = monthly_results[month]
                
                # 이진 감성 트렌드
                trends['positive_ratios'].append(result['binary_sentiment']['positive_ratio'])
                trends['negative_ratios'].append(result['binary_sentiment']['negative_ratio'])
                
                # 감정 트렌드
                emotion_dist = result['emotion_sentiment']['emotion_distribution']
                for emotion in trends['emotion_trends']:
                    trends['emotion_trends'][emotion].append(emotion_dist.get(emotion, 0.0))
            
            return trends
            
        except Exception as e:
            self.logger.error(f"❌ {target_name} 감성 트렌드 분석 실패: {str(e)}")
            raise
    
    def analyze_sentiment_context(self, texts: List[str], timestamps: List[str] = None) -> Dict:
        """
        맥락 기반 감성 분석
        Args:
            texts: 분석할 텍스트 리스트
            timestamps: 타임스탬프 리스트 (선택사항)
        Returns:
            맥락 정보가 포함된 감성 분석 결과
        """
        try:
            self.logger.info(f"🔍 맥락 기반 감성 분석 시작: {len(texts):,}개 텍스트")
            
            # 기본 감성 분석
            binary_results = self.analyze_binary_sentiment(texts)
            emotion_results = self.analyze_emotion_sentiment(texts)
            
            # 감정 변화 패턴 분석
            emotion_patterns = self._analyze_emotion_patterns(emotion_results['emotions'])
            
            # 감정 강도 분석
            emotion_intensity = self._analyze_emotion_intensity(texts, emotion_results['emotions'])
            
            # 감정 트리거 키워드 분석
            emotion_triggers = self._analyze_emotion_triggers(texts, emotion_results['emotions'])
            
            # 시간별 감정 변화 (타임스탬프가 있는 경우)
            temporal_analysis = {}
            if timestamps:
                temporal_analysis = self._analyze_temporal_emotions(emotion_results['emotions'], timestamps)
            
            result = {
                'binary_sentiment': binary_results,
                'emotion_sentiment': emotion_results,
                'emotion_patterns': emotion_patterns,
                'emotion_intensity': emotion_intensity,
                'emotion_triggers': emotion_triggers,
                'temporal_analysis': temporal_analysis,
                'context_summary': self._generate_context_summary(binary_results, emotion_results, emotion_patterns)
            }
            
            self.logger.info("✅ 맥락 기반 감성 분석 완료")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ 맥락 기반 감성 분석 실패: {str(e)}")
            raise
    
    def _analyze_emotion_patterns(self, emotions: List[str]) -> Dict:
        """
        감정 패턴 분석
        Args:
            emotions: 감정 리스트
        Returns:
            감정 패턴 분석 결과
        """
        try:
            from collections import Counter
            
            emotion_counter = Counter(emotions)
            total_emotions = len(emotions)
            
            # 감정 분포
            emotion_distribution = {emotion: count/total_emotions 
                                  for emotion, count in emotion_counter.items()}
            
            # 주요 감정 (10% 이상)
            major_emotions = {emotion: ratio for emotion, ratio in emotion_distribution.items() 
                            if ratio >= 0.1}
            
            # 감정 다양성 (엔트로피 계산)
            import math
            entropy = -sum(ratio * math.log2(ratio) for ratio in emotion_distribution.values() if ratio > 0)
            diversity_score = entropy / math.log2(len(emotion_distribution)) if len(emotion_distribution) > 1 else 0
            
            # 감정 극성 분석
            negative_emotions = {'분노', '슬픔', '불안', '상처', '당황'}
            positive_emotions = {'기쁨'}
            
            negative_ratio = sum(emotion_distribution.get(emotion, 0) for emotion in negative_emotions)
            positive_ratio = sum(emotion_distribution.get(emotion, 0) for emotion in positive_emotions)
            
            return {
                'emotion_distribution': emotion_distribution,
                'major_emotions': major_emotions,
                'diversity_score': diversity_score,
                'negative_ratio': negative_ratio,
                'positive_ratio': positive_ratio,
                'dominant_emotion': max(emotion_counter, key=emotion_counter.get) if emotion_counter else '없음'
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ 감정 패턴 분석 실패: {str(e)}")
            return {}
    
    def _analyze_emotion_intensity(self, texts: List[str], emotions: List[str]) -> Dict:
        """
        감정 강도 분석
        Args:
            texts: 텍스트 리스트
            emotions: 감정 리스트
        Returns:
            감정 강도 분석 결과
        """
        try:
            # 감정별 강도 키워드
            intensity_keywords = {
                '분노': ['진짜', '완전', '너무', '정말', '엄청', '미치겠다', '화나다', '빡치다'],
                '슬픔': ['너무', '정말', '진짜', '완전', '엄청', '슬프다', '우울하다'],
                '불안': ['걱정', '불안', '두렵다', '무섭다', '떨린다'],
                '상처': ['아프다', '힘들다', '괴롭다', '서럽다'],
                '당황': ['어떻게', '왜', '뭐지', '헐', '대박'],
                '기쁨': ['좋다', '행복', '기쁘다', '최고', '짱']
            }
            
            emotion_intensities = {}
            
            for emotion in set(emotions):
                if emotion == '없음':
                    continue
                    
                emotion_texts = [text for text, emo in zip(texts, emotions) if emo == emotion]
                
                if not emotion_texts:
                    continue
                
                # 해당 감정의 강도 키워드 빈도 계산
                intensity_scores = []
                keywords = intensity_keywords.get(emotion, [])
                
                for text in emotion_texts:
                    text_lower = text.lower()
                    intensity_count = sum(1 for keyword in keywords if keyword in text_lower)
                    # 텍스트 길이 대비 강도 점수
                    intensity_score = intensity_count / max(len(text.split()), 1)
                    intensity_scores.append(intensity_score)
                
                avg_intensity = sum(intensity_scores) / len(intensity_scores) if intensity_scores else 0
                
                emotion_intensities[emotion] = {
                    'average_intensity': avg_intensity,
                    'high_intensity_ratio': sum(1 for score in intensity_scores if score > 0.1) / len(intensity_scores),
                    'sample_count': len(emotion_texts)
                }
            
            return emotion_intensities
            
        except Exception as e:
            self.logger.warning(f"⚠️ 감정 강도 분석 실패: {str(e)}")
            return {}
    
    def _analyze_emotion_triggers(self, texts: List[str], emotions: List[str]) -> Dict:
        """
        감정 트리거 키워드 분석
        Args:
            texts: 텍스트 리스트
            emotions: 감정 리스트
        Returns:
            감정별 트리거 키워드
        """
        try:
            from collections import Counter
            import re
            
            emotion_triggers = {}
            
            for emotion in set(emotions):
                if emotion == '없음':
                    continue
                
                emotion_texts = [text for text, emo in zip(texts, emotions) if emo == emotion]
                
                if not emotion_texts:
                    continue
                
                # 텍스트에서 키워드 추출
                all_words = []
                for text in emotion_texts:
                    # 한글, 영문, 숫자만 추출
                    words = re.findall(r'[가-힣a-zA-Z0-9]+', text)
                    # 2글자 이상 단어만
                    words = [word for word in words if len(word) >= 2]
                    all_words.extend(words)
                
                # 빈도 계산
                word_counter = Counter(all_words)
                
                # 상위 키워드 (최소 2번 이상 등장)
                top_keywords = [(word, count) for word, count in word_counter.most_common(10) 
                              if count >= 2]
                
                emotion_triggers[emotion] = top_keywords
            
            return emotion_triggers
            
        except Exception as e:
            self.logger.warning(f"⚠️ 감정 트리거 분석 실패: {str(e)}")
            return {}
    
    def _analyze_temporal_emotions(self, emotions: List[str], timestamps: List[str]) -> Dict:
        """
        시간별 감정 변화 분석
        Args:
            emotions: 감정 리스트
            timestamps: 타임스탬프 리스트
        Returns:
            시간별 감정 변화 분석 결과
        """
        try:
            import pandas as pd
            from datetime import datetime
            
            # 데이터프레임 생성
            df = pd.DataFrame({
                'emotion': emotions,
                'timestamp': timestamps
            })
            
            # 타임스탬프 파싱
            df['datetime'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df = df.dropna(subset=['datetime'])
            
            if len(df) == 0:
                return {}
            
            # 시간대별 감정 분포
            df['hour'] = df['datetime'].dt.hour
            hourly_emotions = df.groupby(['hour', 'emotion']).size().unstack(fill_value=0)
            
            # 일별 감정 변화
            df['date'] = df['datetime'].dt.date
            daily_emotions = df.groupby(['date', 'emotion']).size().unstack(fill_value=0)
            
            return {
                'hourly_distribution': hourly_emotions.to_dict() if not hourly_emotions.empty else {},
                'daily_trends': daily_emotions.to_dict() if not daily_emotions.empty else {},
                'peak_emotion_hours': self._find_peak_emotion_hours(hourly_emotions),
                'emotion_volatility': self._calculate_emotion_volatility(daily_emotions)
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ 시간별 감정 분석 실패: {str(e)}")
            return {}
    
    def _find_peak_emotion_hours(self, hourly_emotions) -> Dict:
        """감정별 피크 시간대 찾기"""
        try:
            peak_hours = {}
            for emotion in hourly_emotions.columns:
                peak_hour = hourly_emotions[emotion].idxmax()
                peak_count = hourly_emotions[emotion].max()
                peak_hours[emotion] = {'hour': peak_hour, 'count': peak_count}
            return peak_hours
        except:
            return {}
    
    def _calculate_emotion_volatility(self, daily_emotions) -> Dict:
        """감정 변동성 계산"""
        try:
            volatility = {}
            for emotion in daily_emotions.columns:
                emotion_series = daily_emotions[emotion]
                if len(emotion_series) > 1:
                    volatility[emotion] = emotion_series.std() / emotion_series.mean() if emotion_series.mean() > 0 else 0
                else:
                    volatility[emotion] = 0
            return volatility
        except:
            return {}
    
    def _generate_context_summary(self, binary_results: Dict, emotion_results: Dict, emotion_patterns: Dict) -> str:
        """
        맥락 요약 생성
        Args:
            binary_results: 이진 감성 결과
            emotion_results: 감정 분석 결과
            emotion_patterns: 감정 패턴 결과
        Returns:
            맥락 요약 텍스트
        """
        try:
            summary_parts = []
            
            # 전체 감성 요약
            pos_ratio = binary_results.get('positive_ratio', 0) * 100
            neg_ratio = binary_results.get('negative_ratio', 0) * 100
            summary_parts.append(f"전체 감성: 긍정 {pos_ratio:.1f}%, 부정 {neg_ratio:.1f}%")
            
            # 주요 감정
            dominant_emotion = emotion_patterns.get('dominant_emotion', '없음')
            summary_parts.append(f"주요 감정: {dominant_emotion}")
            
            # 감정 다양성
            diversity = emotion_patterns.get('diversity_score', 0)
            if diversity > 0.8:
                diversity_desc = "매우 다양한"
            elif diversity > 0.6:
                diversity_desc = "다양한"
            elif diversity > 0.4:
                diversity_desc = "보통의"
            else:
                diversity_desc = "단조로운"
            summary_parts.append(f"감정 다양성: {diversity_desc} ({diversity:.2f})")
            
            # 주요 감정들
            major_emotions = emotion_patterns.get('major_emotions', {})
            if major_emotions:
                major_list = [f"{emotion}({ratio*100:.1f}%)" for emotion, ratio in major_emotions.items()]
                summary_parts.append(f"주요 감정 분포: {', '.join(major_list)}")
            
            return " | ".join(summary_parts)
            
        except Exception as e:
            self.logger.warning(f"⚠️ 맥락 요약 생성 실패: {str(e)}")
            return "맥락 요약 생성 실패" 