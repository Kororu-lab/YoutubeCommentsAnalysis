"""
Report Generator Module
보고서 생성 모듈
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
import os
import json
from datetime import datetime
from jinja2 import Template

class ReportGenerator:
    """보고서 생성 클래스"""
    
    def __init__(self, config):
        """
        초기화
        Args:
            config: AnalysisConfig 객체
        """
        self.config = config
        self.logger = self._setup_logger()
        self.output_dir = config.OUTPUT_STRUCTURE['reports']
        
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
    
    def generate_comprehensive_report(self, all_results: Dict, target_name: str) -> str:
        """
        종합 분석 보고서 생성
        Args:
            all_results: 모든 분석 결과
            target_name: 분석 대상 이름
        Returns:
            생성된 보고서 파일 경로
        """
        try:
            self.logger.info(f"📋 {target_name} 종합 분석 보고서 생성 시작")
            
            # 보고서 내용 구성
            report_content = self._build_report_content(all_results, target_name)
            
            # HTML 보고서 생성
            html_report = self._generate_html_report(report_content, target_name)
            
            # 마크다운 보고서 생성
            md_report = self._generate_markdown_report(report_content, target_name)
            
            # JSON 요약 보고서 생성
            json_summary = self._generate_json_summary(all_results, target_name)
            
            self.logger.info(f"✅ {target_name} 종합 분석 보고서 생성 완료")
            return html_report
            
        except Exception as e:
            self.logger.error(f"❌ {target_name} 보고서 생성 실패: {str(e)}")
            raise
    
    def _build_report_content(self, all_results: Dict, target_name: str) -> Dict:
        """보고서 내용 구성"""
        try:
            # 기본 정보
            total_comments = sum(result.get('total_comments', 0) for result in all_results.values())
            analysis_period = {
                'start': min(all_results.keys()) if all_results else 'N/A',
                'end': max(all_results.keys()) if all_results else 'N/A',
                'total_months': len(all_results)
            }
            
            # 감성 분석 요약
            sentiment_summary = self._summarize_sentiment_analysis(all_results)
            
            # 토픽 분석 요약
            topic_summary = self._summarize_topic_analysis(all_results)
            
            # 시간적 변화 분석
            temporal_analysis = self._analyze_temporal_changes(all_results)
            
            # 주요 발견사항
            key_findings = self._extract_key_findings(all_results, target_name)
            
            # 연구 함의
            research_implications = self._generate_research_implications(all_results, target_name)
            
            return {
                'target_name': target_name,
                'analysis_date': datetime.now().strftime('%Y년 %m월 %d일'),
                'total_comments': total_comments,
                'analysis_period': analysis_period,
                'sentiment_summary': sentiment_summary,
                'topic_summary': topic_summary,
                'temporal_analysis': temporal_analysis,
                'key_findings': key_findings,
                'research_implications': research_implications
            }
            
        except Exception as e:
            self.logger.error(f"❌ 보고서 내용 구성 실패: {str(e)}")
            raise
    
    def _summarize_sentiment_analysis(self, all_results: Dict) -> Dict:
        """감성 분석 요약"""
        try:
            if not all_results:
                return {'error': '분석 결과가 없습니다.'}
            
            # 전체 기간 감성 통계
            total_positive = 0
            total_negative = 0
            total_comments = 0
            
            monthly_sentiments = []
            emotion_distributions = []
            
            for month, result in all_results.items():
                if 'binary_sentiment' in result:
                    binary_data = result['binary_sentiment']
                    pos_count = binary_data.get('positive_count', 0)
                    neg_count = binary_data.get('negative_count', 0)
                    
                    total_positive += pos_count
                    total_negative += neg_count
                    total_comments += result.get('total_comments', 0)
                    
                    monthly_sentiments.append({
                        'month': month,
                        'positive_ratio': binary_data.get('positive_ratio', 0),
                        'negative_ratio': binary_data.get('negative_ratio', 0)
                    })
                
                if 'emotion_sentiment' in result:
                    emotion_data = result['emotion_sentiment']
                    emotion_distributions.append({
                        'month': month,
                        'dominant_emotion': emotion_data.get('dominant_emotion', '없음'),
                        'distribution': emotion_data.get('emotion_distribution', {})
                    })
            
            # 전체 감성 비율
            overall_positive_ratio = total_positive / total_comments if total_comments > 0 else 0
            overall_negative_ratio = total_negative / total_comments if total_comments > 0 else 0
            
            # 감성 변화 트렌드
            if len(monthly_sentiments) > 1:
                first_month = monthly_sentiments[0]
                last_month = monthly_sentiments[-1]
                
                positive_change = last_month['positive_ratio'] - first_month['positive_ratio']
                negative_change = last_month['negative_ratio'] - first_month['negative_ratio']
                
                trend_analysis = {
                    'positive_trend': '증가' if positive_change > 0.05 else '감소' if positive_change < -0.05 else '안정',
                    'negative_trend': '증가' if negative_change > 0.05 else '감소' if negative_change < -0.05 else '안정',
                    'positive_change': positive_change,
                    'negative_change': negative_change
                }
            else:
                trend_analysis = {'message': '트렌드 분석을 위한 충분한 데이터가 없습니다.'}
            
            # 주요 감정 분석
            all_emotions = {}
            for emotion_data in emotion_distributions:
                for emotion, ratio in emotion_data['distribution'].items():
                    if emotion not in all_emotions:
                        all_emotions[emotion] = []
                    all_emotions[emotion].append(ratio)
            
            avg_emotions = {emotion: np.mean(ratios) for emotion, ratios in all_emotions.items()}
            dominant_overall_emotion = max(avg_emotions.items(), key=lambda x: x[1])[0] if avg_emotions else '없음'
            
            return {
                'overall_statistics': {
                    'total_comments': total_comments,
                    'positive_ratio': overall_positive_ratio,
                    'negative_ratio': overall_negative_ratio,
                    'dominant_emotion': dominant_overall_emotion
                },
                'trend_analysis': trend_analysis,
                'monthly_data': monthly_sentiments,
                'emotion_analysis': {
                    'average_distribution': avg_emotions,
                    'monthly_emotions': emotion_distributions
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ 감성 분석 요약 실패: {str(e)}")
            return {'error': str(e)}
    
    def _summarize_topic_analysis(self, all_results: Dict) -> Dict:
        """토픽 분석 요약"""
        try:
            if not all_results:
                return {'error': '분석 결과가 없습니다.'}
            
            bertopic_results = []
            lda_results = []
            
            for month, result in all_results.items():
                # BERTopic 결과
                if 'bertopic' in result and result['bertopic']:
                    bertopic_data = result['bertopic']
                    bertopic_results.append({
                        'month': month,
                        'topic_count': len(bertopic_data.get('topic_labels', [])),
                        'main_topics': bertopic_data.get('topic_labels', [])[:3]  # 상위 3개
                    })
                
                # LDA 결과
                if 'lda' in result and result['lda']:
                    lda_data = result['lda']
                    lda_results.append({
                        'month': month,
                        'topic_count': len(lda_data.get('topic_labels', [])),
                        'coherence_score': lda_data.get('coherence_score', 0),
                        'main_topics': lda_data.get('topic_labels', [])[:3]  # 상위 3개
                    })
            
            # 토픽 진화 분석
            topic_evolution = self._analyze_topic_evolution(bertopic_results, lda_results)
            
            # 주요 토픽 추출
            all_bertopic_topics = []
            all_lda_topics = []
            
            for result in bertopic_results:
                all_bertopic_topics.extend(result['main_topics'])
            
            for result in lda_results:
                all_lda_topics.extend(result['main_topics'])
            
            # 토픽 빈도 계산
            from collections import Counter
            bertopic_freq = Counter(all_bertopic_topics)
            lda_freq = Counter(all_lda_topics)
            
            return {
                'bertopic_analysis': {
                    'monthly_results': bertopic_results,
                    'frequent_topics': bertopic_freq.most_common(5),
                    'avg_topics_per_month': np.mean([r['topic_count'] for r in bertopic_results]) if bertopic_results else 0
                },
                'lda_analysis': {
                    'monthly_results': lda_results,
                    'frequent_topics': lda_freq.most_common(5),
                    'avg_topics_per_month': np.mean([r['topic_count'] for r in lda_results]) if lda_results else 0,
                    'avg_coherence': np.mean([r['coherence_score'] for r in lda_results]) if lda_results else 0
                },
                'topic_evolution': topic_evolution
            }
            
        except Exception as e:
            self.logger.error(f"❌ 토픽 분석 요약 실패: {str(e)}")
            return {'error': str(e)}
    
    def _analyze_topic_evolution(self, bertopic_results: List, lda_results: List) -> Dict:
        """토픽 진화 분석"""
        try:
            evolution_analysis = {}
            
            # BERTopic 진화
            if len(bertopic_results) > 1:
                topic_counts = [r['topic_count'] for r in bertopic_results]
                evolution_analysis['bertopic_trend'] = {
                    'trend': '증가' if topic_counts[-1] > topic_counts[0] else '감소' if topic_counts[-1] < topic_counts[0] else '안정',
                    'change': topic_counts[-1] - topic_counts[0],
                    'volatility': np.std(topic_counts)
                }
            
            # LDA 진화
            if len(lda_results) > 1:
                topic_counts = [r['topic_count'] for r in lda_results]
                coherence_scores = [r['coherence_score'] for r in lda_results]
                
                evolution_analysis['lda_trend'] = {
                    'topic_trend': '증가' if topic_counts[-1] > topic_counts[0] else '감소' if topic_counts[-1] < topic_counts[0] else '안정',
                    'coherence_trend': '증가' if coherence_scores[-1] > coherence_scores[0] else '감소' if coherence_scores[-1] < coherence_scores[0] else '안정',
                    'topic_change': topic_counts[-1] - topic_counts[0],
                    'coherence_change': coherence_scores[-1] - coherence_scores[0]
                }
            
            return evolution_analysis
            
        except Exception as e:
            self.logger.error(f"❌ 토픽 진화 분석 실패: {str(e)}")
            return {'error': str(e)}
    
    def _analyze_temporal_changes(self, all_results: Dict) -> Dict:
        """시간적 변화 분석"""
        try:
            if len(all_results) < 2:
                return {'message': '시간적 변화 분석을 위한 충분한 데이터가 없습니다.'}
            
            months = sorted(all_results.keys())
            
            # 댓글 수 변화
            comment_counts = [all_results[month].get('total_comments', 0) for month in months]
            
            # 감성 변화
            positive_ratios = []
            negative_ratios = []
            
            for month in months:
                if 'binary_sentiment' in all_results[month]:
                    binary_data = all_results[month]['binary_sentiment']
                    positive_ratios.append(binary_data.get('positive_ratio', 0))
                    negative_ratios.append(binary_data.get('negative_ratio', 0))
                else:
                    positive_ratios.append(0)
                    negative_ratios.append(0)
            
            # 변화율 계산
            comment_changes = [comment_counts[i] - comment_counts[i-1] for i in range(1, len(comment_counts))]
            positive_changes = [positive_ratios[i] - positive_ratios[i-1] for i in range(1, len(positive_ratios))]
            negative_changes = [negative_ratios[i] - negative_ratios[i-1] for i in range(1, len(negative_ratios))]
            
            # 주요 변화 시점 식별
            significant_changes = []
            
            # 댓글 수 급증/급감 시점
            for i, change in enumerate(comment_changes):
                if abs(change) > np.std(comment_changes) * 2:  # 2 표준편차 이상
                    significant_changes.append({
                        'month': months[i+1],
                        'type': '댓글 수 급증' if change > 0 else '댓글 수 급감',
                        'change': change,
                        'description': f'전월 대비 {abs(change):,}개 {"증가" if change > 0 else "감소"}'
                    })
            
            # 감성 급변 시점
            for i, (pos_change, neg_change) in enumerate(zip(positive_changes, negative_changes)):
                if abs(pos_change) > 0.1 or abs(neg_change) > 0.1:  # 10% 이상 변화
                    significant_changes.append({
                        'month': months[i+1],
                        'type': '감성 급변',
                        'positive_change': pos_change,
                        'negative_change': neg_change,
                        'description': f'긍정 감성 {pos_change:+.1%}, 부정 감성 {neg_change:+.1%} 변화'
                    })
            
            return {
                'comment_trend': {
                    'overall_change': comment_counts[-1] - comment_counts[0],
                    'peak_month': months[comment_counts.index(max(comment_counts))],
                    'peak_count': max(comment_counts),
                    'volatility': np.std(comment_counts)
                },
                'sentiment_trend': {
                    'positive_overall_change': positive_ratios[-1] - positive_ratios[0],
                    'negative_overall_change': negative_ratios[-1] - negative_ratios[0],
                    'most_positive_month': months[positive_ratios.index(max(positive_ratios))],
                    'most_negative_month': months[negative_ratios.index(max(negative_ratios))]
                },
                'significant_changes': significant_changes
            }
            
        except Exception as e:
            self.logger.error(f"❌ 시간적 변화 분석 실패: {str(e)}")
            return {'error': str(e)}
    
    def _extract_key_findings(self, all_results: Dict, target_name: str) -> List[str]:
        """주요 발견사항 추출"""
        try:
            findings = []
            
            # 감성 분석 발견사항
            sentiment_summary = self._summarize_sentiment_analysis(all_results)
            if 'overall_statistics' in sentiment_summary:
                stats = sentiment_summary['overall_statistics']
                
                if stats['positive_ratio'] > 0.6:
                    findings.append(f"{target_name}에 대한 전반적인 여론은 긍정적입니다 ({stats['positive_ratio']:.1%} 긍정).")
                elif stats['negative_ratio'] > 0.6:
                    findings.append(f"{target_name}에 대한 전반적인 여론은 부정적입니다 ({stats['negative_ratio']:.1%} 부정).")
                else:
                    findings.append(f"{target_name}에 대한 여론은 긍정과 부정이 혼재되어 있습니다.")
                
                findings.append(f"주요 감정은 '{stats['dominant_emotion']}'입니다.")
            
            # 시간적 변화 발견사항
            temporal_analysis = self._analyze_temporal_changes(all_results)
            if 'significant_changes' in temporal_analysis:
                significant_events = temporal_analysis['significant_changes']
                if significant_events:
                    findings.append(f"분석 기간 중 {len(significant_events)}개의 주요 변화 시점이 발견되었습니다.")
                    
                    # 가장 큰 변화 시점
                    if any(event['type'] == '댓글 수 급증' for event in significant_events):
                        findings.append("특정 시점에서 댓글 수가 급증하여 높은 관심도를 보였습니다.")
                    
                    if any(event['type'] == '감성 급변' for event in significant_events):
                        findings.append("감성이 급격히 변화한 시점들이 관찰되었습니다.")
            
            # 토픽 분석 발견사항
            topic_summary = self._summarize_topic_analysis(all_results)
            if 'bertopic_analysis' in topic_summary and 'lda_analysis' in topic_summary:
                bertopic_avg = topic_summary['bertopic_analysis']['avg_topics_per_month']
                lda_avg = topic_summary['lda_analysis']['avg_topics_per_month']
                
                if bertopic_avg > 5 or lda_avg > 5:
                    findings.append("다양한 주제들이 논의되고 있어 복합적인 담론 구조를 보입니다.")
                else:
                    findings.append("상대적으로 집중된 주제 논의가 이루어지고 있습니다.")
            
            # 연구 함의
            findings.append(f"이러한 결과는 {target_name} 관련 뉴스 보도만으로는 파악하기 어려운 실제 대중 여론의 복잡성을 보여줍니다.")
            findings.append("시간에 따른 여론 변화는 단순한 뉴스 프레임을 넘어선 다층적 담론 구조를 시사합니다.")
            
            return findings
            
        except Exception as e:
            self.logger.error(f"❌ 주요 발견사항 추출 실패: {str(e)}")
            return [f"발견사항 추출 중 오류 발생: {str(e)}"]
    
    def _generate_research_implications(self, all_results: Dict, target_name: str) -> Dict:
        """연구 함의 생성"""
        try:
            implications = {
                'methodological': [
                    "유튜브 댓글 분석을 통해 기존 뉴스 프레임 연구의 한계를 보완할 수 있음을 확인했습니다.",
                    "월별 시간 분석을 통해 여론의 동적 변화 과정을 추적할 수 있었습니다.",
                    "이진 감성 분석과 다중 감정 분석의 결합으로 더 정교한 감정 분석이 가능했습니다.",
                    "BERTopic과 LDA의 비교 분석으로 토픽 모델링의 신뢰성을 높였습니다."
                ],
                'theoretical': [
                    "공중의 실제 반응은 언론 보도 프레임과 상당한 차이를 보일 수 있음을 시사합니다.",
                    "소셜 미디어 담론은 전통적인 미디어 담론보다 더 복잡하고 다층적인 구조를 가집니다.",
                    "시간적 변화 분석은 여론 형성 과정의 동적 특성을 이해하는 데 중요합니다.",
                    "감정적 반응의 다양성은 단순한 찬반 구조를 넘어선 복합적 여론 구조를 보여줍니다."
                ],
                'practical': [
                    "정책 결정자들은 뉴스 보도뿐만 아니라 실제 대중 반응을 모니터링해야 합니다.",
                    "위기 커뮤니케이션 전략 수립 시 시간적 변화 패턴을 고려해야 합니다.",
                    "소셜 미디어 분석은 여론 조사의 보완적 도구로 활용될 수 있습니다.",
                    "실시간 감정 모니터링을 통한 선제적 대응이 가능합니다."
                ],
                'limitations': [
                    "유튜브 댓글은 전체 인구를 대표하지 않을 수 있습니다.",
                    "익명성으로 인한 극단적 의견의 과대표현 가능성이 있습니다.",
                    "알고리즘에 의한 댓글 노출 편향이 결과에 영향을 줄 수 있습니다.",
                    "감성 분석 모델의 한국어 처리 한계가 존재합니다."
                ]
            }
            
            return implications
            
        except Exception as e:
            self.logger.error(f"❌ 연구 함의 생성 실패: {str(e)}")
            return {'error': str(e)}
    
    def _generate_html_report(self, content: Dict, target_name: str) -> str:
        """HTML 보고서 생성"""
        try:
            html_template = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ target_name }} 유튜브 댓글 분석 보고서</title>
    <style>
        body { font-family: 'Malgun Gothic', sans-serif; line-height: 1.6; margin: 40px; }
        .header { text-align: center; border-bottom: 3px solid #333; padding-bottom: 20px; margin-bottom: 30px; }
        .section { margin-bottom: 30px; }
        .section h2 { color: #2c3e50; border-left: 4px solid #3498db; padding-left: 10px; }
        .section h3 { color: #34495e; margin-top: 20px; }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }
        .stat-card { background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; }
        .stat-value { font-size: 24px; font-weight: bold; color: #2c3e50; }
        .stat-label { color: #7f8c8d; font-size: 14px; }
        .finding { background: #e8f5e8; padding: 10px; margin: 10px 0; border-left: 4px solid #27ae60; }
        .implication { background: #fff3cd; padding: 10px; margin: 10px 0; border-left: 4px solid #ffc107; }
        .limitation { background: #f8d7da; padding: 10px; margin: 10px 0; border-left: 4px solid #dc3545; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <div class="header">
        <h1>{{ target_name }} 유튜브 댓글 분석 보고서</h1>
        <p>분석일: {{ analysis_date }}</p>
        <p>분석 기간: {{ analysis_period.start }} ~ {{ analysis_period.end }} ({{ analysis_period.total_months }}개월)</p>
    </div>

    <div class="section">
        <h2>1. 분석 개요</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{{ "{:,}".format(total_comments) }}</div>
                <div class="stat-label">총 분석 댓글 수</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{{ analysis_period.total_months }}</div>
                <div class="stat-label">분석 기간 (개월)</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{{ "{:.1%}".format(sentiment_summary.overall_statistics.positive_ratio) }}</div>
                <div class="stat-label">전체 긍정 비율</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{{ sentiment_summary.overall_statistics.dominant_emotion }}</div>
                <div class="stat-label">주요 감정</div>
            </div>
        </div>
    </div>

    <div class="section">
        <h2>2. 감성 분석 결과</h2>
        <h3>2.1 전체 감성 분포</h3>
        <p>분석 기간 동안 {{ target_name }}에 대한 전체 댓글의 감성 분포는 다음과 같습니다:</p>
        <ul>
            <li>긍정적 댓글: {{ "{:.1%}".format(sentiment_summary.overall_statistics.positive_ratio) }}</li>
            <li>부정적 댓글: {{ "{:.1%}".format(sentiment_summary.overall_statistics.negative_ratio) }}</li>
            <li>주요 감정: {{ sentiment_summary.overall_statistics.dominant_emotion }}</li>
        </ul>

        <h3>2.2 감성 변화 트렌드</h3>
        {% if sentiment_summary.trend_analysis.positive_trend %}
        <p>긍정 감성 트렌드: <strong>{{ sentiment_summary.trend_analysis.positive_trend }}</strong></p>
        <p>부정 감성 트렌드: <strong>{{ sentiment_summary.trend_analysis.negative_trend }}</strong></p>
        {% endif %}
    </div>

    <div class="section">
        <h2>3. 토픽 분석 결과</h2>
        <h3>3.1 BERTopic 분석</h3>
        <p>월평균 토픽 수: {{ "{:.1f}".format(topic_summary.bertopic_analysis.avg_topics_per_month) }}개</p>
        
        <h3>3.2 LDA 분석</h3>
        <p>월평균 토픽 수: {{ "{:.1f}".format(topic_summary.lda_analysis.avg_topics_per_month) }}개</p>
        <p>평균 일관성 점수: {{ "{:.3f}".format(topic_summary.lda_analysis.avg_coherence) }}</p>
    </div>

    <div class="section">
        <h2>4. 시간적 변화 분석</h2>
        {% if temporal_analysis.comment_trend %}
        <h3>4.1 댓글 수 변화</h3>
        <p>전체 변화량: {{ "{:+,}".format(temporal_analysis.comment_trend.overall_change) }}개</p>
        <p>최고 댓글 수 월: {{ temporal_analysis.comment_trend.peak_month }} ({{ "{:,}".format(temporal_analysis.comment_trend.peak_count) }}개)</p>
        
        <h3>4.2 주요 변화 시점</h3>
        {% for change in temporal_analysis.significant_changes %}
        <div class="finding">
            <strong>{{ change.month }}</strong>: {{ change.description }}
        </div>
        {% endfor %}
        {% endif %}
    </div>

    <div class="section">
        <h2>5. 주요 발견사항</h2>
        {% for finding in key_findings %}
        <div class="finding">{{ finding }}</div>
        {% endfor %}
    </div>

    <div class="section">
        <h2>6. 연구 함의</h2>
        <h3>6.1 방법론적 함의</h3>
        {% for implication in research_implications.methodological %}
        <div class="implication">{{ implication }}</div>
        {% endfor %}

        <h3>6.2 이론적 함의</h3>
        {% for implication in research_implications.theoretical %}
        <div class="implication">{{ implication }}</div>
        {% endfor %}

        <h3>6.3 실무적 함의</h3>
        {% for implication in research_implications.practical %}
        <div class="implication">{{ implication }}</div>
        {% endfor %}
    </div>

    <div class="section">
        <h2>7. 연구 한계</h2>
        {% for limitation in research_implications.limitations %}
        <div class="limitation">{{ limitation }}</div>
        {% endfor %}
    </div>

    <div class="section">
        <h2>8. 결론</h2>
        <p>본 연구는 {{ target_name }} 관련 유튜브 댓글 분석을 통해 기존 뉴스 프레임 연구의 한계를 보완하고, 
        실제 대중 여론의 복잡성과 시간적 변화 양상을 규명했습니다. 
        이는 소셜 미디어 시대의 여론 연구에 새로운 방법론적 접근을 제시합니다.</p>
    </div>
</body>
</html>
            """
            
            template = Template(html_template)
            html_content = template.render(**content)
            
            # 파일 저장
            filename = f"{target_name}_comprehensive_report.html"
            filepath = os.path.join(self.output_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.logger.info(f"✅ HTML 보고서 생성 완료: {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"❌ HTML 보고서 생성 실패: {str(e)}")
            raise
    
    def _generate_markdown_report(self, content: Dict, target_name: str) -> str:
        """마크다운 보고서 생성"""
        try:
            md_content = f"""# {content['target_name']} 유튜브 댓글 분석 보고서

**분석일**: {content['analysis_date']}  
**분석 기간**: {content['analysis_period']['start']} ~ {content['analysis_period']['end']} ({content['analysis_period']['total_months']}개월)

## 1. 분석 개요

- **총 분석 댓글 수**: {content['total_comments']:,}개
- **분석 기간**: {content['analysis_period']['total_months']}개월
- **전체 긍정 비율**: {content['sentiment_summary']['overall_statistics']['positive_ratio']:.1%}
- **주요 감정**: {content['sentiment_summary']['overall_statistics']['dominant_emotion']}

## 2. 감성 분석 결과

### 2.1 전체 감성 분포
- 긍정적 댓글: {content['sentiment_summary']['overall_statistics']['positive_ratio']:.1%}
- 부정적 댓글: {content['sentiment_summary']['overall_statistics']['negative_ratio']:.1%}
- 주요 감정: {content['sentiment_summary']['overall_statistics']['dominant_emotion']}

## 3. 주요 발견사항

"""
            
            for i, finding in enumerate(content['key_findings'], 1):
                md_content += f"{i}. {finding}\n"
            
            md_content += "\n## 4. 연구 함의\n\n"
            md_content += "### 방법론적 함의\n"
            for implication in content['research_implications']['methodological']:
                md_content += f"- {implication}\n"
            
            md_content += "\n### 이론적 함의\n"
            for implication in content['research_implications']['theoretical']:
                md_content += f"- {implication}\n"
            
            # 파일 저장
            filename = f"{target_name}_report.md"
            filepath = os.path.join(self.output_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(md_content)
            
            self.logger.info(f"✅ 마크다운 보고서 생성 완료: {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"❌ 마크다운 보고서 생성 실패: {str(e)}")
            raise
    
    def _convert_numpy_types(self, obj):
        """numpy 타입을 Python 기본 타입으로 변환"""
        import numpy as np
        
        if isinstance(obj, dict):
            return {str(k): self._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        else:
            return obj
    
    def _generate_json_summary(self, all_results: Dict, target_name: str) -> str:
        """JSON 요약 보고서 생성"""
        try:
            summary = {
                'target_name': target_name,
                'analysis_date': datetime.now().isoformat(),
                'total_comments': sum(result.get('total_comments', 0) for result in all_results.values()),
                'analysis_period': {
                    'start': min(all_results.keys()) if all_results else None,
                    'end': max(all_results.keys()) if all_results else None,
                    'total_months': len(all_results)
                },
                'monthly_results': all_results
            }
            
            # numpy 타입 변환
            summary = self._convert_numpy_types(summary)
            
            # 파일 저장
            filename = f"{target_name}_analysis_summary.json"
            filepath = os.path.join(self.output_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
            
            self.logger.info(f"✅ JSON 요약 보고서 생성 완료: {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"❌ JSON 요약 보고서 생성 실패: {str(e)}")
            raise 