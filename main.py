"""
YouTube Comments Analysis - Main Execution Script
유튜브 댓글 분석 메인 실행 스크립트 - 통합 분석 모드
"""

import os
import sys
import logging
import traceback
import argparse
from datetime import datetime

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import AnalysisConfig, validate_config
from src.data_processor import DataProcessor
from src.data_filter import DataFilter
from src.adaptive_time_analyzer import AdaptiveTimeAnalyzer
from src.sentiment_analyzer import SentimentAnalyzer
from src.topic_analyzer import TopicAnalyzer
from src.visualizer import Visualizer
from src.report_generator import ReportGenerator

# 고급 분석 모듈들 (선택적 import)
try:
    from src.advanced_frame_analyzer import AdvancedFrameAnalyzer
    from src.advanced_visualizer import AdvancedVisualizer
    ADVANCED_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ 고급 분석 모듈 import 실패: {e}")
    ADVANCED_MODULES_AVAILABLE = False

def setup_logging(config):
    """로깅 설정"""
    logging.basicConfig(
        level=getattr(logging, config.LOGGING['level']),
        format=config.LOGGING['format'],
        handlers=[
            logging.FileHandler(config.LOGGING['file'], encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def run_basic_analysis(config: AnalysisConfig, logger: logging.Logger):
    """기본 분석 실행"""
    try:
        target_name = config.TARGET_NAME
        
        # 출력 디렉토리 생성
        for dir_path in config.OUTPUT_STRUCTURE.values():
            os.makedirs(dir_path, exist_ok=True)
        
        # 1. 데이터 처리
        logger.info(f"📊 {target_name} 데이터 처리 시작")
        data_processor = DataProcessor(config)
        
        # CSV 파일 로드
        csv_path = config.DATA_FILES['youtube_comments']
        data_processor.load_data(csv_path)
        
        # 전체 데이터를 단일 타겟으로 처리 (키워드 필터링 없이)
        df = data_processor.raw_data
        if df is None or len(df) == 0:
            logger.error(f"❌ 데이터 로드 실패 또는 댓글 없음")
            return None
        
        # 댓글 전처리
        processed_df = data_processor.preprocess_comments(df, target_name)
        
        # 데이터 필터링 적용
        logger.info(f"🔍 {target_name} 데이터 필터링 시작")
        data_filter = DataFilter(config)
        
        # 필터링 설정 검증
        if not data_filter.validate_filtering_config():
            logger.warning("⚠️ 필터링 설정에 문제가 있어 필터링을 건너뜁니다.")
            filtered_df = processed_df
        else:
            # 모든 필터링 적용
            filtered_df = data_filter.apply_all_filters(processed_df)
            
            # 필터링 통계 로깅
            filter_stats = data_filter.get_filtering_stats()
            logger.info(f"📊 필터링 완료: {filter_stats['original_count']:,}개 → {filter_stats['final_count']:,}개 댓글")
        
        # 적응적 시간 분석 (config.py 설정 기반)
        logger.info(f"📊 {target_name} 적응적 시간 분석 시작")
        time_analyzer = AdaptiveTimeAnalyzer(config)
        time_distribution = None  # 초기화
        
        # 적응적 세분화 그룹 생성 (config.py의 ADAPTIVE_TIME_ANALYSIS 설정 사용)
        if config.ADAPTIVE_TIME_ANALYSIS['enabled']:
            logger.info(f"🔄 적응적 시간 세분화 활성화 (임계값: {config.ADAPTIVE_TIME_ANALYSIS['high_ratio_threshold']})")
            time_groups_result = time_analyzer.create_adaptive_subdivided_groups(filtered_df)
            time_groups = time_groups_result['groups']
            optimal_unit = 'adaptive_subdivided'
            
            # 세분화 정보 로깅
            metadata = time_groups_result.get('metadata', {})
            logger.info(f"📅 적응적 세분화 완료: {metadata.get('total_periods', 0)}개 구간")
            logger.info(f"🔧 세분화 적용: {metadata.get('subdivisions_applied', 0)}개 기간")
            
            # 적응적 세분화용 가상 time_distribution 생성
            time_distribution = {
                'optimal_time_unit': optimal_unit,
                'total_comments': len(filtered_df),
                'total_periods': metadata.get('total_periods', 0),
                'subdivisions_applied': metadata.get('subdivisions_applied', 0),
                'recommendation': {
                    'time_unit': optimal_unit,
                    'reasons': ['적응적 시간 세분화가 적용되었습니다.'],
                    'warnings': [],
                    'suggestions': []
                }
            }
        else:
            # 기존 방식 사용
            time_distribution = time_analyzer.analyze_time_distribution(filtered_df)
            
            if time_distribution:
                optimal_unit = time_distribution['optimal_time_unit']
                logger.info(f"📅 최적 시간 단위: {optimal_unit}")
                
                # 적응적 시간 그룹 생성
                time_groups_result = time_analyzer.create_adaptive_time_groups(filtered_df, optimal_unit)
                time_groups = time_groups_result['groups']
                
                # 권장사항 로깅
                recommendation = time_distribution.get('recommendation', {})
                for reason in recommendation.get('reasons', []):
                    logger.info(f"💡 {reason}")
                for warning in recommendation.get('warnings', []):
                    logger.warning(f"⚠️ {warning}")
                for suggestion in recommendation.get('suggestions', []):
                    logger.info(f"🔧 {suggestion}")
            else:
                logger.warning("⚠️ 시간 분포 분석 실패, 월별 분석으로 대체합니다.")
                time_groups = data_processor.group_by_month_single(filtered_df)
                optimal_unit = 'monthly_fallback'
                # 폴백용 time_distribution 생성
                time_distribution = {
                    'optimal_time_unit': optimal_unit,
                    'total_comments': len(filtered_df),
                    'recommendation': {
                        'time_unit': optimal_unit,
                        'reasons': ['시간 분포 분석 실패로 월별 분석을 적용했습니다.'],
                        'warnings': ['데이터 품질이 이상적이지 않을 수 있습니다.'],
                        'suggestions': []
                    }
                }
        
        # 시간 분포 시각화
        time_dist_plot = time_analyzer.visualize_time_distribution(
            {'optimal_time_unit': optimal_unit}, 
            df=filtered_df, 
            date_column='date'
        )
        logger.info(f"📊 시간 분포 분석 완료: {time_dist_plot}")
        
        logger.info(f"✅ 데이터 처리 완료: {len(filtered_df):,}개 댓글 (필터링 후)")
        logger.info(f"📅 분석 기간: {len(time_groups)}개 {optimal_unit} 그룹")
        
        # 2. 감성 분석 (적응적 시간 그룹 사용)
        logger.info(f"😊 {target_name} 감성 분석 시작")
        sentiment_analyzer = SentimentAnalyzer(config)
        sentiment_results = sentiment_analyzer.analyze_time_grouped_sentiment(time_groups, target_name)
        sentiment_trends = sentiment_analyzer.get_sentiment_trends(target_name)
        
        # 3. 토픽 분석 (적응적 시간 그룹 사용) - 조건부 실행
        topic_analyzer = TopicAnalyzer(config)
        if config.TOPIC_ANALYSIS_ENABLED:
            logger.info(f"🔍 {target_name} 토픽 분석 시작")
            topic_results = topic_analyzer.analyze_time_grouped_topics(time_groups, target_name)
            topic_evolution = topic_analyzer.get_topic_evolution(target_name)
        else:
            logger.info(f"⚠️ 토픽 분석이 비활성화되어 있습니다. 키워드 추출만 수행합니다.")
            topic_results = {}
            topic_evolution = {}
        
        # 4. 시각화 생성 (개선된 버전)
        logger.info(f"📊 {target_name} 시각화 생성 시작")
        visualizer = Visualizer(config)
        
        # 5. 종합 결과 구성 먼저 수행
        comprehensive_results = {}
        
        for group_name in time_groups.keys():
            # 기본 종합 결과
            comprehensive_results[group_name] = {
                'total_comments': len(time_groups[group_name]),
                'binary_sentiment': sentiment_results[group_name]['binary_sentiment'] if group_name in sentiment_results else {},
                'emotion_sentiment': sentiment_results[group_name]['emotion_sentiment'] if group_name in sentiment_results else {},
                'bertopic': topic_results[group_name]['bertopic'] if group_name in topic_results else {},
                'lda': topic_results[group_name]['lda'] if group_name in topic_results else {}
            }
        
        # 시간 그룹별 비교 시각화 (감성 중심)
        time_comparison_plot = visualizer.plot_time_grouped_comparison(sentiment_results, target_name, optimal_unit)
        
        # 토픽 분석 대시보드 (새로운 토픽 중심 시각화) - 조건부 실행
        if config.TOPIC_ANALYSIS_ENABLED:
            topic_dashboard_plot = visualizer.create_topic_analysis_dashboard(
                comprehensive_results, target_name, optimal_unit, min_comments_threshold=50
            )
        else:
            topic_dashboard_plot = None
        
        # 기간별 워드클라우드 생성 (모든 시간 그룹에 대해)
        logger.info(f"☁️ {target_name} 기간별 워드클라우드 생성 시작")
        time_group_wordclouds = []
        
        for group_name, group_data in time_groups.items():
            if len(group_data) >= 10:  # 최소 10개 댓글이 있는 그룹만
                try:
                    # 해당 그룹의 텍스트 추출
                    group_texts = group_data['cleaned_text'].tolist()
                    
                    # 키워드 추출
                    group_keywords = topic_analyzer.extract_keywords(group_texts, top_k=50)
                    
                    if group_keywords:
                        # 워드클라우드 생성
                        wordcloud_path = visualizer.create_wordcloud(group_keywords, target_name, group_name)
                        if wordcloud_path:
                            time_group_wordclouds.append(wordcloud_path)
                            logger.info(f"☁️ {group_name} 워드클라우드 생성 완료: {len(group_keywords)}개 키워드")
                    else:
                        logger.warning(f"⚠️ {group_name}: 키워드 추출 실패")
                        
                except Exception as e:
                    logger.error(f"❌ {group_name} 워드클라우드 생성 실패: {str(e)}")
            else:
                logger.info(f"⚠️ {group_name}: 댓글 수 부족({len(group_data)}개), 워드클라우드 생성 건너뜀")
        
        logger.info(f"☁️ 기간별 워드클라우드 생성 완료: {len(time_group_wordclouds)}개")
        
        # 감성 트렌드 시각화 (6감정 포함)
        sentiment_plot = visualizer.plot_sentiment_trends(sentiment_trends, target_name)
        
        # 전체 기간 워드클라우드 (참고용)
        all_texts = []
        for group_data in time_groups.values():
            all_texts.extend(group_data['cleaned_text'].tolist())
        
        keywords = topic_analyzer.extract_keywords(all_texts, top_k=50)
        overall_wordcloud_path = visualizer.create_wordcloud(keywords, target_name, "전체기간")
        
        # 6. 상세 데이터 저장 구성
        time_group_detailed_results = {}
        
        for group_name in time_groups.keys():
            # 상세 시간 그룹별 결과 (CSV 저장용)
            group_texts = time_groups[group_name]['cleaned_text'].tolist()
            group_keywords = topic_analyzer.extract_keywords(group_texts, top_k=20)
            
            time_group_detailed_results[group_name] = {
                'time_group': group_name,
                'total_comments': len(time_groups[group_name]),
                'avg_comment_length': time_groups[group_name]['cleaned_text'].str.len().mean(),
                'top_keywords': [kw[0] for kw in group_keywords[:10]] if group_keywords else [],
                'keyword_scores': [kw[1] for kw in group_keywords[:10]] if group_keywords else [],
                'sentiment_summary': sentiment_results[group_name]['binary_sentiment'] if group_name in sentiment_results else {},
                'emotion_summary': sentiment_results[group_name]['emotion_sentiment'] if group_name in sentiment_results else {},
                'topic_summary': {
                    'bertopic_topics': len(topic_results[group_name]['bertopic'].get('topics', [])) if group_name in topic_results else 0,
                    'lda_topics': len(topic_results[group_name]['lda'].get('topics', [])) if group_name in topic_results else 0
                }
            }
        
        # 시간 그룹별 상세 결과를 CSV로 저장
        import pandas as pd
        time_group_summary_data = []
        for group_name, data in time_group_detailed_results.items():
            row = {
                'time_group': group_name,
                'time_unit': optimal_unit,
                'total_comments': data['total_comments'],
                'avg_comment_length': round(data['avg_comment_length'], 2),
                'top_keywords': ', '.join(data['top_keywords'][:5]),
                'positive_ratio': data['sentiment_summary'].get('positive_ratio', 0),
                'negative_ratio': data['sentiment_summary'].get('negative_ratio', 0),
                'dominant_emotion': data['emotion_summary'].get('dominant_emotion', '없음'),
                'emotion_confidence': data['emotion_summary'].get('confidence', 0),
                'bertopic_topics': data['topic_summary']['bertopic_topics'],
                'lda_topics': data['topic_summary']['lda_topics']
            }
            time_group_summary_data.append(row)
        
        time_group_summary_df = pd.DataFrame(time_group_summary_data)
        time_group_summary_path = os.path.join(config.OUTPUT_STRUCTURE['data_processed'], f'{target_name}_{optimal_unit}_summary.csv')
        time_group_summary_df.to_csv(time_group_summary_path, index=False, encoding='utf-8-sig')
        logger.info(f"📊 시간 그룹별 요약 데이터 저장: {time_group_summary_path}")
        
        # 키워드 상세 분석 결과 저장
        keyword_detailed_data = []
        for group_name, data in time_group_detailed_results.items():
            for i, (keyword, score) in enumerate(zip(data['top_keywords'], data['keyword_scores'])):
                keyword_detailed_data.append({
                    'time_group': group_name,
                    'time_unit': optimal_unit,
                    'rank': i + 1,
                    'keyword': keyword,
                    'score': score,
                    'total_comments': data['total_comments']
                })
        
        if keyword_detailed_data:
            keyword_detailed_df = pd.DataFrame(keyword_detailed_data)
            keyword_detailed_path = os.path.join(config.OUTPUT_STRUCTURE['data_processed'], f'{target_name}_keywords_detailed.csv')
            keyword_detailed_df.to_csv(keyword_detailed_path, index=False, encoding='utf-8-sig')
            logger.info(f"🔍 키워드 상세 데이터 저장: {keyword_detailed_path}")
        
        # 6. 보고서 생성
        logger.info(f"📋 {target_name} 보고서 생성 시작")
        report_generator = ReportGenerator(config)
        report_path = report_generator.generate_comprehensive_report(comprehensive_results, target_name)
        
        # 7. 인터랙티브 대시보드 생성
        dashboard_path = visualizer.create_interactive_dashboard(comprehensive_results, target_name)
        
        logger.info(f"✅ {target_name} 기본 분석 완료")
        
        # 결과 데이터 구성
        processed_result = {
            'total_comments': len(filtered_df),
            'data': filtered_df,
            'filtering_stats': data_filter.get_filtering_stats() if 'data_filter' in locals() else None,
            'time_groups': time_groups,
            'time_analysis': {
                'optimal_unit': optimal_unit,
                'distribution_result': time_distribution,
                'visualization_path': time_dist_plot
            },
            'date_range': {
                'start': filtered_df['date'].min() if 'date' in filtered_df.columns else None,
                'end': filtered_df['date'].max() if 'date' in filtered_df.columns else None
            }
        }
        
        return {
            'target_name': target_name,
            'processed_data': processed_result,
            'sentiment_results': sentiment_results,
            'sentiment_trends': sentiment_trends,
            'topic_results': topic_results,
            'topic_evolution': topic_evolution,
            'comprehensive_results': comprehensive_results,
            'visualizations': {
                'sentiment_plot': sentiment_plot,
                'topic_dashboard_plot': topic_dashboard_plot,
                'time_comparison_plot': time_comparison_plot,
                'time_distribution_plot': time_dist_plot,
                'overall_wordcloud': overall_wordcloud_path,
                'time_group_wordclouds': time_group_wordclouds,
                'dashboard': dashboard_path
            },
            'reports': {
                'comprehensive_report': report_path
            }
        }
        
    except Exception as e:
        logger.error(f"❌ {target_name} 기본 분석 실패: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def run_advanced_analysis(config: AnalysisConfig, logger: logging.Logger, basic_result: dict):
    """고급 분석 실행"""
    try:
        if not ADVANCED_MODULES_AVAILABLE:
            logger.error("❌ 고급 분석 모듈을 사용할 수 없습니다.")
            return None
        
        target_name = config.TARGET_NAME
        logger.info(f"🚀 {target_name} 고급 분석 시작")
        
        # 기본 분석 결과에서 데이터 추출
        filtered_df = basic_result['processed_data']['data']
        time_groups = basic_result['processed_data']['time_groups']
        
        # 고급 분석기 초기화
        advanced_analyzer = AdvancedFrameAnalyzer(config)
        advanced_visualizer = AdvancedVisualizer(config)
        
        # 1. 시간 기반 여론 흐름 분석
        logger.info(f"📊 {target_name} 시간 기반 여론 흐름 분석")
        temporal_flow = advanced_analyzer.analyze_temporal_opinion_flow(filtered_df, target_name)
        
        # 2. 종합적 토픽 분석
        logger.info(f"🔍 {target_name} 종합 토픽 분석")
        comprehensive_topics = advanced_analyzer.analyze_topics_comprehensive(
            filtered_df, target_name, methods=['lda', 'nmf', 'bertopic']
        )
        
        # 3. 키워드 공출현 네트워크 분석
        logger.info(f"🕸️ {target_name} 키워드 네트워크 분석")
        keyword_network = advanced_analyzer.analyze_keyword_cooccurrence_network(
            filtered_df, target_name, min_cooccurrence=3
        )
        
        # 4. 변곡점 및 이상치 탐지
        logger.info(f"📈 {target_name} 변곡점 탐지")
        anomalies = advanced_analyzer.detect_anomalies_and_changepoints(filtered_df, target_name)
        
        # 5. 문맥 임베딩 클러스터링
        logger.info(f"🧠 {target_name} 문맥 임베딩 분석")
        contextual_clusters = advanced_analyzer.analyze_contextual_embeddings(filtered_df, target_name)
        
        # 6. 고급 시각화 생성
        logger.info(f"📊 {target_name} 고급 시각화 생성")
        
        # 시간별 여론 흐름 시각화
        temporal_flow_plot = advanced_visualizer.visualize_temporal_opinion_flow(
            temporal_flow, target_name
        )
        
        # 토픽 진화 시각화
        topic_evolution_plot = advanced_visualizer.visualize_topic_evolution(
            comprehensive_topics, target_name
        )
        
        # 키워드 네트워크 시각화
        network_plot = advanced_visualizer.visualize_keyword_network(
            keyword_network, target_name
        )
        
        # 종합 대시보드 생성
        comprehensive_dashboard = advanced_visualizer.create_comprehensive_dashboard(
            {
                'temporal_flow': temporal_flow,
                'topics': comprehensive_topics,
                'network': keyword_network,
                'anomalies': anomalies,
                'clusters': contextual_clusters
            },
            target_name
        )
        
        # 7. 고급 분석 결과 저장
        advanced_results = {
            'temporal_flow': temporal_flow,
            'comprehensive_topics': comprehensive_topics,
            'keyword_network': keyword_network,
            'anomalies': anomalies,
            'contextual_clusters': contextual_clusters
        }
        
        # 결과 저장
        advanced_results_path = os.path.join(
            config.OUTPUT_STRUCTURE['data_processed'], 
            f'{target_name}_advanced_analysis.pkl'
        )
        advanced_analyzer.save_analysis_results(advanced_results_path)
        
        logger.info(f"✅ {target_name} 고급 분석 완료")
        
        return {
            'advanced_results': advanced_results,
            'advanced_visualizations': {
                'temporal_flow_plot': temporal_flow_plot,
                'topic_evolution_plot': topic_evolution_plot,
                'network_plot': network_plot,
                'comprehensive_dashboard': comprehensive_dashboard
            }
        }
        
    except Exception as e:
        logger.error(f"❌ {target_name} 고급 분석 실패: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def main():
    """메인 실행 함수"""
    # 인자 파서 설정
    parser = argparse.ArgumentParser(description='YouTube 댓글 분석 도구')
    parser.add_argument('--basic-only', action='store_true', 
                       help='기본 분석만 수행 (감성 분석 + 키워드 추출)')
    
    args = parser.parse_args()
    
    # 분석 수행
    if args.basic_only:
        # 기본 분석만 수행
        print("📊 기본 분석 모드")
        try:
            # 1. 설정 검증
            print("🔧 설정 검증 중...")
            if not validate_config():
                print("❌ 설정 검증 실패")
                return
            
            config = AnalysisConfig()
            logger = setup_logging(config)
            
            logger.info("🚀 YouTube Comments Analysis 시작")
            logger.info(f"📱 디바이스: {config.DEVICE}")
            logger.info(f"🎯 분석 모드: {config.ANALYSIS_MODE}")
            logger.info(f"📄 분석 파일: {config.DATA_FILES['youtube_comments']}")
            
            # 2. 기본 분석 실행
            print(f"\n🎯 {config.TARGET_NAME} 기본 분석 시작...")
            
            basic_result = run_basic_analysis(config, logger)
            
            if not basic_result:
                print("❌ 기본 분석이 실패했습니다.")
                return
            
            print(f"✅ {config.TARGET_NAME} 기본 분석 완료")
            
            # 주요 결과 출력
            total_comments = basic_result['processed_data']['total_comments']
            date_range = basic_result['processed_data']['date_range']
            
            print(f"   📊 총 댓글 수: {total_comments:,}개")
            if date_range['start'] and date_range['end']:
                print(f"   📅 분석 기간: {date_range['start'].strftime('%Y-%m-%d')} ~ {date_range['end'].strftime('%Y-%m-%d')}")
            
            # 감성 분석 요약
            if basic_result['sentiment_trends']:
                months = basic_result['sentiment_trends']['months']
                if months:
                    avg_positive = sum(basic_result['sentiment_trends']['positive_ratios']) / len(months)
                    avg_negative = sum(basic_result['sentiment_trends']['negative_ratios']) / len(months)
                    print(f"   😊 평균 긍정 비율: {avg_positive:.1%}")
                    print(f"   😞 평균 부정 비율: {avg_negative:.1%}")
            
            # 토픽 분석 요약
            if basic_result['topic_results']:
                bertopic_months = len([m for m, r in basic_result['topic_results'].items() 
                                     if r.get('bertopic') and r['bertopic'].get('topic_labels')])
                lda_months = len([m for m, r in basic_result['topic_results'].items() 
                                if r.get('lda') and r['lda'].get('topic_labels')])
                print(f"   🔍 BERTopic 분석 완료: {bertopic_months}개월")
                print(f"   📚 LDA 분석 완료: {lda_months}개월")
            
            # 4. 생성된 파일들 출력
            print(f"\n📂 생성된 파일들:")
            if basic_result['reports']['comprehensive_report']:
                print(f"   📋 종합 보고서: {os.path.basename(basic_result['reports']['comprehensive_report'])}")
            
            viz = basic_result['visualizations']
            if viz.get('sentiment_plot'):
                print(f"   📊 감성 트렌드: {os.path.basename(viz['sentiment_plot'])}")
            if viz.get('topic_dashboard_plot'):
                print(f"   🔍 토픽 대시보드: {os.path.basename(viz['topic_dashboard_plot'])}")
            if viz.get('time_comparison_plot'):
                print(f"   📈 시간별 비교: {os.path.basename(viz['time_comparison_plot'])}")
            if viz.get('time_distribution_plot'):
                print(f"   📊 시간 분포 분석: {os.path.basename(viz['time_distribution_plot'])}")
            if viz.get('overall_wordcloud'):
                print(f"   ☁️ 워드클라우드: {os.path.basename(viz['overall_wordcloud'])}")
            if viz.get('time_group_wordclouds') and len(viz['time_group_wordclouds']) > 0:
                print(f"   ☁️ 기간별 워드클라우드: {len(viz['time_group_wordclouds'])}개")
            if viz.get('dashboard'):
                print(f"   📱 대시보드: {os.path.basename(viz['dashboard'])}")
            
            print(f"\n✅ 분석 완료!")
            print(f"📁 결과 저장 위치: {config.OUTPUT_DIR}")
            
            # 사용법 안내
            print(f"\n💡 사용법:")
            print(f"   기본 분석: uv run python main.py --basic-only")
            
            logger.info("🎉 YouTube Comments Analysis 완료")
            
        except Exception as e:
            print(f"❌ 메인 실행 오류: {str(e)}")
            print(traceback.format_exc())
            return
    else:
        # 전체 분석 수행 (기본 + 고급)
        print("📊 기본 분석 모드 (기본값)")
        try:
            # 1. 설정 검증
            print("🔧 설정 검증 중...")
            if not validate_config():
                print("❌ 설정 검증 실패")
                return
            
            config = AnalysisConfig()
            logger = setup_logging(config)
            
            logger.info("🚀 YouTube Comments Analysis 시작")
            logger.info(f"📱 디바이스: {config.DEVICE}")
            logger.info(f"🎯 분석 모드: {config.ANALYSIS_MODE}")
            logger.info(f"📄 분석 파일: {config.DATA_FILES['youtube_comments']}")
            
            # 2. 기본 분석 실행
            print(f"\n🎯 {config.TARGET_NAME} 기본 분석 시작...")
            
            basic_result = run_basic_analysis(config, logger)
            
            if not basic_result:
                print("❌ 기본 분석이 실패했습니다.")
                return
            
            print(f"✅ {config.TARGET_NAME} 기본 분석 완료")
            
            # 주요 결과 출력
            total_comments = basic_result['processed_data']['total_comments']
            date_range = basic_result['processed_data']['date_range']
            
            print(f"   📊 총 댓글 수: {total_comments:,}개")
            if date_range['start'] and date_range['end']:
                print(f"   📅 분석 기간: {date_range['start'].strftime('%Y-%m-%d')} ~ {date_range['end'].strftime('%Y-%m-%d')}")
            
            # 감성 분석 요약
            if basic_result['sentiment_trends']:
                months = basic_result['sentiment_trends']['months']
                if months:
                    avg_positive = sum(basic_result['sentiment_trends']['positive_ratios']) / len(months)
                    avg_negative = sum(basic_result['sentiment_trends']['negative_ratios']) / len(months)
                    print(f"   😊 평균 긍정 비율: {avg_positive:.1%}")
                    print(f"   😞 평균 부정 비율: {avg_negative:.1%}")
            
            # 토픽 분석 요약
            if basic_result['topic_results']:
                bertopic_months = len([m for m, r in basic_result['topic_results'].items() 
                                     if r.get('bertopic') and r['bertopic'].get('topic_labels')])
                lda_months = len([m for m, r in basic_result['topic_results'].items() 
                                if r.get('lda') and r['lda'].get('topic_labels')])
                print(f"   🔍 BERTopic 분석 완료: {bertopic_months}개월")
                print(f"   📚 LDA 분석 완료: {lda_months}개월")
            
            # 3. 고급 분석 실행 (옵션)
            advanced_result = None
            print(f"\n🚀 {config.TARGET_NAME} 고급 분석 시작...")
            advanced_result = run_advanced_analysis(config, logger, basic_result)
            
            if advanced_result:
                print(f"✅ {config.TARGET_NAME} 고급 분석 완료")
                print(f"   🔍 시간 기반 여론 흐름 분석 완료")
                print(f"   📊 종합 토픽 분석 완료")
                print(f"   🕸️ 키워드 네트워크 분석 완료")
                print(f"   📈 변곡점 탐지 완료")
                print(f"   🧠 문맥 임베딩 분석 완료")
            else:
                print("⚠️ 고급 분석이 실패했지만 기본 분석은 완료되었습니다.")
            
            # 4. 생성된 파일들 출력
            print(f"\n📂 생성된 파일들:")
            if basic_result['reports']['comprehensive_report']:
                print(f"   📋 종합 보고서: {os.path.basename(basic_result['reports']['comprehensive_report'])}")
            
            viz = basic_result['visualizations']
            if viz.get('sentiment_plot'):
                print(f"   📊 감성 트렌드: {os.path.basename(viz['sentiment_plot'])}")
            if viz.get('topic_dashboard_plot'):
                print(f"   🔍 토픽 대시보드: {os.path.basename(viz['topic_dashboard_plot'])}")
            if viz.get('time_comparison_plot'):
                print(f"   📈 시간별 비교: {os.path.basename(viz['time_comparison_plot'])}")
            if viz.get('time_distribution_plot'):
                print(f"   📊 시간 분포 분석: {os.path.basename(viz['time_distribution_plot'])}")
            if viz.get('overall_wordcloud'):
                print(f"   ☁️ 워드클라우드: {os.path.basename(viz['overall_wordcloud'])}")
            if viz.get('time_group_wordclouds') and len(viz['time_group_wordclouds']) > 0:
                print(f"   ☁️ 기간별 워드클라우드: {len(viz['time_group_wordclouds'])}개")
            if viz.get('dashboard'):
                print(f"   📱 대시보드: {os.path.basename(viz['dashboard'])}")
            
            # 고급 분석 결과 파일들
            if advanced_result:
                adv_viz = advanced_result['advanced_visualizations']
                if adv_viz.get('temporal_flow_plot'):
                    print(f"   📊 시간별 여론 흐름: {os.path.basename(adv_viz['temporal_flow_plot'])}")
                if adv_viz.get('topic_evolution_plot'):
                    print(f"   🔍 토픽 진화: {os.path.basename(adv_viz['topic_evolution_plot'])}")
                if adv_viz.get('network_plot'):
                    print(f"   🕸️ 키워드 네트워크: {os.path.basename(adv_viz['network_plot'])}")
                if adv_viz.get('comprehensive_dashboard'):
                    print(f"   📱 종합 대시보드: {os.path.basename(adv_viz['comprehensive_dashboard'])}")
            
            print(f"\n✅ 분석 완료!")
            print(f"📁 결과 저장 위치: {config.OUTPUT_DIR}")
            
            # 사용법 안내
            print(f"\n💡 사용법:")
            print(f"   기본 분석: uv run python main.py --basic-only")
            
            logger.info("🎉 YouTube Comments Analysis 완료")
            
            # 8. 종합 결과 준비
            comprehensive_results = {
                'basic_analysis': basic_result,
                'advanced_analysis': advanced_result,
                'target_name': config.TARGET_NAME
            }
            
            # 보고서 생성을 위한 기본 분석 결과 추출 (comprehensive_results 형태로 변환)
            basic_analysis_for_report = basic_result['comprehensive_results']
            
            # 9. 연구 목적 특화 분석 (뉴스 vs 대중 담론 차이)
            print(f"\n🔬 {config.TARGET_NAME} 연구 목적 특화 분석...")
            
            # visualizer 객체 생성
            from src.visualizer import Visualizer
            visualizer = Visualizer(config)
            
            # 프레임 변화 속도 분석
            try:
                frame_velocity_plot = visualizer.plot_frame_velocity_analysis(
                    basic_result['topic_results'], config.TARGET_NAME
                )
                if frame_velocity_plot:
                    print(f"   ✅ 프레임 변화 속도 분석: {os.path.basename(frame_velocity_plot)}")
            except Exception as e:
                logger.warning(f"⚠️ 프레임 변화 속도 분석 실패: {str(e)}")
            
            # 숨겨진 하위 담론 발굴 분석
            try:
                hidden_discourse_plot = visualizer.plot_hidden_discourse_analysis(
                    basic_result['topic_results'], config.TARGET_NAME
                )
                if hidden_discourse_plot:
                    print(f"   ✅ 숨겨진 하위 담론 분석: {os.path.basename(hidden_discourse_plot)}")
            except Exception as e:
                logger.warning(f"⚠️ 숨겨진 하위 담론 분석 실패: {str(e)}")
            
            # 감정 강도 분석 (기존)
            try:
                emotion_intensity_plot = visualizer.plot_emotion_intensity_analysis(
                    basic_result['topic_results'], config.TARGET_NAME
                )
                if emotion_intensity_plot:
                    print(f"   ✅ 감정 강도 분석: {os.path.basename(emotion_intensity_plot)}")
            except Exception as e:
                logger.warning(f"⚠️ 감정 강도 분석 실패: {str(e)}")
            
            # 10. 종합 보고서 생성
            logger.info(f"📋 {config.TARGET_NAME} 보고서 생성 시작")
            from src.report_generator import ReportGenerator
            report_generator = ReportGenerator(config)
            report_path = report_generator.generate_comprehensive_report(basic_analysis_for_report, config.TARGET_NAME)
            
            # 11. 인터랙티브 대시보드 생성
            dashboard_path = visualizer.create_interactive_dashboard(basic_analysis_for_report, config.TARGET_NAME)
            
            logger.info(f"✅ {config.TARGET_NAME} 보고서 생성 완료")
            
        except Exception as e:
            print(f"❌ 메인 실행 오류: {str(e)}")
            print(traceback.format_exc())
            return

if __name__ == "__main__":
    main() 