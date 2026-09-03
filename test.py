#!/usr/bin/env python3
"""
Test suite for the Kenyan Stock Analyzer.

Tests data acquisition, analysis engine, report generation,
sector analysis, email/Telegram notifiers, and config loading.
"""

import sys
import os
import unittest
import tempfile
from unittest.mock import patch, Mock
import pandas as pd
import numpy as np
import requests

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from config import Config
from logger import setup_logging, get_logger
from utils import retry, safe_float, detect_support_resistance
from analysis_engine import AnalysisEngine
from sector_analysis import SectorAnalyzer

# Quiet logging during tests
import logging
logging.disable(logging.CRITICAL)


def make_sample_data(periods=200, seed=42):
    """Generate reproducible sample OHLCV data."""
    np.random.seed(seed)
    dates = pd.date_range('2025-01-01', periods=periods, freq='B')
    close = np.random.randn(periods).cumsum() + 100
    return pd.DataFrame({
        'open': close + np.random.randn(periods) * 0.5,
        'high': close + abs(np.random.randn(periods)) * 2,
        'low': close - abs(np.random.randn(periods)) * 2,
        'close': close,
        'volume': np.random.randint(50000, 500000, periods),
    }, index=dates)


class TestConfig(unittest.TestCase):
    """Test configuration loading."""

    def test_config_loads(self):
        config = Config()
        self.assertIsNotNone(config.stock_symbols)
        self.assertGreater(len(config.stock_symbols), 0)
        self.assertIn('SCOM', config.stock_symbols)
        self.assertIsNotNone(config.data_sources)
        self.assertIn('nse_pdf', config.data_sources)

    def test_analysis_params(self):
        config = Config()
        self.assertEqual(config.rsi_period, 14)
        self.assertEqual(config.macd_fast, 12)
        self.assertEqual(config.macd_slow, 26)


class TestUtils(unittest.TestCase):
    """Test utility functions."""

    def test_safe_float(self):
        self.assertEqual(safe_float('123.45'), 123.45)
        self.assertEqual(safe_float('abc'), 0.0)
        self.assertEqual(safe_float(None), 0.0)

    def test_support_resistance(self):
        prices = np.array([10, 11, 12, 11, 10, 9, 10, 11, 12, 13, 12, 11, 10, 11, 12])
        supports, resistances = detect_support_resistance(prices, window=3)
        self.assertIsInstance(supports, list)
        self.assertIsInstance(resistances, list)

    def test_retry_decorator(self):
        call_count = [0]

        @retry(max_attempts=3, backoff=0.01, exceptions=(ValueError,))
        def flaky():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ValueError("fail")
            return "success"

        result = flaky()
        self.assertEqual(result, "success")
        self.assertEqual(call_count[0], 3)


class TestAnalysisEngine(unittest.TestCase):
    """Test technical analysis calculations."""

    @classmethod
    def setUpClass(cls):
        cls.engine = AnalysisEngine()
        cls.data = make_sample_data(200)
        cls.result = cls.engine.analyze_stock(cls.data)

    def test_indicators_present(self):
        df = self.result['data']
        expected = ['sma_20', 'sma_50', 'ema_12', 'ema_26', 'rsi',
                     'macd', 'macd_signal', 'macd_hist',
                     'bb_upper', 'bb_middle', 'bb_lower',
                     'atr', 'obv', 'stoch_k', 'stoch_d', 'volume_sma_20']
        for col in expected:
            self.assertIn(col, df.columns, f"Missing column: {col}")

    def test_signals_present(self):
        signals = self.result['signals']
        expected = ['ma_crossover', 'rsi', 'macd', 'bollinger', 'trend',
                     'stochastic', 'volume', 'overall']
        for sig in expected:
            self.assertIn(sig, signals, f"Missing signal: {sig}")

    def test_signal_values(self):
        signals = self.result['signals']
        valid = {'bullish', 'bearish', 'overbought', 'oversold', 'neutral',
                 'above_upper', 'below_lower', 'within_bands', 'undefined',
                 'golden_cross', 'death_cross', 'bullish_cross', 'bearish_cross',
                 'high_volume', 'low_volume', 'normal'}
        for name, value in signals.items():
            self.assertIn(value, valid, f"Invalid signal '{value}' for '{name}'")

    def test_support_resistance(self):
        supports = self.result.get('support', [])
        resistances = self.result.get('resistance', [])
        self.assertIsInstance(supports, list)
        self.assertIsInstance(resistances, list)

    def test_daily_change(self):
        chg = self.result.get('daily_change_pct')
        self.assertIsNotNone(chg)
        self.assertIsInstance(chg, float)

    def test_multi_stock_analysis(self):
        data_dict = {
            'SCOM': make_sample_data(100, seed=1),
            'EQTY': make_sample_data(100, seed=2),
        }
        results = self.engine.analyze_multiple_stocks(data_dict)
        self.assertIn('SCOM', results)
        self.assertIn('EQTY', results)

    def test_market_breadth(self):
        data_dict = {
            'SCOM': make_sample_data(100, seed=1),
            'EQTY': make_sample_data(100, seed=2),
            'KCB': make_sample_data(100, seed=3),
        }
        results = self.engine.analyze_multiple_stocks(data_dict)
        breadth = self.engine.calculate_market_breadth(results)
        self.assertIn('total_stocks', breadth)
        self.assertEqual(breadth['total_stocks'], 3)
        self.assertIn('pct_above_sma50', breadth)

    def test_rsi_range(self):
        df = self.result['data']
        rsi = df['rsi'].dropna()
        if len(rsi) > 0:
            self.assertTrue((rsi >= 0).all(), "RSI should be >= 0")
            self.assertTrue((rsi <= 100).all(), "RSI should be <= 100")

    def test_empty_data(self):
        result = self.engine.analyze_stock(pd.DataFrame())
        self.assertEqual(result, {})


class TestSectorAnalysis(unittest.TestCase):
    """Test sector analysis."""

    @classmethod
    def setUpClass(cls):
        cls.analyzer = SectorAnalyzer()
        cls.engine = AnalysisEngine()

    def test_sector_mapping(self):
        self.assertEqual(self.analyzer.get_sector('SCOM'), 'Telecommunication')
        self.assertEqual(self.analyzer.get_sector('EQTY'), 'Banking')
        self.assertEqual(self.analyzer.get_sector('KCB'), 'Banking')
        self.assertEqual(self.analyzer.get_sector('EABL'), 'Manufacturing')
        self.assertEqual(self.analyzer.get_sector('UNKNOWN'), 'Other')

    def test_analyze_sectors(self):
        data_dict = {
            'SCOM': make_sample_data(100, seed=1),
            'EQTY': make_sample_data(100, seed=2),
            'KCB': make_sample_data(100, seed=3),
        }
        results = self.engine.analyze_multiple_stocks(data_dict)
        sectors = self.analyzer.analyze_sectors(data_dict, results)

        self.assertIn('Telecommunication', sectors)
        self.assertIn('Banking', sectors)

        banking = sectors['Banking']
        self.assertIn('EQTY', banking['symbols'])
        self.assertIn('KCB', banking['symbols'])
        self.assertEqual(banking['count'], 2)
        self.assertIn('avg_change_pct', banking)
        self.assertIn('avg_rsi', banking)
        self.assertIn('bullish_ratio', banking)


class TestScoring(unittest.TestCase):
    """Test the transparent factor-scoring module, including sector-relative
    valuation (IMPROVEMENTS.txt item 1). Pure computation -- no network."""

    def test_score_stock_absolute_only(self):
        """No sector_medians given -> scores on the absolute curve alone,
        same as before sector-relative scoring existed."""
        from scoring import score_stock
        fund = {
            'pe_ratio': 8.0, 'price_to_book': 1.0, 'roe': 25.0,
            'net_margin': 20.0, 'debt_to_equity': 0.0, 'current_ratio': 2.0,
            'dividend_yield': 8.0, 'dividend_payout_ratio': 50.0,
            'value_traded': 1e9, 'sector': 'Banking',
        }
        result = score_stock('TEST', {}, fund)
        self.assertIsNotNone(result['value'])
        self.assertIsNotNone(result['overall'])
        self.assertTrue(all('sector:' not in r for r in result['reasons']['value']))

    def test_sector_relative_blend_rewards_cheap_vs_peers(self):
        """A stock cheap relative to its sector's P/E median should score
        higher on 'value' than the same stock scored on the absolute curve
        alone -- the whole point of IMPROVEMENTS.txt item 1."""
        from scoring import _score_value

        fund = {'pe_ratio': 20.0, 'sector': 'Banking'}
        absolute_score, _ = _score_value(fund)  # no sector_medians

        # Sector median P/E of 60 -- this stock (P/E 20) is cheap vs peers.
        sector_medians = {'Banking': {'pe_ratio': 60.0, 'count': 5}}
        blended_score, reasons = _score_value(fund, sector_medians)

        self.assertGreater(blended_score, absolute_score)
        self.assertTrue(any('cheaper than sector' in r for r in reasons))

    def test_sector_relative_blend_penalises_expensive_vs_peers(self):
        """A stock pricier than its sector median should score lower than
        the absolute curve alone."""
        from scoring import _score_value

        fund = {'pe_ratio': 20.0, 'sector': 'Banking'}
        absolute_score, _ = _score_value(fund)

        sector_medians = {'Banking': {'pe_ratio': 10.0, 'count': 5}}
        blended_score, reasons = _score_value(fund, sector_medians)

        self.assertLess(blended_score, absolute_score)
        self.assertTrue(any('pricier than sector' in r for r in reasons))

    def test_thin_sector_falls_back_to_absolute(self):
        """Fewer than MIN_SECTOR_PEERS in the sector -> ignore the median
        entirely rather than scoring against a noisy 1-2 stock 'average'."""
        from scoring import _score_value, MIN_SECTOR_PEERS

        fund = {'pe_ratio': 15.0, 'sector': 'Investment'}
        absolute_score, _ = _score_value(fund)

        thin_sector_medians = {'Investment': {'pe_ratio': 100.0, 'count': MIN_SECTOR_PEERS - 1}}
        blended_score, reasons = _score_value(fund, thin_sector_medians)

        self.assertEqual(blended_score, absolute_score)
        self.assertTrue(all('sector:' not in r for r in reasons))

    def test_missing_sector_median_metric_degrades_gracefully(self):
        """A sector with enough peers but no P/E median on record (e.g. no
        stock in it has a positive P/E) must not raise -- falls back to the
        absolute score for that metric, per CLAUDE.md's fail-safe rule."""
        from scoring import _score_value

        fund = {'pe_ratio': 15.0, 'sector': 'Banking'}
        sector_medians = {'Banking': {'pe_ratio': None, 'count': 5}}
        score, reasons = _score_value(fund, sector_medians)

        self.assertIsNotNone(score)
        self.assertTrue(all('sector:' not in r for r in reasons))

    def test_roe_and_dividend_yield_also_sector_blended(self):
        """Item 1 covers P/E, P/B, dividend yield, and ROE -- check the
        other two functions got the same treatment."""
        from scoring import _score_quality, _score_dividend

        fund_q = {'roe': 30.0, 'sector': 'Banking'}
        sector_medians_q = {'Banking': {'roe': 15.0, 'count': 5}}
        _, reasons_q = _score_quality(fund_q, sector_medians_q)
        self.assertTrue(any('above sector' in r for r in reasons_q))

        fund_d = {'dividend_yield': 2.0, 'sector': 'Banking'}
        sector_medians_d = {'Banking': {'dividend_yield': 8.0, 'count': 5}}
        _, reasons_d = _score_dividend(fund_d, sector_medians_d)
        self.assertTrue(any('below sector' in r for r in reasons_d))

    def test_generate_alerts(self):
        from scoring import generate_alerts
        analysis_result = {'latest': {'rsi': 25, 'close': 10.3},
                           'signals': {'macd': 'bullish_cross'}}
        fund = {'price_52w_low': 10.3}
        alerts = generate_alerts('TEST', analysis_result, fund)
        self.assertTrue(any('Oversold' in a for a in alerts))
        self.assertTrue(any('MACD bullish' in a for a in alerts))
        self.assertTrue(any('52-week low' in a for a in alerts))

    # ---- Horizon classification (IMPROVEMENTS.txt item 8) ----

    def test_horizon_short_term_momentum_only(self):
        """Strong momentum, no confirming fundamentals -> SHORT-TERM, period
        tied to the specific technical trigger (MACD cross)."""
        from scoring import _classify_horizon
        scores = {'value': None, 'quality': None, 'momentum': 80, 'dividend': None, 'liquidity': 50}
        analysis_result = {'signals': {'macd': 'bullish_cross'}, 'latest': {'rsi': 55}}
        h = _classify_horizon(scores, analysis_result)
        self.assertEqual(h['label'], 'SHORT-TERM')
        self.assertEqual(h['period'], '1-4 weeks')
        self.assertIn('momentum', h['drivers'])

    def test_horizon_short_term_rsi_oversold_trigger(self):
        from scoring import _classify_horizon
        scores = {'value': None, 'quality': None, 'momentum': 70, 'dividend': None, 'liquidity': None}
        analysis_result = {'signals': {'macd': 'bullish'}, 'latest': {'rsi': 25}}
        h = _classify_horizon(scores, analysis_result)
        self.assertEqual(h['label'], 'SHORT-TERM')
        self.assertEqual(h['period'], 'days')

    def test_horizon_long_term_fundamentals_agree(self):
        """Value, quality and dividend all attractive and agreeing, no
        strong momentum either way -> LONG-TERM, 6-12+ months."""
        from scoring import _classify_horizon
        scores = {'value': 80, 'quality': 75, 'momentum': 50, 'dividend': 70, 'liquidity': 40}
        h = _classify_horizon(scores, {})
        self.assertEqual(h['label'], 'LONG-TERM')
        self.assertEqual(h['period'], '6-12+ months')
        self.assertEqual(set(h['drivers']), {'value', 'quality', 'dividend'})

    def test_horizon_long_term_with_tailwind(self):
        """Fundamentals agree AND momentum is also strong -> LONG-TERM, but
        the period notes the near-term tailwind."""
        from scoring import _classify_horizon
        scores = {'value': 80, 'quality': 75, 'momentum': 85, 'dividend': 70, 'liquidity': 40}
        h = _classify_horizon(scores, {})
        self.assertEqual(h['label'], 'LONG-TERM')
        self.assertIn('tailwind', h['period'])
        self.assertIn('momentum', h['drivers'])

    def test_horizon_mixed_cheap_but_bearish(self):
        """The spec's own example: fundamentals attractive but technically
        bearish -> MIXED, not forced into either single label."""
        from scoring import _classify_horizon
        scores = {'value': 85, 'quality': 70, 'momentum': 20, 'dividend': 60, 'liquidity': 50}
        h = _classify_horizon(scores, {})
        self.assertEqual(h['label'], 'MIXED')
        self.assertIsNone(h['period'])
        self.assertIn('bearish', h['reason'])

    def test_horizon_mixed_bullish_but_weak_fundamentals(self):
        """Momentum strong but fundamentals actively weak -> MIXED (a
        momentum trade, not a fundamentals-backed one)."""
        from scoring import _classify_horizon
        scores = {'value': 20, 'quality': 25, 'momentum': 80, 'dividend': 30, 'liquidity': 50}
        h = _classify_horizon(scores, {})
        self.assertEqual(h['label'], 'MIXED')

    def test_horizon_unclear_no_data(self):
        from scoring import _classify_horizon
        scores = {'value': None, 'quality': None, 'momentum': None, 'dividend': None, 'liquidity': None}
        h = _classify_horizon(scores, {})
        self.assertEqual(h['label'], 'UNCLEAR')
        self.assertIsNone(h['period'])

    def test_horizon_wired_into_score_stock(self):
        from scoring import score_stock
        fund = {'pe_ratio': 8.0, 'roe': 25.0, 'dividend_yield': 8.0, 'sector': 'Banking'}
        analysis_result = {'signals': {'overall': 'bullish', 'macd': 'bullish_cross'},
                           'latest': {'rsi': 60}}
        result = score_stock('TEST', analysis_result, fund)
        self.assertIn('horizon', result)
        self.assertIn(result['horizon']['label'], ('SHORT-TERM', 'LONG-TERM', 'MIXED', 'UNCLEAR'))


class TestHistoryTracker(unittest.TestCase):
    """Test the daily-snapshot history CSV, including sub-score columns
    (IMPROVEMENTS.txt item 2 depends on these being recorded)."""

    def test_record_and_load_round_trip(self):
        from history_tracker import HistoryTracker
        with tempfile.TemporaryDirectory() as tmp:
            ht = HistoryTracker(data_dir=tmp)
            analysis_results = {'SCOM': {'latest': {'close': 35.0, 'rsi': 55.0},
                                         'signals': {'overall': 'bullish'},
                                         'daily_change_pct': 1.2}}
            scores = {'SCOM': {'overall': 68, 'value': 55, 'quality': 70,
                              'momentum': 75, 'dividend': 60, 'liquidity': 80}}
            n = ht.record_snapshot(analysis_results, {'SCOM': {'tech_rating': 0.18}}, scores)
            self.assertEqual(n, 1)

            history = ht.load_history()
            row = history[history['symbol'] == 'SCOM'].iloc[0]
            self.assertEqual(row['score'], 68)
            self.assertEqual(row['score_value'], 55)
            self.assertEqual(row['score_quality'], 70)
            self.assertEqual(row['score_momentum'], 75)
            self.assertEqual(row['score_dividend'], 60)
            self.assertEqual(row['score_liquidity'], 80)

    def test_same_day_rerun_replaces_not_duplicates(self):
        from history_tracker import HistoryTracker
        with tempfile.TemporaryDirectory() as tmp:
            ht = HistoryTracker(data_dir=tmp)
            results = {'SCOM': {'latest': {'close': 35.0}, 'signals': {}, 'daily_change_pct': 0}}
            ht.record_snapshot(results, date='2026-01-01')
            ht.record_snapshot(results, date='2026-01-01')
            self.assertEqual(ht.days_recorded(), 1)
            self.assertEqual(len(ht.load_history()), 1)


class TestBacktest(unittest.TestCase):
    """Test the accuracy backtest module (IMPROVEMENTS.txt item 2). Pure
    computation over synthetic history -- no network, no real CSV needed."""

    @staticmethod
    def _directional_history(n_days=10):
        """AAA gains 2%/day (bullish, high score); BBB loses 2%/day
        (bearish, low score) -- deterministic, so hit rates should be
        exactly 100% and the score-bucket split should be clean."""
        rows = []
        price_a, price_b = 100.0, 100.0
        for d in range(1, n_days + 1):
            date = f'2026-01-{d:02d}'
            rows.append({'date': date, 'symbol': 'AAA', 'price': price_a,
                        'overall_signal': 'bullish', 'score': 85,
                        'score_value': 85, 'score_quality': 85, 'score_momentum': 85,
                        'score_dividend': 85, 'score_liquidity': 85})
            rows.append({'date': date, 'symbol': 'BBB', 'price': price_b,
                        'overall_signal': 'bearish', 'score': 10,
                        'score_value': 10, 'score_quality': 10, 'score_momentum': 10,
                        'score_dividend': 10, 'score_liquidity': 10})
            price_a *= 1.02
            price_b *= 0.98
        return pd.DataFrame(rows)

    @staticmethod
    def _score_proportional_history(n_days=10):
        """5 symbols whose daily return is proportional to their (constant)
        score -- score should end up strongly positively correlated with
        forward return at any horizon within range."""
        rows = []
        for score in (10, 30, 50, 70, 90):
            daily_return = (score - 50) / 1000  # score 90 -> +4%/day, 10 -> -4%/day
            price = 100.0
            sym = f'S{score}'
            signal = 'bullish' if score >= 60 else 'bearish' if score <= 40 else 'neutral'
            for d in range(1, n_days + 1):
                rows.append({'date': f'2026-01-{d:02d}', 'symbol': sym, 'price': price,
                            'overall_signal': signal, 'score': score,
                            'score_value': score, 'score_quality': score,
                            'score_momentum': score, 'score_dividend': score,
                            'score_liquidity': score})
                price *= (1 + daily_return)
        return pd.DataFrame(rows)

    def test_forward_return_computation(self):
        from backtest import _forward_return_for_horizon
        df = self._directional_history(n_days=10)
        out = _forward_return_for_horizon(df, 5)
        first_aaa = out[out['symbol'] == 'AAA'].iloc[0]
        expected = (1.02 ** 5 - 1) * 100
        self.assertAlmostEqual(first_aaa['fwd_return_5'], expected, places=6)
        # The last 5 rows of each symbol have no row 5 runs ahead yet.
        self.assertTrue(out[out['symbol'] == 'AAA']['fwd_return_5'].tail(5).isna().all())

    def test_hit_rate_perfect_directional_data(self):
        from backtest import _forward_return_for_horizon, hit_rate_by_signal
        df = self._directional_history(n_days=10)
        out = _forward_return_for_horizon(df, 5)
        hr = hit_rate_by_signal(out, 5)
        self.assertEqual(hr['bullish']['hit_rate'], 100.0)
        self.assertEqual(hr['bearish']['hit_rate'], 100.0)

    def test_score_bucket_returns(self):
        from backtest import _forward_return_for_horizon, avg_return_by_score_bucket
        df = self._directional_history(n_days=10)
        out = _forward_return_for_horizon(df, 5)
        buckets = avg_return_by_score_bucket(out, 5)
        self.assertIn('80-100', buckets)
        self.assertIn('0-20', buckets)
        self.assertGreater(buckets['80-100']['avg_return'], 0)
        self.assertLess(buckets['0-20']['avg_return'], 0)

    def test_factor_correlation_positive(self):
        from backtest import _forward_return_for_horizon, factor_correlation
        df = self._score_proportional_history(n_days=10)
        out = _forward_return_for_horizon(df, 5)
        corr = factor_correlation(out, 5)
        for factor in ('value', 'quality', 'momentum', 'dividend', 'liquidity'):
            self.assertIn(factor, corr)
            self.assertGreater(corr[factor]['correlation'], 0.9)

    def test_factor_correlation_missing_column_skipped(self):
        """A history DataFrame from before sub-scores were tracked (no
        score_value etc. columns) must not raise -- those factors are
        just absent from the result, not an error."""
        from backtest import _forward_return_for_horizon, factor_correlation
        df = self._directional_history(n_days=10).drop(columns=['score_value'])
        out = _forward_return_for_horizon(df, 5)
        corr = factor_correlation(out, 5)
        self.assertNotIn('value', corr)
        self.assertIn('quality', corr)

    def test_insufficient_history_empty(self):
        from backtest import run_backtest
        summary = run_backtest(pd.DataFrame())
        self.assertEqual(summary['days_recorded'], 0)
        for h in (5, 20, 60):
            self.assertEqual(summary['horizons'][h]['status'], 'insufficient_history')

    def test_insufficient_history_none(self):
        from backtest import run_backtest
        summary = run_backtest(None)
        self.assertEqual(summary['days_recorded'], 0)

    def test_insufficient_history_too_few_days_for_horizon(self):
        """3 days recorded -> the 5-day horizon can't have a single pair
        yet, but must degrade gracefully, not raise or divide by zero."""
        from backtest import run_backtest
        df = self._directional_history(n_days=3)
        summary = run_backtest(df, horizons=(5,))
        self.assertEqual(summary['days_recorded'], 3)
        self.assertEqual(summary['horizons'][5]['status'], 'insufficient_history')

    def test_run_backtest_ok_status_with_enough_history(self):
        from backtest import run_backtest
        df = self._directional_history(n_days=10)
        summary = run_backtest(df, horizons=(5,))
        self.assertEqual(summary['horizons'][5]['status'], 'ok')
        self.assertGreater(summary['horizons'][5]['n_pairs'], 0)
        self.assertIn('hit_rate', summary['horizons'][5])
        self.assertIn('score_buckets', summary['horizons'][5])
        self.assertIn('factor_correlation', summary['horizons'][5])


class TestReportGenerator(unittest.TestCase):
    """Test report generation."""

    @classmethod
    def setUpClass(cls):
        from report_generator import ReportGenerator
        cls.engine = AnalysisEngine()
        cls.tempdir = tempfile.mkdtemp()
        cls.rg = ReportGenerator(output_dir=cls.tempdir)

    def test_stock_report_html(self):
        data = make_sample_data(60)
        result = self.engine.analyze_stock(data)
        path = self.rg.generate_stock_report('TEST', result, report_type='html')
        self.assertIsNotNone(path)
        self.assertTrue(os.path.exists(path))
        self.assertGreater(os.path.getsize(path), 500)

    def test_market_summary_html(self):
        data_dict = {
            'SCOM': make_sample_data(60, seed=1),
            'EQTY': make_sample_data(60, seed=2),
        }
        results = self.engine.analyze_multiple_stocks(data_dict)
        path = self.rg.generate_market_summary(results, report_type='html')
        self.assertIsNotNone(path)
        self.assertTrue(os.path.exists(path))
        self.assertGreater(os.path.getsize(path), 500)

    def test_excel_export(self):
        data_dict = {
            'SCOM': make_sample_data(60, seed=1),
            'EQTY': make_sample_data(60, seed=2),
        }
        results = self.engine.analyze_multiple_stocks(data_dict)
        path = self.rg.export_to_excel(results)
        self.assertIsNotNone(path)
        self.assertTrue(os.path.exists(path))
        self.assertGreater(os.path.getsize(path), 1000)


class TestEmailNotifier(unittest.TestCase):
    """Test email notification module."""

    def test_email_body_generation(self):
        from email_notifier import EmailNotifier
        config = Config()
        notifier = EmailNotifier(config)

        engine = AnalysisEngine()
        data_dict = {
            'SCOM': make_sample_data(60, seed=1),
            'EQTY': make_sample_data(60, seed=2),
        }
        results = engine.analyze_multiple_stocks(data_dict)
        breadth = engine.calculate_market_breadth(results)

        from sector_analysis import SectorAnalyzer
        sa = SectorAnalyzer()
        sectors = sa.analyze_sectors(data_dict, results)

        body = notifier.generate_email_body(results, sectors, breadth)
        self.assertIsInstance(body, str)
        self.assertIn('SCOM', body)
        self.assertIn('EQTY', body)
        self.assertIn('NSE Daily Market Report', body)


class TestTelegramNotifier(unittest.TestCase):
    """Test Telegram notification module. Never hits the real Bot API."""

    def _sample_summary_inputs(self):
        engine = AnalysisEngine()
        data_dict = {
            'SCOM': make_sample_data(60, seed=1),
            'EQTY': make_sample_data(60, seed=2),
        }
        results = engine.analyze_multiple_stocks(data_dict)
        breadth = engine.calculate_market_breadth(results)

        from sector_analysis import SectorAnalyzer
        sectors = SectorAnalyzer().analyze_sectors(data_dict, results)
        return results, sectors, breadth

    def test_summary_text_generation(self):
        from telegram_notifier import TelegramNotifier
        config = Config()
        notifier = TelegramNotifier(config)

        results, sectors, breadth = self._sample_summary_inputs()
        text = notifier.generate_summary_text(results, sectors, breadth, checkpoint='close')
        self.assertIsInstance(text, str)
        self.assertIn('SCOM', text)
        self.assertIn('EQTY', text)
        self.assertIn('Market Close', text)

    def test_summary_text_includes_alerts(self):
        from telegram_notifier import TelegramNotifier
        config = Config()
        notifier = TelegramNotifier(config)

        results, sectors, breadth = self._sample_summary_inputs()
        alerts = {'SCOM': ['🟢 Oversold (RSI 25)'], 'EQTY': []}
        text = notifier.generate_summary_text(results, checkpoint='mid', alerts=alerts)
        self.assertIn('Alerts', text)
        self.assertIn('Oversold (RSI 25)', text)

    def test_send_message_success(self):
        from telegram_notifier import TelegramNotifier
        config = Config()
        config.telegram_bot_token = 'test-token'
        config.telegram_chat_ids = ['12345']
        notifier = TelegramNotifier(config)

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'ok': True}

        with patch('telegram_notifier.requests.post', return_value=mock_response) as mock_post:
            result = notifier.send_message('Test message')

        self.assertTrue(result)
        mock_post.assert_called_once()

    def test_send_message_api_error(self):
        """Failure mode: Telegram returns a non-ok response -- must degrade to False."""
        from telegram_notifier import TelegramNotifier
        config = Config()
        config.telegram_bot_token = 'test-token'
        config.telegram_chat_ids = ['12345']
        notifier = TelegramNotifier(config)

        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.json.return_value = {'ok': False, 'description': 'bad chat id'}
        mock_response.text = '{"ok": false, "description": "bad chat id"}'

        with patch('telegram_notifier.requests.post', return_value=mock_response):
            result = notifier.send_message('Test message')

        self.assertFalse(result)

    def test_send_message_network_failure(self):
        """Failure mode: a network error must degrade to False, not raise."""
        from telegram_notifier import TelegramNotifier
        config = Config()
        config.telegram_bot_token = 'test-token'
        config.telegram_chat_ids = ['12345']
        notifier = TelegramNotifier(config)

        with patch('telegram_notifier.requests.post',
                   side_effect=requests.exceptions.ConnectionError('boom')):
            result = notifier.send_message('Test message')

        self.assertFalse(result)

    def test_send_message_not_configured(self):
        """Failure mode: missing bot token/chat ID must fail safe, not raise."""
        from telegram_notifier import TelegramNotifier
        config = Config()
        config.telegram_bot_token = ''
        config.telegram_chat_ids = []
        notifier = TelegramNotifier(config)

        result = notifier.send_message('Test message')
        self.assertFalse(result)

    def test_send_document_not_configured(self):
        """Failure mode: send_document must also fail safe when unconfigured."""
        from telegram_notifier import TelegramNotifier
        config = Config()
        config.telegram_bot_token = ''
        config.telegram_chat_ids = []
        notifier = TelegramNotifier(config)

        result = notifier.send_document('/nonexistent/path.pdf')
        self.assertFalse(result)


def run_tests():
    """Run all tests and print results."""
    print("=" * 60)
    print("KENYAN STOCK ANALYZER — TEST SUITE")
    print("=" * 60)
    print()

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestUtils))
    suite.addTests(loader.loadTestsFromTestCase(TestAnalysisEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestSectorAnalysis))
    suite.addTests(loader.loadTestsFromTestCase(TestScoring))
    suite.addTests(loader.loadTestsFromTestCase(TestHistoryTracker))
    suite.addTests(loader.loadTestsFromTestCase(TestBacktest))
    suite.addTests(loader.loadTestsFromTestCase(TestReportGenerator))
    suite.addTests(loader.loadTestsFromTestCase(TestEmailNotifier))
    suite.addTests(loader.loadTestsFromTestCase(TestTelegramNotifier))

    # Run
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print()
    print("=" * 60)
    if result.wasSuccessful():
        print("ALL TESTS PASSED!")
    else:
        print(f"FAILURES: {len(result.failures)}, ERRORS: {len(result.errors)}")
    print("=" * 60)

    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())