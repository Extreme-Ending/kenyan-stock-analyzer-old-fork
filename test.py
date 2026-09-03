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


class TestOutlierFiltering(unittest.TestCase):
    """Test OHLCV outlier/bad-tick filtering (IMPROVEMENTS.txt item 7).
    Pure computation over synthetic data -- no network."""

    @staticmethod
    def _clean_series(days=30, base=100.0, daily_move=0.3, seed=1):
        """A calm, low-volatility price series with steady volume."""
        np.random.seed(seed)
        dates = pd.date_range('2026-01-01', periods=days, freq='B')
        closes = base + np.cumsum(np.random.uniform(-daily_move, daily_move, days))
        return pd.DataFrame({
            'open': closes, 'high': closes + 0.5, 'low': closes - 0.5,
            'close': closes, 'volume': 100_000,
        }, index=dates)

    def test_no_change_on_clean_data(self):
        from utils import filter_ohlcv_outliers
        df = self._clean_series()
        clean, dropped = filter_ohlcv_outliers(df)
        self.assertEqual(dropped, 0)
        self.assertEqual(len(clean), len(df))

    def test_drops_high_below_low(self):
        from utils import filter_ohlcv_outliers
        df = self._clean_series()
        df.iloc[10, df.columns.get_loc('high')] = df.iloc[10]['low'] - 1
        clean, dropped = filter_ohlcv_outliers(df)
        self.assertEqual(dropped, 1)
        self.assertEqual(len(clean), len(df) - 1)

    def test_drops_close_outside_range(self):
        from utils import filter_ohlcv_outliers
        df = self._clean_series()
        df.iloc[10, df.columns.get_loc('close')] = df.iloc[10]['high'] + 50
        clean, dropped = filter_ohlcv_outliers(df)
        self.assertEqual(dropped, 1)

    def test_drops_open_outside_range(self):
        from utils import filter_ohlcv_outliers
        df = self._clean_series()
        df.iloc[10, df.columns.get_loc('open')] = df.iloc[10]['low'] - 50
        clean, dropped = filter_ohlcv_outliers(df)
        self.assertEqual(dropped, 1)

    def test_drops_extreme_move_without_volume_spike(self):
        from utils import filter_ohlcv_outliers
        df = self._clean_series(days=30)
        idx = 25
        bad_price = df.iloc[idx]['close'] * 10
        for col in ('open', 'close'):
            df.iloc[idx, df.columns.get_loc(col)] = bad_price
        df.iloc[idx, df.columns.get_loc('high')] = bad_price + 1
        df.iloc[idx, df.columns.get_loc('low')] = bad_price - 1
        # volume stays at the normal 100_000 -- no matching spike
        clean, dropped = filter_ohlcv_outliers(df)
        self.assertGreaterEqual(dropped, 1)
        self.assertNotIn(df.index[idx], clean.index)

    def test_keeps_extreme_move_with_matching_volume_spike(self):
        """A huge move on huge volume is a real event (earnings, corporate
        action) -- must be kept, not dropped."""
        from utils import filter_ohlcv_outliers
        df = self._clean_series(days=30)
        idx = 25
        bad_price = df.iloc[idx]['close'] * 10
        for col in ('open', 'close'):
            df.iloc[idx, df.columns.get_loc(col)] = bad_price
        df.iloc[idx, df.columns.get_loc('high')] = bad_price + 1
        df.iloc[idx, df.columns.get_loc('low')] = bad_price - 1
        df.iloc[idx, df.columns.get_loc('volume')] = 100_000 * 10
        clean, dropped = filter_ohlcv_outliers(df)
        self.assertIn(df.index[idx], clean.index)

    def test_persisting_real_move_not_dropped_even_if_statistically_extreme(self):
        """Regression case found against real NSE data (EQTY, 2026-05-25):
        an ordinary-looking ~6.6% drop after an unusually calm month
        registered as an 8-sigma 'anomaly' on the sigma+volume check alone,
        but the price never reverted -- it held at the new level, which is
        the signature of a real repricing, not a bad print. Requiring
        reversion is what tells these apart; a persisting move must never
        be dropped just for being statistically large."""
        from utils import filter_ohlcv_outliers
        # An unusually calm run (tiny, near-constant daily moves) so the
        # rolling std is small enough that an ordinary ~6% move registers
        # as many sigma -- exactly the real-world setup that caused this.
        dates = pd.date_range('2026-01-01', periods=30, freq='B')
        closes = [100.0 + 0.01 * i for i in range(25)]  # dead calm
        closes += [93.4, 93.4, 93.4, 93.4, 93.4]  # real ~6.6% drop, then holds
        df = pd.DataFrame({
            'open': closes, 'high': [c + 0.3 for c in closes],
            'low': [c - 0.3 for c in closes], 'close': closes,
            'volume': 100_000,
        }, index=dates)
        clean, dropped = filter_ohlcv_outliers(df)
        self.assertEqual(dropped, 0)
        self.assertEqual(len(clean), len(df))

    def test_short_history_skips_statistical_check(self):
        """Fewer than 15 bars -- can't judge 'normal' volatility yet, so
        the statistical check is skipped; an internally-consistent (if
        large) bar is left alone rather than guessed at."""
        from utils import filter_ohlcv_outliers
        df = self._clean_series(days=10)
        idx = 5
        for col in ('open', 'high', 'low', 'close'):
            df.iloc[idx, df.columns.get_loc(col)] = df.iloc[idx][col] * 50
        clean, dropped = filter_ohlcv_outliers(df)
        self.assertEqual(dropped, 0)
        self.assertEqual(len(clean), len(df))

    def test_empty_and_none_input(self):
        from utils import filter_ohlcv_outliers
        clean, dropped = filter_ohlcv_outliers(None)
        self.assertIsNone(clean)
        self.assertEqual(dropped, 0)
        clean, dropped = filter_ohlcv_outliers(pd.DataFrame())
        self.assertTrue(clean.empty)
        self.assertEqual(dropped, 0)

    def test_malformed_dataframe_fails_safe(self):
        """Missing required columns must not raise -- returns the original
        frame unchanged rather than corrupting or crashing."""
        from utils import filter_ohlcv_outliers
        df = pd.DataFrame({'close': [1, 2, 3]})
        clean, dropped = filter_ohlcv_outliers(df)
        self.assertEqual(dropped, 0)
        pd.testing.assert_frame_equal(clean, df)


class TestDataAcquisition(unittest.TestCase):
    """Test the outlier-filter wiring into the fetch path (the
    data_acquisition.py half of IMPROVEMENTS.txt item 7). Mocks the
    source fetch -- never hits a real network source."""

    def test_fetch_stock_data_filters_outliers_before_caching(self):
        from data_acquisition import DataAcquisition
        with tempfile.TemporaryDirectory() as tmp:
            da = DataAcquisition(data_sources=['tradingview'], cache_dir=tmp)
            np.random.seed(1)
            dates = pd.date_range('2026-01-01', periods=20, freq='B')
            closes = 100 + np.cumsum(np.random.uniform(-0.3, 0.3, 20))
            df = pd.DataFrame({
                'open': closes, 'high': closes + 0.5, 'low': closes - 0.5,
                'close': closes, 'volume': 100_000,
            }, index=dates)
            df.iloc[5, df.columns.get_loc('high')] = df.iloc[5]['low'] - 1  # bad bar

            with patch.object(da, '_fetch_from_source', return_value=df):
                result = da.fetch_stock_data('TEST', force_refresh=True)

            self.assertEqual(len(result), 19)


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

    def test_overall_signal_tie_is_neutral_regardless_of_rsi(self):
        """IMPROVEMENTS.txt item 4: RSI must no longer break a tie among
        ma_crossover/macd/trend for `overall` -- that coupling let
        scoring.py's _score_momentum() double-count RSI (once via
        `overall`, once as its own raw-RSI input). A genuine tie (one
        bullish, one bearish, one undefined) must land on 'neutral' even
        with an extreme RSI on either side, not flip with it."""
        df = pd.DataFrame({
            'sma_20': [104, 105], 'sma_50': [100, 100],
            'close': [96, 95],
            'macd': [np.nan, np.nan], 'macd_signal': [np.nan, np.nan],
            'bb_upper': [110, 110], 'bb_lower': [90, 90],
            'stoch_k': [50, 50], 'volume': [1000, 1000], 'volume_sma_20': [1000, 1000],
            'rsi': [80, 85],  # overbought -- old code would have forced 'bearish'
        })
        signals = self.engine._generate_signals(df)
        self.assertEqual(signals['ma_crossover'], 'bullish')
        self.assertEqual(signals['trend'], 'bearish')
        self.assertEqual(signals['macd'], 'undefined')
        self.assertEqual(signals['rsi'], 'overbought')
        self.assertEqual(signals['overall'], 'neutral')

        df['rsi'] = [25, 20]  # oversold -- old code would have forced 'bullish'
        signals = self.engine._generate_signals(df)
        self.assertEqual(signals['rsi'], 'oversold')
        self.assertEqual(signals['overall'], 'neutral')

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


class TestFundamentalAnalysis(unittest.TestCase):
    """Test the fundamentals sanity/bounds check (IMPROVEMENTS.txt item 6).
    Pure computation over synthetic dicts -- no network."""

    def test_plausible_data_untouched(self):
        from fundamental_analysis import sanity_check_fundamentals
        fund = {'revenue_ttm': 5_000_000, 'market_cap': 50_000_000,
               'dividend_yield': 4.5, 'current_ratio': 1.8, 'roe': 22.0,
               'eps_ttm': 3.5, 'net_income_ttm': 1_200_000}
        original = dict(fund)
        flags = sanity_check_fundamentals(fund)
        self.assertEqual(flags, [])
        self.assertEqual(fund, original)
        self.assertNotIn('_sanity_flags', fund)

    def test_rejects_negative_revenue(self):
        from fundamental_analysis import sanity_check_fundamentals
        fund = {'revenue_ttm': -100}
        flags = sanity_check_fundamentals(fund)
        self.assertIsNone(fund['revenue_ttm'])
        self.assertEqual(len(flags), 1)

    def test_rejects_negative_market_cap_yield_current_ratio(self):
        from fundamental_analysis import sanity_check_fundamentals
        fund = {'market_cap': -1, 'dividend_yield': -2.0, 'current_ratio': -0.5}
        flags = sanity_check_fundamentals(fund)
        self.assertIsNone(fund['market_cap'])
        self.assertIsNone(fund['dividend_yield'])
        self.assertIsNone(fund['current_ratio'])
        self.assertEqual(len(flags), 3)

    def test_rejects_implausible_roe(self):
        from fundamental_analysis import sanity_check_fundamentals
        fund = {'roe': 450.0}
        flags = sanity_check_fundamentals(fund)
        self.assertIsNone(fund['roe'])
        self.assertEqual(len(flags), 1)

    def test_plausible_roe_not_rejected(self):
        from fundamental_analysis import sanity_check_fundamentals
        fund = {'roe': -60.0}  # a real, if bad, ROE -- well within bounds
        flags = sanity_check_fundamentals(fund)
        self.assertEqual(fund['roe'], -60.0)
        self.assertEqual(flags, [])

    def test_rejects_eps_net_income_sign_mismatch(self):
        from fundamental_analysis import sanity_check_fundamentals
        fund = {'eps_ttm': 2.5, 'net_income_ttm': -500_000}
        flags = sanity_check_fundamentals(fund)
        self.assertIsNone(fund['eps_ttm'])
        self.assertIsNone(fund['net_income_ttm'])
        self.assertEqual(len(flags), 2)

    def test_agreeing_signs_not_rejected(self):
        from fundamental_analysis import sanity_check_fundamentals
        fund = {'eps_ttm': -1.2, 'net_income_ttm': -300_000}  # both negative -- consistent
        flags = sanity_check_fundamentals(fund)
        self.assertEqual(fund['eps_ttm'], -1.2)
        self.assertEqual(fund['net_income_ttm'], -300_000)
        self.assertEqual(flags, [])

    def test_zero_values_do_not_false_trigger_sign_check(self):
        """0 is neither positive nor negative -- must not be treated as a
        sign mismatch against a nonzero value on the other field."""
        from fundamental_analysis import sanity_check_fundamentals
        fund = {'eps_ttm': 0, 'net_income_ttm': 500_000}
        flags = sanity_check_fundamentals(fund)
        self.assertEqual(flags, [])

    def test_missing_fields_skipped_gracefully(self):
        from fundamental_analysis import sanity_check_fundamentals
        flags = sanity_check_fundamentals({})
        self.assertEqual(flags, [])


class TestPriceValidation(unittest.TestCase):
    """Test price validation: liquidity-scaled thresholds and the NSE-PDF
    fallback reference source (IMPROVEMENTS.txt item 5). Never hits afx.
    kwayisi.org or the real NSE PDF -- all network boundaries are mocked."""

    @staticmethod
    def _volume_history(avg_close=100.0, avg_volume=0, days=20):
        """OHLCV frame with a fixed close and volume, for a known avg value
        traded = avg_close * avg_volume. Ends at today so the freshness
        check (unrelated to what these tests target) never flags it stale."""
        dates = pd.date_range(end=pd.Timestamp.now().normalize(), periods=days, freq='B')
        return pd.DataFrame({
            'open': avg_close, 'high': avg_close, 'low': avg_close,
            'close': avg_close, 'volume': avg_volume,
        }, index=dates)

    def test_avg_value_traded_computation(self):
        from price_validation import PriceValidator
        df = self._volume_history(avg_close=50.0, avg_volume=200_000)
        result = PriceValidator._avg_value_traded(df)
        self.assertAlmostEqual(result, 50.0 * 200_000, places=2)

    def test_avg_value_traded_no_volume_column(self):
        from price_validation import PriceValidator
        df = pd.DataFrame({'close': [10, 11, 12]})
        self.assertIsNone(PriceValidator._avg_value_traded(df))

    def test_avg_value_traded_all_zero_volume(self):
        from price_validation import PriceValidator
        df = self._volume_history(avg_volume=0)
        self.assertIsNone(PriceValidator._avg_value_traded(df))

    def test_avg_value_traded_empty_or_none(self):
        from price_validation import PriceValidator
        self.assertIsNone(PriceValidator._avg_value_traded(None))
        self.assertIsNone(PriceValidator._avg_value_traded(pd.DataFrame()))

    def test_liquidity_multiplier_tiers(self):
        from price_validation import PriceValidator
        self.assertEqual(PriceValidator._liquidity_multiplier(50_000_000), 1.0)
        self.assertEqual(PriceValidator._liquidity_multiplier(10_000_000), 1.0)
        self.assertEqual(PriceValidator._liquidity_multiplier(5_000_000), 2.0)
        self.assertEqual(PriceValidator._liquidity_multiplier(1_000_000), 2.0)
        self.assertEqual(PriceValidator._liquidity_multiplier(500_000), 3.0)
        self.assertEqual(PriceValidator._liquidity_multiplier(100_000), 3.0)
        self.assertEqual(PriceValidator._liquidity_multiplier(50_000), 5.0)
        self.assertEqual(PriceValidator._liquidity_multiplier(0), 5.0)
        self.assertEqual(PriceValidator._liquidity_multiplier(None), 5.0)

    def test_validate_same_pct_diff_liquid_flags_illiquid_does_not(self):
        """The whole point of item 5: a flat threshold would flag both of
        these identically (same 2% diff) -- liquidity-scaling must only
        flag the liquid one."""
        from price_validation import PriceValidator
        with tempfile.TemporaryDirectory() as tmp:
            pv = PriceValidator(cache_dir=tmp, disagree_threshold_pct=1.0)
            pv._reference = {'AAA': {'price': 100.0, 'volume': 1000, 'change': None}}
            pv._reference_source = 'afx.kwayisi.org'

            liquid_history = self._volume_history(avg_close=100.0, avg_volume=500_000)  # 50M/day
            illiquid_history = self._volume_history(avg_close=100.0, avg_volume=500)     # 50K/day

            liquid_result = pv.validate('AAA', 102.0, liquid_history)   # 2% diff
            illiquid_result = pv.validate('AAA', 102.0, illiquid_history)  # same 2% diff

            self.assertEqual(liquid_result['status'], 'mismatch')
            self.assertEqual(liquid_result['threshold_pct'], 1.0)
            self.assertEqual(illiquid_result['status'], 'ok')
            self.assertEqual(illiquid_result['threshold_pct'], 5.0)

    def test_validate_unverified_when_no_reference(self):
        from price_validation import PriceValidator
        with tempfile.TemporaryDirectory() as tmp:
            pv = PriceValidator(cache_dir=tmp)
            pv._reference = {}
            pv._reference_source = None
            result = pv.validate('ZZZ', 50.0)
            self.assertEqual(result['status'], 'unverified')
            self.assertIsNone(result['reference_price'])

    def test_fetch_reference_prices_falls_back_to_pdf_when_afx_fails(self):
        from price_validation import PriceValidator, PDF_SOURCE
        with tempfile.TemporaryDirectory() as tmp:
            pv = PriceValidator(cache_dir=tmp)
            with patch.object(pv, '_fetch_afx_reference', return_value={}), \
                 patch.object(pv, '_fetch_pdf_reference',
                              return_value={'AAA': {'price': 10.0, 'volume': 100, 'change': None}}) as mock_pdf:
                ref = pv.fetch_reference_prices()
            mock_pdf.assert_called_once()
            self.assertEqual(ref, {'AAA': {'price': 10.0, 'volume': 100, 'change': None}})
            self.assertEqual(pv._reference_source, PDF_SOURCE)

    def test_fetch_reference_prices_skips_pdf_when_afx_succeeds(self):
        """Never pay for the (slow, OCR-based) PDF fallback when the
        primary source already answered."""
        from price_validation import PriceValidator, AFX_SOURCE
        with tempfile.TemporaryDirectory() as tmp:
            pv = PriceValidator(cache_dir=tmp)
            with patch.object(pv, '_fetch_afx_reference',
                              return_value={'AAA': {'price': 10.0, 'volume': 100, 'change': None}}), \
                 patch.object(pv, '_fetch_pdf_reference') as mock_pdf:
                pv.fetch_reference_prices()
            mock_pdf.assert_not_called()
            self.assertEqual(pv._reference_source, AFX_SOURCE)

    def test_fetch_reference_prices_both_sources_fail(self):
        """Failure mode: neither source answers -- must degrade to {} /
        'unverified' downstream, never raise."""
        from price_validation import PriceValidator
        with tempfile.TemporaryDirectory() as tmp:
            pv = PriceValidator(cache_dir=tmp)
            with patch.object(pv, '_fetch_afx_reference', return_value={}), \
                 patch.object(pv, '_fetch_pdf_reference', return_value={}):
                ref = pv.fetch_reference_prices()
            self.assertEqual(ref, {})
            self.assertIsNone(pv._reference_source)

    def test_fetch_pdf_reference_reshapes_board(self):
        """The PDF board's {open,high,low,close,volume} shape must map
        close->price, volume->volume, and have no 'change' (not in the PDF)."""
        from price_validation import PriceValidator
        with tempfile.TemporaryDirectory() as tmp:
            pv = PriceValidator(cache_dir=tmp)
            fake_board = {'AAA': {'open': 9.5, 'high': 10.5, 'low': 9.0, 'close': 10.0, 'volume': 5000}}
            with patch('data_acquisition.DataAcquisition.get_pdf_price_board', return_value=fake_board):
                result = pv._fetch_pdf_reference()
            self.assertEqual(result, {'AAA': {'price': 10.0, 'volume': 5000, 'change': None}})


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
    suite.addTests(loader.loadTestsFromTestCase(TestOutlierFiltering))
    suite.addTests(loader.loadTestsFromTestCase(TestDataAcquisition))
    suite.addTests(loader.loadTestsFromTestCase(TestAnalysisEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestSectorAnalysis))
    suite.addTests(loader.loadTestsFromTestCase(TestScoring))
    suite.addTests(loader.loadTestsFromTestCase(TestHistoryTracker))
    suite.addTests(loader.loadTestsFromTestCase(TestBacktest))
    suite.addTests(loader.loadTestsFromTestCase(TestFundamentalAnalysis))
    suite.addTests(loader.loadTestsFromTestCase(TestPriceValidation))
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