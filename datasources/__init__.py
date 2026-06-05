"""
Módulo de datasources - Centraliza todas as conexões com APIs externas.
"""

from datasources.binance_source import BinanceDataSource
from datasources.yfinance_source import YFinanceDataSource
from datasources.metatrader_source import MetaTraderDataSource
from datasources.base_source import BaseDataSource

__all__ = [
    "BinanceDataSource",
    "YFinanceDataSource",
    "MetaTraderDataSource",
    "BaseDataSource",
]
