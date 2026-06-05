"""
Módulo de Indicadores Técnicos - Cálculo de médias móveis e ADX
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class IndicadoresTA:
    """
    Classe para cálculo de indicadores técnicos.
    
    Suporta:
    - Médias Móveis Exponenciais (EMA)
    - Médias Móveis Simples (SMA)
    - Average Directional Index (ADX)
    """
    
    @staticmethod
    def ema(series: pd.Series, periodo: int) -> pd.Series:
        """
        Calcula Média Móvel Exponencial (EMA).
        
        Args:
            series: Série de preços (close)
            periodo: Número de períodos
        
        Returns:
            Series com EMA calculada
        """
        return series.ewm(span=periodo, adjust=False).mean()
    
    @staticmethod
    def sma(series: pd.Series, periodo: int) -> pd.Series:
        """
        Calcula Média Móvel Simples (SMA).
        
        Args:
            series: Série de preços
            periodo: Número de períodos
        
        Returns:
            Series com SMA calculada
        """
        return series.rolling(window=periodo).mean()
    
    @staticmethod
    def calcular_adx(df: pd.DataFrame, periodo: int = 14) -> pd.DataFrame:
        """
        Calcula o Average Directional Index (ADX).
        
        ADX mede a força de uma tendência (0-100).
        - Valores > 25: tendência forte
        - Valores < 20: tendência fraca/lateral
        
        Args:
            df: DataFrame com colunas [high, low, close]
            periodo: Período do ADX (default 14)
        
        Returns:
            DataFrame com colunas: +DI, -DI, ADX
        """
        try:
            df = df.copy()
            
            # Calcular True Range
            df['hl'] = df['high'] - df['low']
            df['hc'] = abs(df['high'] - df['close'].shift(1))
            df['lc'] = abs(df['low'] - df['close'].shift(1))
            df['tr'] = df[['hl', 'hc', 'lc']].max(axis=1)
            
            # Calcular Directional Movement
            df['up'] = df['high'].diff()
            df['down'] = -df['low'].diff()
            
            df['pos_dm'] = np.where((df['up'] > df['down']) & (df['up'] > 0), df['up'], 0)
            df['neg_dm'] = np.where((df['down'] > df['up']) & (df['down'] > 0), df['down'], 0)
            
            # Suavizar com média móvel (14 períodos default)
            df['tr_sum'] = df['tr'].rolling(window=periodo).sum()
            df['pos_dm_sum'] = df['pos_dm'].rolling(window=periodo).sum()
            df['neg_dm_sum'] = df['neg_dm'].rolling(window=periodo).sum()
            
            # Calcular Directional Indicators
            df['+DI'] = 100 * (df['pos_dm_sum'] / df['tr_sum'])
            df['-DI'] = 100 * (df['neg_dm_sum'] / df['tr_sum'])
            
            # Calcular DX
            dx = 100 * abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])
            
            # Calcular ADX (média móvel suavizada do DX)
            df['ADX'] = dx.rolling(window=periodo).mean()
            
            # Limpar colunas auxiliares
            cols_drop = ['hl', 'hc', 'lc', 'tr', 'up', 'down', 
                        'pos_dm', 'neg_dm', 'tr_sum', 'pos_dm_sum', 'neg_dm_sum']
            df = df.drop(columns=cols_drop, errors='ignore')
            
            return df[['+DI', '-DI', 'ADX']]
            
        except Exception as e:
            logger.error(f"Erro ao calcular ADX: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def adicionar_indicadores(df: pd.DataFrame, ema_rapida: int = 9, 
                             ema_lenta: int = 21, adx_periodo: int = 14) -> pd.DataFrame:
        """
        Adiciona todos os indicadores necessários ao DataFrame.
        
        Args:
            df: DataFrame com OHLCV
            ema_rapida: Período da EMA rápida (default 9)
            ema_lenta: Período da EMA lenta (default 21)
            adx_periodo: Período do ADX (default 14)
        
        Returns:
            DataFrame com indicadores adicionados
        """
        try:
            df = df.copy()
            
            # Adicionar EMAs
            df['EMA_9'] = IndicadoresTA.ema(df['close'], ema_rapida)
            df['EMA_21'] = IndicadoresTA.ema(df['close'], ema_lenta)
            
            # Adicionar ADX
            adx_df = IndicadoresTA.calcular_adx(df, adx_periodo)
            df['+DI'] = adx_df['+DI']
            df['-DI'] = adx_df['-DI']
            df['ADX'] = adx_df['ADX']
            
            return df
            
        except Exception as e:
            logger.error(f"Erro ao adicionar indicadores: {e}")
            return df
