"""
Módulo de Estratégia de Trading - Lógica de sinais de compra/venda
"""

import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass
from typing import Optional
from enum import Enum

logger = logging.getLogger(__name__)


class Sinal(Enum):
    """Enumeração de sinais possíveis."""
    COMPRA = "COMPRA"
    VENDA = "VENDA"
    AGUARDAR = "AGUARDAR"


@dataclass
class ConfiguracaoEstrategia:
    """Configuração dos parâmetros da estratégia."""
    
    # Gestão de risco
    percentual_operacao: float = 0.05  # 5% do saldo
    stop_loss_pct: float = 0.015       # 1.5% stop loss
    take_profit_pct: float = 0.03      # 3% take profit
    
    # Filtro de tendência
    adx_minimo: float = 25.0           # ADX > 25 para tendência forte
    
    # Indicadores
    ema_rapida: int = 9
    ema_lenta: int = 21
    adx_periodo: int = 14
    
    def __post_init__(self):
        """Valida os parâmetros."""
        if not (0 < self.percentual_operacao <= 1):
            raise ValueError("percentual_operacao deve estar entre 0 e 1")
        if self.stop_loss_pct >= self.take_profit_pct:
            raise ValueError("stop_loss_pct deve ser menor que take_profit_pct")


@dataclass
class OrdemTrade:
    """Representa uma ordem de trading."""
    
    acao: str              # COMPRA ou VENDA
    entrada: float         # Preço de entrada
    quantidade: float      # Quantidade operada
    stop_loss: float       # Preço de stop loss
    take_profit: float     # Preço de take profit
    timestamp: Optional[str] = None  # Timestamp da operação
    sinal_confirmado: bool = False   # Se há confirmação de sinal


class Estrategia:
    """
    Estratégia de trading baseada em cruzamento de EMAs com filtro ADX.
    
    Lógica:
    1. Compra quando EMA 9 cruza acima de EMA 21 E ADX > 25
    2. Vende quando EMA 9 cruza abaixo de EMA 21
    3. Stop Loss em -1.5%, Take Profit em +3%
    """
    
    def __init__(self, config: Optional[ConfiguracaoEstrategia] = None):
        """
        Inicializa a estratégia.
        
        Args:
            config: ConfiguracaoEstrategia com parâmetros customizados
        """
        self.config = config or ConfiguracaoEstrategia()
        logger.info(f"Estratégia inicializada com config: {self.config}")
    
    def gerar_sinal(self, df: pd.DataFrame) -> Sinal:
        """
        Gera sinal de compra/venda baseado em cruzamento de EMAs.
        
        Args:
            df: DataFrame com indicadores calculados (EMA_9, EMA_21)
        
        Returns:
            Sinal.COMPRA, Sinal.VENDA ou Sinal.AGUARDAR
        
        Raises:
            ValueError: Se dados insuficientes ou colunas faltando
        """
        if len(df) < 2:
            raise ValueError("DataFrame deve ter pelo menos 2 linhas")
        
        colunas_requeridas = ['EMA_9', 'EMA_21']
        if not all(col in df.columns for col in colunas_requeridas):
            raise ValueError(f"DataFrame deve conter colunas: {colunas_requeridas}")
        
        # Pegar última e penúltima linhas
        atual = df.iloc[-1]
        anterior = df.iloc[-2]
        
        # Validar NaN
        if pd.isna(atual['EMA_9']) or pd.isna(atual['EMA_21']):
            return Sinal.AGUARDAR
        
        # Cruzamento EMA 9 acima de EMA 21 (COMPRA)
        compra = (
            anterior['EMA_9'] <= anterior['EMA_21'] and
            atual['EMA_9'] > atual['EMA_21']
        )
        
        # Cruzamento EMA 9 abaixo de EMA 21 (VENDA)
        venda = (
            anterior['EMA_9'] >= anterior['EMA_21'] and
            atual['EMA_9'] < atual['EMA_21']
        )
        
        if compra:
            return Sinal.COMPRA
        elif venda:
            return Sinal.VENDA
        else:
            return Sinal.AGUARDAR
    
    def validar_tendencia(self, df: pd.DataFrame) -> bool:
        """
        Valida se há tendência forte usando ADX.
        
        Args:
            df: DataFrame com indicador ADX
        
        Returns:
            True se ADX > ADX_MINIMO, False caso contrário
        """
        if 'ADX' not in df.columns or len(df) == 0:
            logger.warning("Coluna ADX não encontrada ou DataFrame vazio")
            return True  # Assume que é válido se não conseguir validar
        
        adx_atual = df.iloc[-1]['ADX']
        
        if pd.isna(adx_atual):
            return True  # Ainda não calculado, assume válido
        
        validado = adx_atual > self.config.adx_minimo
        
        if not validado:
            logger.info(f"Tendência fraca: ADX = {adx_atual:.2f} (mínimo: {self.config.adx_minimo})")
        
        return validado
    
    def calcular_quantidade(self, saldo: float, preco: float) -> float:
        """
        Calcula quantidade de ativos a operar.
        
        Aloca percentual_operacao% do saldo em caixa.
        
        Args:
            saldo: Saldo disponível
            preco: Preço atual do ativo
        
        Returns:
            Quantidade a operar (arredondada a 6 casas decimais)
        """
        if saldo <= 0 or preco <= 0:
            logger.error(f"Saldo ou preço inválido: saldo={saldo}, preço={preco}")
            return 0.0
        
        valor_operacao = saldo * self.config.percentual_operacao
        quantidade = valor_operacao / preco
        
        return round(quantidade, 6)
    
    def gerar_ordem(self, df: pd.DataFrame, saldo: float, 
                   forcar_sinal: Optional[Sinal] = None) -> Optional[OrdemTrade]:
        """
        Gera ordem de trading baseada na estratégia.
        
        Args:
            df: DataFrame com indicadores
            saldo: Saldo disponível
            forcar_sinal: Força um sinal específico (para testes)
        
        Returns:
            OrdemTrade se há sinal válido, None caso contrário
        """
        # Gerar sinal
        sinal = forcar_sinal or self.gerar_sinal(df)
        
        if sinal == Sinal.AGUARDAR:
            return None
        
        # Validar tendência (apenas para COMPRA)
        if sinal == Sinal.COMPRA and not self.validar_tendencia(df):
            logger.info("Sinal de COMPRA ignorado: tendência fraca")
            return None
        
        # Obter preço de entrada
        preco_entrada = float(df.iloc[-1]['close'])
        quantidade = self.calcular_quantidade(saldo, preco_entrada)
        
        if quantidade == 0:
            logger.warning(f"Quantidade zero calculada. Saldo: {saldo}, Preço: {preco_entrada}")
            return None
        
        # Calcular níveis de risco
        if sinal == Sinal.COMPRA:
            stop_loss = preco_entrada * (1 - self.config.stop_loss_pct)
            take_profit = preco_entrada * (1 + self.config.take_profit_pct)
        else:  # VENDA
            stop_loss = preco_entrada * (1 + self.config.stop_loss_pct)
            take_profit = preco_entrada * (1 - self.config.take_profit_pct)
        
        timestamp = df.iloc[-1]['datetime'] if 'datetime' in df.columns else None
        
        ordem = OrdemTrade(
            acao=sinal.value,
            entrada=preco_entrada,
            quantidade=quantidade,
            stop_loss=stop_loss,
            take_profit=take_profit,
            timestamp=str(timestamp) if timestamp else None,
            sinal_confirmado=True
        )
        
        logger.info(f"Ordem gerada: {ordem}")
        return ordem
    
    def avaliar_posicao(self, preco_atual: float, ordem: OrdemTrade) -> Optional[str]:
        """
        Avalia se uma posição aberta deve ser fechada.
        
        Args:
            preco_atual: Preço atual do ativo
            ordem: OrdemTrade aberta
        
        Returns:
            'TP' (Take Profit), 'SL' (Stop Loss) ou None se ainda aberta
        """
        if ordem.acao == 'COMPRA':
            if preco_atual >= ordem.take_profit:
                return 'TP'
            elif preco_atual <= ordem.stop_loss:
                return 'SL'
        elif ordem.acao == 'VENDA':
            if preco_atual <= ordem.take_profit:
                return 'TP'
            elif preco_atual >= ordem.stop_loss:
                return 'SL'
        
        return None
