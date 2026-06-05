"""
Módulo de Backtest - Motor de simulação histórica
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from indicadores import IndicadoresTA
from estrategia import Estrategia, ConfiguracaoEstrategia, OrdemTrade, Sinal

logger = logging.getLogger(__name__)


@dataclass
class ResultadoOperacao:
    """Resultado de uma operação completa (aberta e fechada)."""
    
    numero: int              # ID da operação
    data_entrada: str        # Data de entrada
    acao: str                # COMPRA ou VENDA
    preco_entrada: float     # Preço de entrada
    quantidade: float        # Quantidade operada
    preco_saida: float       # Preço de saída
    motivo_saida: str        # TP, SL ou TIME (timeout)
    lucro_bruto: float       # Lucro/prejuízo em reais
    retorno_pct: float       # Retorno em percentual
    comissao: float          # Comissão cobrada
    lucro_liquido: float     # Lucro após comissão
    duracao_barras: int      # Número de candles
    timestamp_saida: Optional[str] = None
    
    def __str__(self) -> str:
        """String formatada da operação."""
        sinal = "+" if self.lucro_liquido >= 0 else ""
        return (
            f"Op #{self.numero:03d} | {self.acao:5s} | "
            f"Entrada: R$ {self.preco_entrada:10.2f} | "
            f"Saída: R$ {self.preco_saida:10.2f} | "
            f"Lucro: {sinal}R$ {self.lucro_liquido:8.2f} ({sinal}{self.retorno_pct:5.2f}%)"
        )


@dataclass
class RelatorioBacktest:
    """Relatório completo do backtest."""
    
    capital_inicial: float
    capital_final: float
    total_operacoes: int = 0
    operacoes_vencedoras: int = 0
    operacoes_perdedoras: int = 0
    taxa_acerto: float = 0.0
    
    lucro_total: float = 0.0
    retorno_total_pct: float = 0.0
    
    maior_ganha: float = 0.0
    maior_perda: float = 0.0
    media_ganha: float = 0.0
    media_perda: float = 0.0
    
    drawdown_maximo: float = 0.0
    comissoes_totais: float = 0.0
    
    operacoes: List[ResultadoOperacao] = field(default_factory=list)
    
    def calcular_metricas(self):
        """Calcula métricas derivadas."""
        if not self.operacoes:
            return
        
        self.total_operacoes = len(self.operacoes)
        self.operacoes_vencedoras = sum(1 for op in self.operacoes if op.lucro_liquido > 0)
        self.operacoes_perdedoras = sum(1 for op in self.operacoes if op.lucro_liquido < 0)
        
        if self.total_operacoes > 0:
            self.taxa_acerto = self.operacoes_vencedoras / self.total_operacoes
        
        self.lucro_total = sum(op.lucro_liquido for op in self.operacoes)
        self.comissoes_totais = sum(op.comissao for op in self.operacoes)
        
        if self.capital_inicial > 0:
            self.retorno_total_pct = (self.lucro_total / self.capital_inicial) * 100
        
        # Ganhos e perdas
        ganhos = [op.lucro_liquido for op in self.operacoes if op.lucro_liquido > 0]
        perdas = [op.lucro_liquido for op in self.operacoes if op.lucro_liquido < 0]
        
        if ganhos:
            self.maior_ganha = max(ganhos)
            self.media_ganha = sum(ganhos) / len(ganhos)
        
        if perdas:
            self.maior_perda = min(perdas)
            self.media_perda = sum(perdas) / len(perdas)
    
    def __str__(self) -> str:
        """String formatada do relatório."""
        return f"""
{'='*70}
RELATÓRIO DE BACKTEST
{'='*70}
Capital Inicial:         R$ {self.capital_inicial:>12,.2f}
Capital Final:           R$ {self.capital_final:>12,.2f}
Lucro Total:             R$ {self.lucro_total:>12,.2f}
Retorno:                 {self.retorno_total_pct:>14.2f}%

Total de Operações:      {self.total_operacoes:>14}
Operações Vencedoras:    {self.operacoes_vencedoras:>14}
Operações Perdedoras:    {self.operacoes_perdedoras:>14}
Taxa de Acerto:          {self.taxa_acerto*100:>14.2f}%

Maior Ganha:             R$ {self.maior_ganha:>12,.2f}
Maior Perda:             R$ {self.maior_perda:>12,.2f}
Média Ganha:             R$ {self.media_ganha:>12,.2f}
Média Perda:             R$ {self.media_perda:>12,.2f}

Drawdown Máximo:         {self.drawdown_maximo:>14.2f}%
Comissões Totais:        R$ {self.comissoes_totais:>12,.2f}
{'='*70}
"""


class Backtest:
    """
    Motor de simulação histórica (backtesting).
    
    Processa dados históricos e simula a execução da estratégia,
    gerando relatório de performance.
    """
    
    def __init__(
        self,
        capital_inicial: float = 10000.0,
        comissao_pct: float = 0.001,
        config_estrategia: Optional[ConfiguracaoEstrategia] = None,
        max_operacoes_abertas: int = 1,
        max_barras_posicao: int = None
    ):
        """
        Inicializa o backtest.
        
        Args:
            capital_inicial: Capital inicial para simulação
            comissao_pct: Percentual de comissão por operação
            config_estrategia: Configuração da estratégia
            max_operacoes_abertas: Máximo de operações simultâneas
            max_barras_posicao: Máximo de candles por posição (None = ilimitado)
        """
        self.capital_inicial = capital_inicial
        self.comissao_pct = comissao_pct
        self.config_estrategia = config_estrategia or ConfiguracaoEstrategia()
        self.max_operacoes_abertas = max_operacoes_abertas
        self.max_barras_posicao = max_barras_posicao
        
        self.estrategia = Estrategia(self.config_estrategia)
        
        logger.info(f"Backtest inicializado com capital: R$ {capital_inicial:,.2f}")
    
    def executar(self, df: pd.DataFrame, ativo: str = "ATIVO") -> RelatorioBacktest:
        """
        Executa o backtest nos dados históricos.
        
        Args:
            df: DataFrame com dados OHLCV (deve incluir 'datetime')
            ativo: Nome do ativo para logging
        
        Returns:
            RelatorioBacktest com resultados
        """
        logger.info(f"Iniciando backtest para {ativo} com {len(df)} candles")
        
        # Validar dados
        if df.empty:
            logger.error("DataFrame vazio fornecido")
            return RelatorioBacktest(
                capital_inicial=self.capital_inicial,
                capital_final=self.capital_inicial
            )
        
        # Adicionar indicadores
        df = IndicadoresTA.adicionar_indicadores(df)
        
        # Inicializar variáveis
        saldo = self.capital_inicial
        posicoes_abertas: Dict[int, Tuple[int, OrdemTrade]] = {}  # id_posição: (idx_entrada, ordem)
        operacoes_completas: List[ResultadoOperacao] = []
        num_operacao = 0
        equidade_serie = []
        
        # Iterar pelas barras
        for idx in range(21, len(df)):  # Começar após EMA de 21 períodos
            barra_atual = df.iloc[idx]
            df_ate_aqui = df.iloc[:idx+1]
            
            preco_atual = float(barra_atual['close'])
            data_atual = str(barra_atual['datetime'])
            
            # Atualizar equidade
            equidade = saldo
            for id_pos, (idx_entrada, ordem) in posicoes_abertas.items():
                p_l = self._calcular_p_l(ordem, preco_atual)
                equidade += p_l
            equidade_serie.append(equidade)
            
            # Avaliar posições abertas (Stop Loss / Take Profit / Timeout)
            posicoes_para_fechar = []
            for id_pos, (idx_entrada, ordem) in posicoes_abertas.items():
                # Verificar timeout
                barras_decorridas = idx - idx_entrada
                if self.max_barras_posicao and barras_decorridas >= self.max_barras_posicao:
                    posicoes_para_fechar.append((id_pos, preco_atual, 'TIME'))
                    continue
                
                # Verificar TP/SL
                resultado = self.estrategia.avaliar_posicao(preco_atual, ordem)
                if resultado:
                    posicoes_para_fechar.append((id_pos, preco_atual, resultado))
            
            # Fechar posições
            for id_pos, preco_saida, motivo in posicoes_para_fechar:
                idx_entrada, ordem = posicoes_abertas.pop(id_pos)
                
                resultado_op = self._gerar_resultado_operacao(
                    num_operacao, idx_entrada, idx, df, ordem, preco_saida, motivo
                )
                operacoes_completas.append(resultado_op)
                
                # Atualizar saldo
                saldo += resultado_op.lucro_liquido
                logger.info(str(resultado_op))
                num_operacao += 1
            
            # Gerar novas posições (se houver espaço)
            if len(posicoes_abertas) < self.max_operacoes_abertas:
                ordem = self.estrategia.gerar_ordem(df_ate_aqui, saldo)
                if ordem:
                    posicoes_abertas[id(ordem)] = (idx, ordem)
                    logger.info(f"Posição aberta em {data_atual}: {ordem}")
        
        # Fechar posições abertas ao final
        for id_pos, (idx_entrada, ordem) in list(posicoes_abertas.items()):
            preco_saida = float(df.iloc[-1]['close'])
            resultado_op = self._gerar_resultado_operacao(
                num_operacao, idx_entrada, len(df)-1, df, ordem, preco_saida, 'END'
            )
            operacoes_completas.append(resultado_op)
            saldo += resultado_op.lucro_liquido
            logger.info(str(resultado_op))
            num_operacao += 1
        
        # Calcular drawdown máximo
        equidade_array = np.array(equidade_serie)
        picos = np.maximum.accumulate(equidade_array)
        drawdowns = (equidade_array - picos) / picos
        drawdown_maximo = np.min(drawdowns) * 100 if len(drawdowns) > 0 else 0
        
        # Gerar relatório
        relatorio = RelatorioBacktest(
            capital_inicial=self.capital_inicial,
            capital_final=saldo,
            drawdown_maximo=drawdown_maximo,
            operacoes=operacoes_completas
        )
        relatorio.calcular_metricas()
        
        logger.info(f"Backtest concluído: {len(operacoes_completas)} operações executadas")
        
        return relatorio
    
    def _calcular_p_l(self, ordem: OrdemTrade, preco_atual: float) -> float:
        """Calcula P&L não realizado de uma posição aberta."""
        if ordem.acao == 'COMPRA':
            return (preco_atual - ordem.entrada) * ordem.quantidade
        else:
            return (ordem.entrada - preco_atual) * ordem.quantidade
    
    def _gerar_resultado_operacao(
        self,
        num_op: int,
        idx_entrada: int,
        idx_saida: int,
        df: pd.DataFrame,
        ordem: OrdemTrade,
        preco_saida: float,
        motivo: str
    ) -> ResultadoOperacao:
        """Gera resultado de uma operação completa."""
        
        data_entrada = str(df.iloc[idx_entrada]['datetime'])
        data_saida = str(df.iloc[idx_saida]['datetime'])
        
        if ordem.acao == 'COMPRA':
            lucro_bruto = (preco_saida - ordem.entrada) * ordem.quantidade
        else:
            lucro_bruto = (ordem.entrada - preco_saida) * ordem.quantidade
        
        comissao = ordem.entrada * ordem.quantidade * self.comissao_pct * 2  # Ida e volta
        lucro_liquido = lucro_bruto - comissao
        
        retorno_pct = (lucro_liquido / (ordem.entrada * ordem.quantidade)) * 100 if ordem.entrada > 0 else 0
        
        return ResultadoOperacao(
            numero=num_op,
            data_entrada=data_entrada,
            acao=ordem.acao,
            preco_entrada=ordem.entrada,
            quantidade=ordem.quantidade,
            preco_saida=preco_saida,
            motivo_saida=motivo,
            lucro_bruto=lucro_bruto,
            retorno_pct=retorno_pct,
            comissao=comissao,
            lucro_liquido=lucro_liquido,
            duracao_barras=idx_saida - idx_entrada,
            timestamp_saida=data_saida
        )
