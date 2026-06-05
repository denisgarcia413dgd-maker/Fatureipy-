"""
Script de Teste Completo - Valida toda arquitetura do FatureiPy
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Importar módulos
from datasources import BinanceDataSource, YFinanceDataSource
from indicadores import IndicadoresTA
from estrategia import Estrategia, ConfiguracaoEstrategia, Sinal
from backtest import Backtest


def gerar_dados_teste(num_candles: int = 500, volatilidade: float = 0.02) -> pd.DataFrame:
    """
    Gera dados OHLCV sintéticos para testes.
    
    Args:
        num_candles: Número de candles a gerar
        volatilidade: Volatilidade dos preços
    
    Returns:
        DataFrame com dados OHLCV
    """
    logger.info(f"Gerando {num_candles} candles sintéticos com volatilidade {volatilidade:.2%}")
    
    # Gerar série de preços com random walk
    np.random.seed(42)
    precos = [100.0]
    
    for _ in range(num_candles - 1):
        mudanca = np.random.normal(0, volatilidade)
        novo_preco = precos[-1] * (1 + mudanca)
        precos.append(novo_preco)
    
    precos = np.array(precos)
    
    # Criar OHLCV
    df = pd.DataFrame({
        'datetime': pd.date_range(start='2024-01-01', periods=num_candles, freq='15min'),
        'open': precos * (1 + np.random.uniform(-0.002, 0.002, num_candles)),
        'high': precos * (1 + np.random.uniform(0, 0.005, num_candles)),
        'low': precos * (1 - np.random.uniform(0, 0.005, num_candles)),
        'close': precos,
        'volume': np.random.uniform(100000, 1000000, num_candles)
    })
    
    # Garantir que high >= close >= low
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)
    
    logger.info(f"✅ Dados sintéticos gerados: {len(df)} candles")
    return df


def teste_indicadores():
    """Testa o módulo de indicadores."""
    logger.info("\n" + "="*70)
    logger.info("TESTE 1: INDICADORES")
    logger.info("="*70)
    
    try:
        # Gerar dados
        df = gerar_dados_teste(100)
        
        # Testar EMA
        logger.info("Testando EMA...")
        ema9 = IndicadoresTA.ema(df['close'], 9)
        assert len(ema9) == len(df), "EMA tem tamanho incorreto"
        assert ema9.isna().sum() > 0, "EMA deveria ter NaNs iniciais"
        logger.info(f"✅ EMA calculada: {ema9.iloc[-1]:.2f}")
        
        # Testar SMA
        logger.info("Testando SMA...")
        sma21 = IndicadoresTA.sma(df['close'], 21)
        assert len(sma21) == len(df), "SMA tem tamanho incorreto"
        logger.info(f"✅ SMA calculada: {sma21.iloc[-1]:.2f}")
        
        # Testar ADX
        logger.info("Testando ADX...")
        adx = IndicadoresTA.calcular_adx(df, periodo=14)
        assert not adx.empty, "ADX retornou vazio"
        assert 'ADX' in adx.columns, "Coluna ADX não encontrada"
        logger.info(f"✅ ADX calculado: {adx['ADX'].iloc[-1]:.2f}")
        
        # Testar batch
        logger.info("Testando adicionar_indicadores...")
        df_completo = IndicadoresTA.adicionar_indicadores(df)
        assert 'EMA_9' in df_completo.columns, "EMA_9 não adicionada"
        assert 'EMA_21' in df_completo.columns, "EMA_21 não adicionada"
        assert 'ADX' in df_completo.columns, "ADX não adicionado"
        logger.info(f"✅ Indicadores batch adicionados com sucesso")
        
        logger.info("✅ TESTE 1 PASSOU\n")
        return True
        
    except Exception as e:
        logger.error(f"❌ TESTE 1 FALHOU: {e}\n")
        return False


def teste_estrategia():
    """Testa o módulo de estratégia."""
    logger.info("="*70)
    logger.info("TESTE 2: ESTRATÉGIA")
    logger.info("="*70)
    
    try:
        # Gerar dados
        df = gerar_dados_teste(100)
        df = IndicadoresTA.adicionar_indicadores(df)
        
        # Criar estratégia
        logger.info("Criando estratégia com config padrão...")
        config = ConfiguracaoEstrategia()
        estrategia = Estrategia(config)
        logger.info(f"✅ Estratégia criada: {config}")
        
        # Testar geração de sinal
        logger.info("Testando gerar_sinal...")
        sinal = estrategia.gerar_sinal(df)
        assert isinstance(sinal, Sinal), "Sinal retornado não é do tipo Sinal"
        logger.info(f"✅ Sinal gerado: {sinal.value}")
        
        # Testar validação de tendência
        logger.info("Testando validar_tendencia...")
        tendencia_valida = estrategia.validar_tendencia(df)
        assert isinstance(tendencia_valida, bool), "validar_tendencia deveria retornar bool"
        logger.info(f"✅ Tendência válida: {tendencia_valida} (ADX: {df.iloc[-1]['ADX']:.2f})")
        
        # Testar cálculo de quantidade
        logger.info("Testando calcular_quantidade...")
        saldo = 10000.0
        preco = df.iloc[-1]['close']
        quantidade = estrategia.calcular_quantidade(saldo, preco)
        assert quantidade > 0, "Quantidade deveria ser positiva"
        valor_operacao = quantidade * preco
        percentual_esperado = valor_operacao / saldo
        logger.info(f"✅ Quantidade: {quantidade:.6f} (valor: R$ {valor_operacao:.2f}, {percentual_esperado:.2%})")
        
        # Testar geração de ordem
        logger.info("Testando gerar_ordem...")
        ordem = estrategia.gerar_ordem(df, saldo)
        if ordem:
            logger.info(f"✅ Ordem gerada: {ordem.acao} @ R$ {ordem.entrada:.2f}")
            logger.info(f"   SL: R$ {ordem.stop_loss:.2f} | TP: R$ {ordem.take_profit:.2f}")
        else:
            logger.info("ℹ️  Nenhuma ordem gerada (sinal não confirmado)")
        
        # Testar avaliação de posição
        if ordem:
            logger.info("Testando avaliar_posicao...")
            # Simular preço acima de take profit
            preco_tp = ordem.take_profit + 1
            resultado_tp = estrategia.avaliar_posicao(preco_tp, ordem)
            assert resultado_tp == 'TP', "Deveria atingir TP"
            logger.info(f"✅ TP acionado em R$ {preco_tp:.2f}")
            
            # Simular preço abaixo de stop loss
            preco_sl = ordem.stop_loss - 1
            resultado_sl = estrategia.avaliar_posicao(preco_sl, ordem)
            assert resultado_sl == 'SL', "Deveria acionat SL"
            logger.info(f"✅ SL acionado em R$ {preco_sl:.2f}")
        
        logger.info("✅ TESTE 2 PASSOU\n")
        return True
        
    except Exception as e:
        logger.error(f"❌ TESTE 2 FALHOU: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def teste_backtest():
    """Testa o módulo de backtest."""
    logger.info("="*70)
    logger.info("TESTE 3: BACKTEST")
    logger.info("="*70)
    
    try:
        # Gerar dados
        logger.info("Gerando dados para backtest...")
        df = gerar_dados_teste(500)
        
        # Criar backtest
        logger.info("Criando engine de backtest...")
        capital = 10000.0
        backtest = Backtest(
            capital_inicial=capital,
            comissao_pct=0.001,
            max_operacoes_abertas=1,
            max_barras_posicao=100
        )
        logger.info(f"✅ Backtest criado com capital: R$ {capital:,.2f}")
        
        # Executar backtest
        logger.info("Executando backtest...")
        relatorio = backtest.executar(df, ativo="TESTE_SINTETICO")
        
        # Validar resultados
        assert relatorio.capital_final > 0, "Capital final deveria ser positivo"
        logger.info(f"✅ Backtest executado com sucesso!")
        
        # Exibir relatório
        print("\n" + str(relatorio))
        
        # Validar métricas
        logger.info("Validando métricas...")
        if relatorio.total_operacoes > 0:
            logger.info(f"📊 Total de operações: {relatorio.total_operacoes}")
            logger.info(f"📊 Vencedoras: {relatorio.operacoes_vencedoras} ({relatorio.taxa_acerto*100:.1f}%)")
            logger.info(f"📊 Perdedoras: {relatorio.operacoes_perdedoras}")
            logger.info(f"📊 Lucro total: R$ {relatorio.lucro_total:,.2f}")
            logger.info(f"📊 Retorno: {relatorio.retorno_total_pct:.2f}%")
            
            # Exibir detalhes de operações
            logger.info("\nDetalhes das operações:")
            for op in relatorio.operacoes[:5]:  # Primeiras 5
                print(f"  {op}")
            if len(relatorio.operacoes) > 5:
                print(f"  ... ({len(relatorio.operacoes)-5} operações adicionais)")
        
        logger.info("✅ TESTE 3 PASSOU\n")
        return True
        
    except Exception as e:
        logger.error(f"❌ TESTE 3 FALHOU: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def teste_integracao_completa():
    """Testa integração de todos os módulos."""
    logger.info("="*70)
    logger.info("TESTE 4: INTEGRAÇÃO COMPLETA")
    logger.info("="*70)
    
    try:
        # Simular fluxo completo
        logger.info("Etapa 1: Buscar dados via datasource...")
        try:
            logger.info("Tentando buscar dados reais da Binance...")
            binance = BinanceDataSource(market_type="spot")
            df_real = binance.fetch_data("BTCUSDT", interval="1h", limit=100)
            
            if not df_real.empty:
                logger.info(f"✅ Dados reais obtidos: {len(df_real)} candles")
                df = df_real
            else:
                logger.warning("⚠️  Sem dados reais, usando dados sintéticos")
                df = gerar_dados_teste(100)
        except Exception as e:
            logger.warning(f"⚠️  Falha ao buscar dados reais: {e}, usando sintéticos")
            df = gerar_dados_teste(100)
        
        # Etapa 2: Calcular indicadores
        logger.info("Etapa 2: Calcular indicadores...")
        df = IndicadoresTA.adicionar_indicadores(df)
        logger.info(f"✅ Indicadores calculados")
        
        # Etapa 3: Gerar sinais
        logger.info("Etapa 3: Gerar sinais...")
        estrategia = Estrategia(ConfiguracaoEstrategia())
        sinal = estrategia.gerar_sinal(df)
        logger.info(f"✅ Sinal gerado: {sinal.value}")
        
        # Etapa 4: Executar backtest
        logger.info("Etapa 4: Executar backtest...")
        backtest = Backtest(capital_inicial=10000.0)
        relatorio = backtest.executar(df)
        logger.info(f"✅ Backtest executado: {relatorio.total_operacoes} operações")
        
        # Exibir resumo
        print("\n" + str(relatorio))
        
        logger.info("✅ TESTE 4 PASSOU\n")
        return True
        
    except Exception as e:
        logger.error(f"❌ TESTE 4 FALHOU: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Executa todos os testes."""
    logger.info("\n" + "="*70)
    logger.info("🚀 FATUREIPY - SUITE DE TESTES")
    logger.info("="*70 + "\n")
    
    resultados = {
        "Indicadores": teste_indicadores(),
        "Estratégia": teste_estrategia(),
        "Backtest": teste_backtest(),
        "Integração Completa": teste_integracao_completa(),
    }
    
    # Resumo
    logger.info("="*70)
    logger.info("📋 RESUMO DOS TESTES")
    logger.info("="*70)
    
    passou = sum(1 for v in resultados.values() if v)
    total = len(resultados)
    
    for nome, resultado in resultados.items():
        status = "✅ PASSOU" if resultado else "❌ FALHOU"
        logger.info(f"{nome:30s} {status}")
    
    logger.info("="*70)
    logger.info(f"Total: {passou}/{total} testes passaram")
    
    if passou == total:
        logger.info("🎉 TODOS OS TESTES PASSARAM! O FatureiPy está pronto para uso!\n")
    else:
        logger.warning(f"⚠️  {total - passou} teste(s) falharam. Verifique os erros acima.\n")
    
    return passou == total


if __name__ == "__main__":
    sucesso = main()
    exit(0 if sucesso else 1)
