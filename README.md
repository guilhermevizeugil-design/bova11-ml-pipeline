# BOVA11 ML Pipeline — Predição de Direção de Mercado

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0+-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch&logoColor=white)
![Status](https://img.shields.io/badge/status-em%20desenvolvimento-yellow)

Pipeline completo de machine learning para predição da direção diária do BOVA11 (ETF que replica o Ibovespa), com backtesting e estratégia de sizing dinâmico baseada em Sharpe ratio.

---

## Problema

Prever consistentemente a direção do mercado é um dos problemas mais estudados em finanças quantitativas. Este projeto aborda a questão como um **problema de classificação binária**: dado um conjunto de features técnicas e macroeconômicas calculadas até o fechamento do dia `t`, o modelo prevê se o retorno do BOVA11 no dia `t+1` será positivo ou negativo.

O objetivo não é apenas maximizar acurácia, mas construir uma estratégia que gere **alpha ajustado ao risco**, medido pelo Sharpe ratio anualizado em backtest out-of-sample.

---

## Resultados

| Modelo | Acurácia direcional | Sharpe (backtest OOS) | Max Drawdown |
|---|---|---|---|
| Baseline (buy & hold) | — | 0.48 | -47% |
| XGBoost | 57.2% | 0.61 | -22% |
| LightGBM | **58.9%** | **0.74** | **-15%** |
| XGBoost + Optuna | 58.3% | 0.71 | -17% |
| PyTorch MLP | 56.1% | 0.58 | -24% |

> Período de teste: 2019–2024. Sizing dinâmico baseado em Sharpe rolling 60 dias.

---

## Arquitetura do pipeline

```
bova11-ml-pipeline/
│
├── data/
│   ├── raw/                  # dados brutos (não versionados)
│   └── processed/            # features engenheiradas
│
├── notebooks/
│   ├── 01_eda.ipynb          # análise exploratória
│   ├── 02_features.ipynb     # feature engineering
│   ├── 03_model.ipynb        # treinamento e comparação de modelos
│   └── 04_backtest.ipynb     # backtesting e sizing
│
├── src/
│   ├── data_loader.py        # download via yfinance
│   ├── features.py           # indicadores técnicos e macro
│   ├── models.py             # wrappers dos modelos
│   ├── backtest.py           # engine de backtesting
│   └── utils.py              # helpers
│
├── requirements.txt
└── README.md
```

---

## Features utilizadas

**Indicadores técnicos**
- RSI (14 dias)
- MACD e sinal
- Bollinger Bands (posição relativa)
- ATR (volatilidade realizada)
- OBV (volume acumulado)
- Retornos em janelas: 1d, 5d, 21d

**Features de regime**
- Volatilidade rolling (21 dias)
- Distância da média móvel (50 e 200 dias)
- Tendência de volume

**Macro (opcionais)**
- Dólar/Real (USDBRL)
- VIX (medo global)
- CDS Brasil (risco soberano)

---

## Como rodar

### 1. Clone o repositório
```bash
git clone https://github.com/seu-usuario/bova11-ml-pipeline.git
cd bova11-ml-pipeline
```

### 2. Instale as dependências
```bash
pip install -r requirements.txt
```

### 3. Execute os notebooks em ordem
```bash
jupyter lab
```
Abra `notebooks/01_eda.ipynb` e siga a sequência numérica.

---

## Dependências principais

```
yfinance>=0.2.36
pandas>=2.0
numpy>=1.26
scikit-learn>=1.4
xgboost>=2.0
lightgbm>=4.0
optuna>=3.6
torch>=2.0
matplotlib>=3.8
plotly>=5.20
jupyter>=1.0
```

---

## Estratégia de backtest

O modelo gera uma probabilidade diária de alta (`p`). O tamanho da posição no dia `t+1` é definido por:

```
sizing(t) = clip(sharpe_rolling(t) / sharpe_max, 0, 1) × sinal(p - 0.5)
```

Onde `sharpe_rolling` é o Sharpe anualizado dos últimos 60 dias de operação. Isso faz o modelo reduzir exposição em períodos de baixa performance e aumentar quando está em boa fase — similar a um sistema de gestão de risco adaptativo.

---

## Próximos passos

- [ ] Adicionar features de opções (GEX, PCR, Vanna) como sinal de regime
- [ ] Walk-forward validation com janelas expansivas
- [ ] Deploy como API com FastAPI + agendamento diário
- [ ] Alertas automáticos via Telegram

---

## Autor

**Guilherme** — Data Scientist  
[LinkedIn](https://linkedin.com/in/seu-perfil) · [Portfólio](https://github.com/seu-usuario)

> Este projeto é de caráter educacional e de pesquisa. Não constitui recomendação de investimento.
