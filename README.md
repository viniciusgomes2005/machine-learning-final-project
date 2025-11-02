# SleepGuard (Projeto Aplicado)

Este repositório contém um MVP para prever **baixa eficiência do sono** e exibir recomendações de higiene do sono.

## Estrutura
```
sleepguard/
├─ requirements.txt
└─ src/
   ├─ __init__.py
   ├─ config.py
   ├─ data.py
   ├─ features.py
   ├─ model.py
   ├─ train.py
   ├─ evaluate.py
   └─ app.py
```
- Coloque seu CSV em `sleepguard/data/sleep.csv` (ajuste nomes de colunas conforme seu dataset).
- O pipeline de código foi **propositadamente alterado** em sala para fins didáticos.

## Comandos (exemplo)
```bash
# criar venv e instalar deps
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# treinar (gera artifacts/)
python -m src.train

# avaliar
python -m src.evaluate

# servir API (após treinar)
uvicorn src.app:app --reload
```
