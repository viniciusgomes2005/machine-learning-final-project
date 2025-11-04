# Sleep Efficiency — ML Final Project (Notebook Único)

Este repositório contém **apenas um Jupyter Notebook** que executa todo o pipeline de ponta a ponta: carregamento do CSV, engenharia de features, validação por grupos, verificação rápida de vazamento (rótulos embaralhados), treino final, seleção de limiar e **gráficos de ROC/PR + matriz de confusão**, além de um relatório `JSON` com métricas (AUC, precisão, recall, F1).

---

## 📄 Arquivo principal

* `sleep_efficiency_single_notebook.ipynb`

> Abra e execute as células **de cima para baixo**.

---

## 📦 Requisitos (mínimos)

Você pode usar **JupyterLab**, **VS Code** (extensão Jupyter) ou **Google Colab**.

Pacotes Python usados no notebook:

* `numpy`, `pandas`, `matplotlib`
* `scikit-learn`
* `lightgbm` (usado em um dos candidatos de modelo)
* `python-dateutil` (via `pandas.to_datetime`, já costuma vir)

Se quiser rodar localmente num ambiente virtual:

```bash
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install numpy pandas matplotlib scikit-learn lightgbm
```

---

## 📂 Dados esperados

Coloque o CSV em:

```
data/sleep.csv
```

Campos típicos (nomes exatos podem variar, mas o notebook tenta se adaptar):

* `Sleep efficiency` (alvo binário: ineficiente se < 0.85)
* `Bedtime`, `Wakeup time`
* `Awakenings`, `Sleep duration`
* `REM sleep percentage`, `Deep sleep percentage`, `Light sleep percentage`
* `Caffeine intake`, `Alcohol consumption`, `Exercise frequency`
* `Age`, `Gender`, `Smoking status`
* `Subject ID` (usado como **grupo** quando disponível)

> Se usar outros nomes, ajuste a constante `DATA_PATH` na primeira célula ou renomeie as colunas no CSV.

---

## ▶️ Como executar

1. Abra o `sleep_efficiency_single_notebook.ipynb` no Jupyter/VS Code/Colab.
2. Garanta que o arquivo `data/sleep.csv` existe.
3. Rode **todas as células** em ordem.

O notebook criará automaticamente a pasta:

```
reports/
```

e salvará:

* `holdin_roc.png`
* `holdin_pr.png`
* `holdin_cm.png`
* `final_report.json` (métricas e caminhos dos gráficos)
* `train_debug.json` (detalhes da validação cruzada e checagem com rótulos embaralhados)

---

## 🧪 O que o notebook faz

* **Engenharia de features**: codificação circular de horários (bedtime/wakeup), algumas variáveis de hábitos/demografia, e limpeza básica.
* **Validação por grupo**: `GroupKFold` quando há `Subject ID` (evita “mesma pessoa” em treino e validação).
* **Sanity check (anti-vazamento)**: repete a validação com **rótulos embaralhados** para conferir se o AUC cai para ~0.5.
* **Seleção de modelo**: compara candidatos (logística / LightGBM calibrado) por AUC médio.
* **Treino final**: treina no conjunto todo com o melhor candidato.
* **Seleção de limiar**: escolhe limiar que maximiza **F1** (pode trocar fácil para precisão/recall).
* **Gráficos**: ROC, Precision–Recall e matriz de confusão no limiar escolhido.
* **Relatório**: precision/recall/F1/AUC + suporte da classe positiva, número de amostras e caminhos dos gráficos.

---

## 📊 Resultados e relatórios

* **`reports/final_report.json`** contém:

  * `threshold` usado,
  * `auc`, `precision`, `recall`, `f1` para a classe positiva,
  * `support_pos`, `n_samples`,
  * caminhos dos gráficos.

* **Curvas**:

  * `reports/holdin_roc.png`
  * `reports/holdin_pr.png`
  * `reports/holdin_cm.png`

> Observação: as curvas “hold-in” são calculadas no conjunto completo para visualização/diagnóstico; para avaliação fora-da-amostra, use a célula opcional com `GroupShuffleSplit` (se você a mantiver no notebook).

---

## 🛡️ Notas sobre vazamento e métricas altas

* O notebook **embaralha os rótulos** e repete a validação para gerar um **baseline aleatório**. Se o AUC nessa condição ficar próximo de 0.5, é um bom indício de que não há vazamento óbvio no pipeline.
* AUCs altos podem ocorrer por:

  * Alvo fácil/fortemente correlacionado com hábitos/horários,
  * Conjunto pequeno e relativamente homogêneo,
  * Distribuição de classes e features bem separáveis.
* Para uma avaliação mais conservadora, adicione/ative um **holdout por grupo** (já há utilitários no notebook para isso).

---

## ❓ Perguntas frequentes

**1) Posso rodar sem `Subject ID`?**
Sim. O código cai automaticamente para um agrupamento pelo índice. Ainda assim, recomenda-se adicionar um identificador de sujeito se existirem múltiplas linhas por pessoa.

**2) Onde altero o caminho do CSV?**
Na primeira célula, edite `DATA_PATH`.

**3) Onde mudo a métrica/limiar?**
Na célula da seleção de limiar, você pode otimizar por `precision` ou `recall` em vez de F1, ou fixar um limiar como 0.5.

---

## 📜 Licença

Use livremente para fins acadêmicos. Cite o repositório/notebook quando apropriado.
