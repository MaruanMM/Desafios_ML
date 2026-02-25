# 🚀 Portfólio de Inteligência Artificial - Águia Sistemas

Este repositório apresenta a resolução técnica de 4 desafios de Machine Learning, focados em transformar dados brutos em decisões estratégicas. 

---

## 🏥 Desafio 1: Previsão de Custos Médicos
**Objetivo:** Prever custos hospitalares para otimizar a precificação de seguros.

* **Métricas de Desempenho:**
    * **R² (Média):** ~0.83 (O modelo explica 83% da variação dos custos).
    * **MAE (Erro Médio Absoluto):** ~$2,500.00.
* **Análise de Variáveis:** O **tabagismo** foi a variável mais determinante (61.8% de importância), seguido pelo **IMC**.
* **Decisão Técnica:** Utilizei o `RandomForestRegressor` com K-Fold Cross-Validation para garantir que o modelo seja robusto e não sofra de overfitting. Teve um resultado melho que o modelo de Regressão Linear.

---

## 🚢 Desafio 2: Sobrevivência no Titanic
**Objetivo:** Classificação binária para identificar perfis de sobrevivência.

* **Métricas de Desempenho:**
    * **Acurácia:** ~82% no conjunto de teste.
    * **Insights:**  Os dados confirmam que gênero e classe social foram determinantes, mas também revelam que crianças e famílias pequenas tiveram maiores chances de sobrevivência, reforçando desigualdades sociais e logísticas do resgate.
* **Engenharia de Atributos:** Criei a variável `FamilySize` (Tamanho da Família) para capturar o impacto de viajar acompanhado, o que se mostrou um fator relevante na sobrevivência.

---

## 📉 Desafio 3: Predição de Churn (Retenção de Clientes)
**Objetivo:** Reduzir a perda de receita identificando clientes propensos a cancelar.

* **Métricas de Desempenho:**
    * **Recall:** Consegue prever 51% dos clientes que realmente vão cancelar. 
    * **F1-Score:** Focado no equilíbrio entre identificar o churn real e evitar alarmes falsos.
* **Relação com Negócio:** * **Impacto Financeiro:** Identificamos um prejuízo mensal de **$139,130.85** com clientes que saíram.
    * **Causa Raiz:** O suporte técnico e o tipo de contrato (mensal vs anual) são os principais gatilhos de saída.
    * **Estratégia:** Recomendamos foco imediato nos 5 clientes com maior **CLTV** e **Churn Score > 80**.

---

## 🛡️ Desafio 4: Detecção de Fraude em Cartões
**Objetivo:** Identificar transações fraudulentas em cenários de alta assimetria.

* **Métricas de Desempenho (Cruciais):**
    * **Recall (Sensibilidade):** 92% - Priorizamos não deixar nenhuma fraude passar.
    * **Precisão:** 99% - Garantindo que clientes legítimos não sejam bloqueados indevidamente.
* **Estratégia de Dados:** Utilizei **Undersampling** para equilibrar a base (492 fraudes vs 492 normais).
* **Métrica Prioritária:** O **Recall** foi escolhido como métrica guia, pois o custo de uma fraude não detectada é superior ao custo operacional de uma verificação adicional.


