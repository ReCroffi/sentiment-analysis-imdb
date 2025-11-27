# 🎬 Análise de Sentimentos em Reviews de Filmes  
### Classificação Automática de Reviews do IMDb com Machine Learning e NLP

---

## 📌 Visão Geral

Este projeto implementa um pipeline **completo e profissional** de *Natural Language Processing (NLP)* para classificar reviews de filmes como **positivos** ou **negativos**.  
O objetivo foi construir um modelo limpo, interpretável e com desempenho competitivo, ideal para compor um portfólio em Ciência de Dados.

---

## ⭐ Principais Resultados

- ✔️ Acurácia final: **~89%** (Logistic Regression e SVM)  
- ✔️ Pipeline completo: EDA → Limpeza → Vetorização → Modelagem → Interpretação  
- ✔️ TF‑IDF com **10.000 features**  
- ✔️ Extração de *features* mais importantes  
- ✔️ Projeto totalmente replicável

---

---

## 🔍 1. EDA — Análise Exploratória

A análise inicial identificou:

- Dataset balanceado entre reviews positivos e negativos  
- Textos com grande variação de tamanho  
- Presença de HTML, tags `<br>`, pontuação e caracteres especiais  
- Necessidade de limpeza profunda antes da vetorização  

Durante o EDA foram criados gráficos como:

- Distribuição dos tamanhos dos textos  
- Frequência das palavras mais comuns  
- Nuvens de palavras (pos/neg)

---

## 🧹 2. Pré-processamento

As reviews passaram por:

- Remoção de HTML  
- Remoção de pontuação  
- Normalização (lowercase)  
- Remoção de múltiplos espaços  
- Tokenização opcional  

Exemplo da função de limpeza:

```python
def clean_text(text):
    text = remove_html(text)
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
```

---

## 🔠 3. Vetorização TF‑IDF

O texto limpo foi transformado em vetores usando:

```python
TfidfVectorizer(
    max_features=10000,
    stop_words="english",
    ngram_range=(1,1)
)
```

- 10k features → captura boa variedade sem explodir dimensionalidade  
- Stopwords → reduz ruído  
- Unigramas → melhor desempenho para textos curtos  

---

## 🤖 4. Modelos Treinados

| Modelo | Acurácia |
|--------|----------|
| Logistic Regression | **0.89** |
| Linear SVM | **0.89** |
| Multinomial Naive Bayes | 0.85 |

A escolha final foi entre **Logistic Regression** ou **SVM**, ambos com desempenho similar.

---

## 🔍 5. Interpretação das Features

A partir dos coeficientes da Regressão Logística, foram extraídas as palavras:

- **Mais associadas a reviews positivos**  
- **Mais associadas a reviews negativos**

Esse tipo de interpretação é essencial em projetos reais para explicar decisões do modelo.

---

## 📦 6. Como Executar

### 1 — Instale as dependências:

```bash
pip install -r requirements.txt
```

### 2 — Execute o notebook:

```bash
jupyter notebook notebooks/analise_sentimento.ipynb
```

---

## 🚀 7. Possíveis Melhorias

- Usar embeddings (Word2Vec, FastText, BERT)  
- Criar API com FastAPI para servir o modelo  
- Criar dashboard com Streamlit  
- Fine-tuning de modelos Transformers  

---

## 🧑‍💻 Autor

**Renan Croffi**  
Projeto desenvolvido para portfólio de Ciência de Dados.  

---

## 📝 Licença  
Este projeto está sob a licença MIT.
