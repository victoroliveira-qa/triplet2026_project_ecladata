# TRIPLET Workshop @ ESWC 2026 - Joint Relation Extraction Between Texts and Tables

O objetivo desta tarefa é extrair automaticamente conhecimento de tabelas e textos relacionados. Para isso, criamos o ReTaT, um conjunto de dados que pode ser usado para treinar e avaliar sistemas de extração dessas relações. Este conjunto de dados é composto por pares (tabela, texto circundante) extraídos de páginas da Wikipédia e anotados manualmente com triplas de relações. O ReTaT está organizado em três subconjuntos com características distintas: domínio (negócios, telecomunicações e celebridades femininas), tamanho (de 50 a 255 pares), idioma (inglês vs. francês), tipo de relação (dados vs. propriedades de objetos), lista fechada vs. lista aberta de relações e tamanho do texto circundante (parágrafo vs. página inteira). Em seguida, avaliamos sua qualidade e adequação para a tarefa de extração conjunta de relações entre tabela e texto usando Modelos de Linguagem de Grande Porte (LLMs).

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)

## 📋 Funcionalidades

O `CorpusAnalyzer` realiza uma análise completa do corpus:

1.  **Processamento de Dados**: Lê arquivos JSON complexos e os normaliza em DataFrames (Pandas).
2.  **Análise Estrutural**: Contagem de documentos, textos, tabelas e anotações.
3.  **Análise de Conteúdo**: Estatísticas de tamanho de texto, dimensões de tabelas e tipos de colunas.
4.  **Análise Semântica (ER)**: Extração e contagem de Entidades e Relações (Predicados), incluindo verificação de IDs (Wikidata vs Custom).
5.  **Visualização Amigável**: Exibe no terminal/log uma visão unificada de Texto + Tabela + Anotações para cada documento, incluindo a origem da extração (Text vs Table).
6.  **Geração de Gráficos**: Cria 17 tipos de gráficos estatísticos para insights visuais.
7.  **Exportação**: Gera relatórios em CSV (triplas, entidades, predicados, tabelas).

## 🚀 Instalação

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/victoroliveira-qa/triplet2026_project_ecladata.git](https://github.com/victoroliveira-qa/triplet2026_project_ecladata.git)
    cd triplet2026_project_ecladata
    ```
2.  **Crie um ambiente virtual (Recomendado):**
    ```bash
    python -m venv venv
    # Windows:
    venv\Scripts\activate
    # Linux/Mac:
    source venv/bin/activate
    ```

3.  **Instale as dependências Python:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configuração:**
    O projeto já vem configurado para usar o Ollama por padrão em `src/config.py`. Nenhuma chave de API é necessária, a menos que mude para OpenAI.

---

## 📂 Estrutura do Projeto

A ferramenta espera (e cria) a seguinte estrutura de diretórios:

```text
triplet2026_project_ecladata/
│
├── main.py                 # Script principal
├── json_to_mongodb.py      # Conversor de Json para MongoDB
├── saida_visualizacao.txt  # Log detalhado da execução (gerado automaticamente)
│
├── data/                   # [ENTRADA] Coloque seu arquivo .json aqui
│   └── Corpus_Business_IRIT_ISWC-Train_Joint_(nous)_(without_Pertinence)_OK.json
│
├── csvs/                   # [SAÍDA] Arquivos CSV gerados
└── graficos/               # [SAÍDA] Imagens .png geradas