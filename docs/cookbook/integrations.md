# Integration Examples (Copy-Paste)

This page shows copy-paste integration patterns for SDK-first preprocessing before downstream model calls.

## OpenAI: preprocess before model call

```python
from openai import OpenAI

from open_vernacular_ai_kit import render_codemix

client = OpenAI()

def answer_user(text: str) -> str:
    cleaned = render_codemix(text, language="gu", translit_mode="sentence")

    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {
                "role": "system",
                "content": "You are a customer-support assistant. Keep answers concise.",
            },
            {
                "role": "user",
                "content": cleaned,
            },
        ],
    )
    return response.output_text

print(answer_user("maru order status shu chhe?"))
```

## LangChain: add preprocessing in chain

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

from open_vernacular_ai_kit import render_codemix

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

preprocess = RunnableLambda(
    lambda x: {
        **x,
        "cleaned_text": render_codemix(
            x["text"],
            language="gu",
            translit_mode="sentence",
        ),
    }
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Extract user intent in one line."),
        ("human", "Input: {cleaned_text}"),
    ]
)

chain = preprocess | prompt | llm | StrOutputParser()
print(chain.invoke({"text": "maru payment nathi thayu"}))
```

## RAG: preprocess both documents and queries

```python
from open_vernacular_ai_kit import (
    RagDocument,
    RagIndex,
    load_vernacular_facts_tiny,
    render_codemix,
)


def keyword_embed(texts: list[str]) -> list[list[float]]:
    keys = ["gujarati", "hindi", "tamil", "kannada", "bengali", "marathi"]
    return [[1.0 if k in (t or "").lower() else 0.0 for k in keys] for t in texts]


ds = load_vernacular_facts_tiny()

# Preprocess corpus once before embedding.
normalized_docs = [
    RagDocument(
        doc_id=doc.doc_id,
        text=render_codemix(doc.text, language="gu", translit_mode="sentence"),
        meta=doc.meta,
    )
    for doc in ds.docs
]

index = RagIndex.build(docs=normalized_docs, embed_texts=keyword_embed, embedding_model="keywords")

query = render_codemix("gujarat ma support ma kai language use thay chhe?", language="gu")
hits = index.search(query=query, embed_texts=keyword_embed, topk=3)
print([h.doc_id for h in hits])
```

## Practical defaults

Use these defaults for most app integrations:

- `language="gu"`
- `translit_mode="sentence"`
- `preserve_case=True`
- `preserve_numbers=True`
