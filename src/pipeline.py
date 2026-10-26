from src import config, generation


def answer_question(retriever, question, k=config.TOP_K, mode="hybrid",
                    use_llm=True, api_key=None):
    passages = retriever.search(question, k=k, mode=mode)
    answer = generation.generate_answer(question, passages, api_key=api_key) if use_llm else None
    sources = [f"[{n}] {p['report']}, PDF page {p['page']}" for n, p in enumerate(passages, 1)]
    return {"question": question, "answer": answer, "passages": passages, "sources": sources}


def show(result):
    print("Q:", result["question"], "\n")
    print(result["answer"], "\n")
    print("Sources:")
    for s in result["sources"]:
        print("  ", s)
