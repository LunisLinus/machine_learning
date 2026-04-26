from __future__ import annotations

import argparse
import re
from dataclasses import dataclass

try:
    from datasketch import MinHash
except ImportError as exc:
    raise SystemExit(
        "Не установлена библиотека datasketch. Установите ее командой:\n"
        "python3 -m pip install datasketch"
    ) from exc


_WORD_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)


@dataclass(frozen=True)
class SimilarityResult:
    text1_ngrams: set[str]
    text2_ngrams: set[str]
    exact_jaccard: float
    minhash_jaccard: float


def preprocess(text: str) -> list[str]:
    cleaned = _WORD_RE.sub(" ", text.lower())
    return cleaned.split()


def make_ngrams(tokens: list[str], n: int) -> set[str]:
    if n <= 0:
        raise ValueError("n должно быть положительным числом")
    if len(tokens) < n:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def exact_jaccard(set1: set[str], set2: set[str]) -> float:
    if not set1 and not set2:
        return 1.0
    return len(set1 & set2) / len(set1 | set2)


def build_minhash(items: set[str], num_perm: int) -> MinHash:
    minhash = MinHash(num_perm=num_perm)
    for item in items:
        minhash.update(item.encode("utf-8"))
    return minhash


def compare_texts(text1: str, text2: str, n: int, num_perm: int) -> SimilarityResult:
    ngrams1 = make_ngrams(preprocess(text1), n)
    ngrams2 = make_ngrams(preprocess(text2), n)

    minhash1 = build_minhash(ngrams1, num_perm)
    minhash2 = build_minhash(ngrams2, num_perm)

    return SimilarityResult(
        text1_ngrams=ngrams1,
        text2_ngrams=ngrams2,
        exact_jaccard=exact_jaccard(ngrams1, ngrams2),
        minhash_jaccard=minhash1.jaccard(minhash2),
    )


def similarity_label(score: float) -> str:
    if score >= 0.75:
        return "тексты очень похожи"
    if score >= 0.45:
        return "тексты умеренно похожи"
    if score >= 0.2:
        return "тексты слабо похожи"
    return "тексты почти не похожи"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Оценка схожести двух текстов через datasketch.MinHash."
    )
    parser.add_argument(
        "text1",
        nargs="?",
        default="Машинное обучение анализирует данные и строит модели",
        help="Первый текст",
    )
    parser.add_argument(
        "text2",
        nargs="?",
        default="Машинное обучение изучает данные и улучшает модели",
        help="Второй текст",
    )
    parser.add_argument("-n", "--ngram-size", type=int, default=1, help="Размер n-грамм")
    parser.add_argument(
        "-p",
        "--num-perm",
        type=int,
        default=128,
        help="Количество перестановок в MinHash",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = compare_texts(args.text1, args.text2, args.ngram_size, args.num_perm)

    print(f"Текст 1 n-граммы: {sorted(result.text1_ngrams)}")
    print(f"Текст 2 n-граммы: {sorted(result.text2_ngrams)}")
    print(f"Точное сходство Жаккара: {result.exact_jaccard:.3f}")
    print(f"Оценка Жаккара по datasketch.MinHash: {result.minhash_jaccard:.3f}")
    print(f"Вывод: {similarity_label(result.minhash_jaccard)}.")


if __name__ == "__main__":
    main()
