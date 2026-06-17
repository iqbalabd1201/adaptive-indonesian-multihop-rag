"""
evaluate_llm_judge.py

Cleaned LLM-as-Judge evaluation artifact for the adaptive Indonesian
multi-hop RAG system.

Original workflow:
- Load full-pipeline answer generation results.
- Compare predicted answers with reference answers.
- Use an LLM judge to evaluate semantic correctness.
- Return JSON judgment with is_correct, reasoning, and confidence.
- Compute LLM-as-Judge accuracy, confidence distribution, and discrepancy
  analysis against Exact Match.

Sensitive API keys, Google Drive paths, Colab-specific commands,
manual prompts, and long output logs are removed.
"""

import argparse
import json
import os
import time
from datetime import datetime
from typing import Dict, List

from openai import OpenAI


DEFAULT_JUDGE_MODEL = "gpt-4o-mini"


def load_results(file_path: str) -> List[Dict]:
    """Load pipeline results from a JSON file."""
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, dict) and "results" in data:
        return data["results"]

    if isinstance(data, list):
        return data

    raise ValueError("Unsupported result file format.")


def build_judge_prompt(question: str, predicted_answer: str, reference_answer: str) -> str:
    """Build Indonesian LLM-as-Judge prompt."""
    return f"""
Tugas Anda adalah menentukan apakah JAWABAN PREDIKSI secara semantik benar
dibandingkan dengan JAWABAN REFERENSI.

PERTANYAAN:
{question}

JAWABAN REFERENSI:
{reference_answer}

JAWABAN PREDIKSI:
{predicted_answer}

INSTRUKSI PENILAIAN:
1. Jawaban dianggap benar jika memiliki makna atau informasi inti yang sama,
   meskipun bentuk katanya berbeda.
2. Jawaban dianggap benar jika perbedaan hanya berupa sinonim, variasi bahasa,
   variasi terjemahan, atau tambahan detail minor yang tidak mengubah makna.
3. Jawaban dianggap salah jika mengacu pada entitas yang berbeda, bertentangan
   secara faktual, tidak relevan, kosong, atau tidak menjawab pertanyaan.
4. Jika jawaban sulit dinilai, gunakan confidence "low".

Keluarkan hanya JSON dengan format berikut:
{{
  "is_correct": true,
  "reasoning": "Penjelasan singkat.",
  "confidence": "high"
}}

Nilai confidence hanya boleh salah satu dari:
- high
- medium
- low
""".strip()


def call_openai_with_retry(
    client: OpenAI,
    messages: List[Dict],
    model: str,
    max_retries: int = 3,
) -> Dict:
    """Call OpenAI API with simple exponential backoff."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                max_tokens=300,
                response_format={"type": "json_object"},
            )

            response_text = response.choices[0].message.content.strip()
            return json.loads(response_text)

        except Exception as error:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                time.sleep(wait_time)
            else:
                return {
                    "is_correct": False,
                    "reasoning": f"Judge error: {str(error)[:100]}",
                    "confidence": "low",
                }


def judge_answer(
    client: OpenAI,
    question: str,
    predicted_answer: str,
    reference_answer: str,
    model: str,
) -> Dict:
    """Judge whether predicted answer is semantically correct."""
    prompt = build_judge_prompt(
        question=question,
        predicted_answer=predicted_answer,
        reference_answer=reference_answer,
    )

    messages = [
        {
            "role": "system",
            "content": (
                "Anda adalah evaluator objektif untuk sistem question answering. "
                "Nilailah kebenaran jawaban berdasarkan makna semantik, bukan hanya "
                "kecocokan literal."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    return call_openai_with_retry(
        client=client,
        messages=messages,
        model=model,
    )


def evaluate_results(
    results: List[Dict],
    client: OpenAI,
    model: str,
    checkpoint_file: str,
    checkpoint_interval: int = 500,
) -> List[Dict]:
    """Evaluate all results with LLM-as-Judge and checkpoint progress."""
    judged_results = []
    start_index = 0

    if checkpoint_file and os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r", encoding="utf-8") as file:
            checkpoint = json.load(file)

        judged_results = checkpoint.get("judged_results", [])
        start_index = checkpoint.get("progress", len(judged_results))

    for index in range(start_index, len(results)):
        result = results[index]

        question = result.get("question", "")
        predicted_answer = result.get("predicted_answer", "")
        reference_answer = result.get("answer", result.get("reference_answer", ""))

        judgment = judge_answer(
            client=client,
            question=question,
            predicted_answer=predicted_answer,
            reference_answer=reference_answer,
            model=model,
        )

        judged_result = {
            **result,
            "llm_judgment": judgment,
            "llm_correct": bool(judgment.get("is_correct", False)),
            "llm_confidence": judgment.get("confidence", "low"),
            "original_em": result.get("em", result.get("exact_match", 0)),
            "original_f1": result.get("f1", result.get("f1_score", 0.0)),
        }

        judged_results.append(judged_result)

        if checkpoint_file and (index + 1) % checkpoint_interval == 0:
            os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)

            with open(checkpoint_file, "w", encoding="utf-8") as file:
                json.dump(
                    {
                        "progress": index + 1,
                        "judged_results": judged_results,
                    },
                    file,
                    ensure_ascii=False,
                    indent=2,
                )

            print(f"Checkpoint saved at sample {index + 1}/{len(results)}")

    return judged_results


def summarize_judgments(judged_results: List[Dict]) -> Dict:
    """Summarize LLM-as-Judge evaluation results."""
    total = len(judged_results)

    llm_correct = sum(
        1
        for result in judged_results
        if result.get("llm_correct", False)
    )

    confidence_distribution = {
        "high": 0,
        "medium": 0,
        "low": 0,
    }

    em_0_llm_1 = 0
    em_1_llm_0 = 0

    for result in judged_results:
        confidence = result.get("llm_confidence", "low")

        if confidence not in confidence_distribution:
            confidence = "low"

        confidence_distribution[confidence] += 1

        original_em = result.get("original_em", 0)
        llm_correct_flag = result.get("llm_correct", False)

        if original_em == 0 and llm_correct_flag:
            em_0_llm_1 += 1

        elif original_em == 1 and not llm_correct_flag:
            em_1_llm_0 += 1

    llm_accuracy = llm_correct / total * 100 if total else 0

    return {
        "total_samples": total,
        "llm_correct": llm_correct,
        "llm_accuracy": llm_accuracy,
        "confidence_distribution": confidence_distribution,
        "discrepancy_analysis": {
            "em_0_llm_1": em_0_llm_1,
            "em_1_llm_0": em_1_llm_0,
            "total_discrepancies": em_0_llm_1 + em_1_llm_0,
        },
    }


def save_output(output: Dict, output_file: str) -> None:
    """Save evaluation output."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)


def main() -> None:
    """Run LLM-as-Judge evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate generated answers using LLM-as-Judge."
    )

    parser.add_argument("--input", required=True, help="Path to pipeline result JSON file.")
    parser.add_argument("--output", default="outputs/llm_judge_results.json", help="Output JSON file.")
    parser.add_argument("--model", default=DEFAULT_JUDGE_MODEL, help="LLM judge model.")
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable containing the OpenAI API key.",
    )
    parser.add_argument(
        "--checkpoint-file",
        default="outputs/llm_judge_checkpoint.json",
        help="Checkpoint file for long evaluation runs.",
    )

    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env)

    if not api_key:
        raise ValueError(
            f"API key not found. Please set the {args.api_key_env} environment variable."
        )

    client = OpenAI(api_key=api_key)

    start_time = time.time()

    results = load_results(args.input)

    judged_results = evaluate_results(
        results=results,
        client=client,
        model=args.model,
        checkpoint_file=args.checkpoint_file,
    )

    summary = summarize_judgments(judged_results)

    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "judge_model": args.model,
            "input_file": args.input,
            "runtime_seconds": time.time() - start_time,
        },
        "llm_judge_metrics": summary,
        "results": judged_results,
    }

    save_output(output, args.output)

    if args.checkpoint_file and os.path.exists(args.checkpoint_file):
        os.remove(args.checkpoint_file)

    print("LLM-as-Judge evaluation complete")
    print(f"Total samples: {summary['total_samples']}")
    print(f"LLM Judge Accuracy: {summary['llm_accuracy']:.2f}%")
    print(f"High confidence: {summary['confidence_distribution']['high']}")
    print(f"Medium confidence: {summary['confidence_distribution']['medium']}")
    print(f"Low confidence: {summary['confidence_distribution']['low']}")
    print(f"Output saved to: {args.output}")


if __name__ == "__main__":
    main()
