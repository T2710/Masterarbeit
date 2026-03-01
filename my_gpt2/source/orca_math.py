import re
import json
import argparse
import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Sequence

import torch
import tiktoken
from datasets import load_dataset
from tqdm import tqdm


NUMBER_PAT = re.compile(
    r'''
    (?<![A-Za-z0-9_])
    -?
    (?:
        \d+\.\d+      |
        \d+/\d+       |
        \d[\d,]*
    )
    (?![A-Za-z0-9_])
    ''',
    re.VERBOSE,
)


def _encode(enc, text: str) -> List[int]:
    return enc.encode_ordinary(text) if text else []


def clean_numeric_text(text: str) -> str:
    text = text.strip().rstrip('.。')
    text = text.replace(',', '')
    return text.strip()


def split_sentences(text: str) -> List[str]:
    text = text.replace('\r', '\n')
    chunks = []
    for part in text.split('\n'):
        part = part.strip()
        if not part:
            continue
        pieces = re.split(r'(?<=[\.\!\?])\s+', part)
        for p in pieces:
            p = p.strip()
            if p:
                chunks.append(p)
    return chunks


def extract_last_number(text: str) -> Optional[str]:
    matches = NUMBER_PAT.findall(text)
    if not matches:
        return None
    return clean_numeric_text(matches[-1])


def extract_final_answer(raw_answer: str) -> str:
    if raw_answer is None:
        return ''

    text = raw_answer.strip()
    if not text:
        return ''

    explicit_patterns = [
        r'(?is)final answer\s*[:\-]\s*([^\n\r]+)',
        r'(?is)the answer is\s+([^\n\r]+)',
        r'(?is)answer\s*[:\-]\s*([^\n\r]+)',
    ]
    for pat in explicit_patterns:
        m = re.search(pat, text)
        if m:
            candidate = m.group(1).strip()
            num = extract_last_number(candidate)
            if num:
                return num
            return candidate.rstrip('.').strip()

    sentences = split_sentences(text)
    conclusion_prefixes = (
        'therefore', 'thus', 'so', 'hence', 'in conclusion',
        'therefore,', 'thus,', 'so,', 'hence,'
    )
    for sent in reversed(sentences):
        low = sent.lower().strip()
        if low.startswith(conclusion_prefixes):
            num = extract_last_number(sent)
            if num:
                return num

    for sent in reversed(sentences):
        num = extract_last_number(sent)
        if num:
            return num

    num = extract_last_number(text)
    if num:
        return num

    return sentences[-1].rstrip('.').strip() if sentences else ''


def normalize_reasoning(text: str) -> str:
    return (text or '').strip()


def pick_question_and_answer(example: Dict[str, Any]) -> Tuple[str, str]:
    question_keys = ['question', 'problem', 'instruction', 'input']
    answer_keys = ['answer', 'output', 'response', 'solution']

    q = ''
    a = ''

    for k in question_keys:
        v = example.get(k, None)
        if isinstance(v, str) and v.strip():
            q = v.strip()
            break

    for k in answer_keys:
        v = example.get(k, None)
        if isinstance(v, str) and v.strip():
            a = v.strip()
            break

    return q, a


def build_prompt_and_target(question: str, final_answer: str, reasoning: str, mode: str) -> Tuple[str, str]:
    question = question.strip()
    final_answer = final_answer.strip()
    reasoning = normalize_reasoning(reasoning)

    if mode == 'direct':
        prompt = (
            'Question:\n'
            f'{question}\n\n'
            'Final Answer:\n'
        )
        target = final_answer
        return prompt, target

    if mode == 'cot':
        prompt = (
            'Question:\n'
            f'{question}\n\n'
            'Reasoning:\n'
        )
        if reasoning:
            target = reasoning + '\n\nFinal Answer:\n' + final_answer
        else:
            target = 'Final Answer:\n' + final_answer
        return prompt, target

    raise ValueError(f'Unsupported mode: {mode}')


def tokenize_example(
    question: str,
    raw_answer: str,
    enc,
    eos_id: int,
    mode: str,
    supervise_eos: bool = True,
) -> Optional[Dict[str, Any]]:
    final_answer = extract_final_answer(raw_answer)
    if not final_answer:
        return None

    reasoning = raw_answer.strip()
    prompt_text, target_text = build_prompt_and_target(question, final_answer, reasoning, mode)

    prompt_ids = _encode(enc, prompt_text)
    target_ids = _encode(enc, target_text)

    if len(prompt_ids) == 0 or len(target_ids) == 0:
        return None

    ids: List[int] = [eos_id]
    mask: List[int] = [0]

    ids.extend(prompt_ids)
    mask.extend([0] * len(prompt_ids))

    ids.extend(target_ids)
    mask.extend([1] * len(target_ids))

    ids.append(eos_id)
    mask.append(1 if supervise_eos else 0)

    return {
        'ids': ids,
        'mask': mask,
        'question': question,
        'raw_answer': raw_answer,
        'reasoning': reasoning,
        'final_answer': final_answer,
        'prompt_text': prompt_text,
        'target_text': target_text,
        'full_text': prompt_text + target_text,
    }


def save_shard(buffer: List[Dict[str, List[int]]], out_dir: Path, shard_idx: int, prefix: str) -> Path:
    out_path = out_dir / f'{prefix}_{shard_idx:06d}.pt'
    torch.save(buffer, out_path)
    return out_path


def maybe_overwrite_dir(out_dir: Path, prefix: str, overwrite: bool):
    out_dir.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        return
    for p in out_dir.glob(f'{prefix}_*.pt'):
        p.unlink(missing_ok=True)
    for aux in ['meta.json', 'preview.txt', 'indices.json']:
        ap = out_dir / aux
        if ap.exists():
            ap.unlink()


def make_train_val_indices(num_examples: int, val_fraction: float, seed: int) -> Tuple[List[int], List[int]]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError('val_fraction must be between 0 and 1')
    indices = list(range(num_examples))
    rng = random.Random(seed)
    rng.shuffle(indices)
    n_val = max(1, int(round(num_examples * val_fraction)))
    val_indices = sorted(indices[:n_val])
    train_indices = sorted(indices[n_val:])
    return train_indices, val_indices


def load_orca_math(repo: str, split: str):
    return load_dataset(repo, split=split)


def build_mode_split(
    ds,
    indices: Sequence[int],
    out_dir: Path,
    prefix: str,
    mode: str,
    shard_size_examples: int,
    max_seq_len: int,
    supervise_eos: bool,
    overwrite: bool,
    do_preview: bool,
    repo: str,
    hf_split: str,
    split_name: str,
    seed: int,
):
    maybe_overwrite_dir(out_dir, prefix, overwrite)

    enc = tiktoken.get_encoding('gpt2')
    eos_id = enc._special_tokens['<|endoftext|>']

    buffer: List[Dict[str, List[int]]] = []
    shard_idx = 0
    n_written = 0
    n_skipped_empty = 0
    n_skipped_long = 0
    total_tokens = 0
    total_supervised_tokens = 0
    preview_items = []

    pbar = tqdm(indices, desc=f'Build {split_name}/{mode}')
    for i in pbar:
        ex = ds[i]
        question, raw_answer = pick_question_and_answer(ex)

        if not question or not raw_answer:
            n_skipped_empty += 1
            continue

        rec = tokenize_example(
            question=question,
            raw_answer=raw_answer,
            enc=enc,
            eos_id=eos_id,
            mode=mode,
            supervise_eos=supervise_eos,
        )

        if rec is None:
            n_skipped_empty += 1
            continue

        if len(rec['ids']) > max_seq_len:
            n_skipped_long += 1
            continue

        buffer.append({'ids': rec['ids'], 'mask': rec['mask']})
        n_written += 1
        total_tokens += len(rec['ids'])
        total_supervised_tokens += sum(rec['mask'])

        if do_preview and len(preview_items) < 5:
            preview_items.append(
                '=' * 100
                + f'\nMODE: {mode} | SPLIT: {split_name} | INDEX: {i}\n'
                + '-' * 100
                + '\nSEQUENCE WRITTEN TO SHARD:\n'
                + rec['full_text']
                + '\n'
            )

        if len(buffer) >= shard_size_examples:
            path = save_shard(buffer, out_dir, shard_idx, prefix)
            buffer = []
            shard_idx += 1
            pbar.set_postfix_str(f'written={n_written} last={path.name}')

    if buffer:
        path = save_shard(buffer, out_dir, shard_idx, prefix)
        pbar.set_postfix_str(f'written={n_written} last={path.name}')

    pbar.close()

    with open(out_dir / 'indices.json', 'w', encoding='utf-8') as f:
        json.dump(
            {
                'split_name': split_name,
                'mode': mode,
                'num_indices': len(indices),
                'indices': list(indices),
            },
            f,
            indent=2,
        )

    meta = {
        'repo': repo,
        'hf_split': hf_split,
        'local_split': split_name,
        'mode': mode,
        'num_examples_requested': len(indices),
        'written_examples': n_written,
        'skipped_empty_or_unparsed': n_skipped_empty,
        'skipped_too_long': n_skipped_long,
        'shard_size_examples': shard_size_examples,
        'tokenizer': 'tiktoken:gpt2',
        'eos_supervised': supervise_eos,
        'max_seq_len': max_seq_len,
        'seed': seed,
        'avg_total_tokens': (total_tokens / n_written) if n_written else 0.0,
        'avg_supervised_tokens': (total_supervised_tokens / n_written) if n_written else 0.0,
        'sequence_format': {
            'direct': 'Question:\\n<task>\\n\\nFinal Answer:\\n<answer>',
            'cot': 'Question:\\n<task>\\n\\nReasoning:\\n<full solution>\\n\\nFinal Answer:\\n<answer>',
        },
    }
    with open(out_dir / 'meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    if do_preview and preview_items:
        with open(out_dir / 'preview.txt', 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(preview_items))

    print(
        f'Finished {split_name}/{mode}: written={n_written}, '
        f'skipped_empty={n_skipped_empty}, skipped_too_long={n_skipped_long}, '
        f'avg_tokens={(total_tokens / n_written) if n_written else 0:.1f}, '
        f'avg_supervised={(total_supervised_tokens / n_written) if n_written else 0:.1f}'
    )


def main():
    ap = argparse.ArgumentParser(description='Build Orca Math direct/CoT SFT datasets.')
    ap.add_argument('--out_dir', type=str, default='my_gpt2/source/datasets/orca_math')
    ap.add_argument('--repo', type=str, default='microsoft/orca-math-word-problems-200k')
    ap.add_argument('--split', type=str, default='train', choices=['train', 'test'])
    ap.add_argument('--num_examples', type=int, default=100000, help='-1 = all examples')
    ap.add_argument('--shard_size_examples', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max_seq_len', type=int, default=1024)
    ap.add_argument('--mode', type=str, default='both', choices=['direct', 'cot', 'both'])
    ap.add_argument('--supervise_eos', action='store_true')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--no_preview', action='store_true')
    ap.add_argument('--make_val_from_train', action='store_true',
                    help='If set, take HF train split and split into train/val locally.')
    ap.add_argument('--val_fraction', type=float, default=0.1)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    do_preview = not args.no_preview

    ds = load_orca_math(args.repo, split='train' if args.make_val_from_train else args.split)
    total_avail = len(ds)
    total_requested = total_avail if args.num_examples == -1 else min(args.num_examples, total_avail)

    if args.make_val_from_train:
        train_indices, val_indices = make_train_val_indices(total_requested, args.val_fraction, args.seed)
        split_map = [
            ('train', train_indices),
            ('val', val_indices),
        ]
    else:
        split_map = [(args.split, list(range(total_requested)))]

    modes = ['direct', 'cot'] if args.mode == 'both' else [args.mode]

    print('Building Orca Math datasets:')
    print(f'repo={args.repo}')
    print(f'modes={modes}')
    print(f'total_requested={total_requested}')
    if args.make_val_from_train:
        print(f'train_examples={len(split_map[0][1])} val_examples={len(split_map[1][1])}')
    print(f'output_root={out_dir}')

    for mode in modes:
        for split_name, indices in split_map:
            local_split = f'{split_name}_{mode}' if args.make_val_from_train else split_name
            build_mode_split(
                ds=ds,
                indices=indices,
                out_dir=out_dir / local_split,
                prefix=f'orca_math_{mode}',
                mode=mode,
                shard_size_examples=args.shard_size_examples,
                max_seq_len=args.max_seq_len,
                supervise_eos=args.supervise_eos,
                overwrite=args.overwrite,
                do_preview=do_preview,
                repo=args.repo,
                hf_split='train' if args.make_val_from_train else args.split,
                split_name=split_name,
                seed=args.seed,
            )


if __name__ == '__main__':
    main()
