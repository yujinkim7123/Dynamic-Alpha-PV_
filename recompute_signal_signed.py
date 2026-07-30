"""
recompute_signal_signed.py

기존 Signal(unsigned) vs 수정된 Signal(signed)을 
모든 방법 x 모든 backbone에 대해 재계산하는 스크립트.

입력: C_matrix_*.json (raw_runs 포함된 원본 결과 파일들)
출력: 콘솔에 비교표 + signal_signed_full_results.json 저장
"""
import json
import glob

NAME_MAP = {
    'OPN': 'openness', 'CON': 'conscientiousness', 'EXT': 'extraversion',
    'AGR': 'agreeableness', 'NEU': 'neuroticism',
}
BIG5 = ['OPN', 'CON', 'EXT', 'AGR', 'NEU']


def build_C_from_raw(fname):
    """저장된 JSON에서 5x5 C-matrix(diagonal + off-diagonal 평균값)를 복원"""
    d = json.load(open(fname))
    raw = d['raw']
    C = {}
    for i in BIG5:
        C[i] = {}
        for j in BIG5:
            key = NAME_MAP[j]
            entry = raw[i]
            # 파일 스키마가 두 가지 존재: {'avg': {...}} 형태 또는 바로 {...} 형태
            C[i][j] = entry['avg'][key] if 'avg' in entry else entry[key]
    return C


def signal_unsigned(C):
    """기존 방식: Signal = Σ |C[i,i]|"""
    return sum(abs(C[i][i]) for i in BIG5)


def signal_signed(C):
    """수정 방식: Signal = Σ C[i,i]  (절댓값 제거)"""
    return sum(C[i][i] for i in BIG5)


def interference_unsigned(C):
    """Interference는 원래부터 off-diagonal 절댓값 평균이라 그대로 유지"""
    vals = [abs(C[i][j]) for i in BIG5 for j in BIG5 if i != j]
    return sum(vals) / len(vals)


def main():
    # 재계산 대상 파일들 (방법명 -> 파일경로)
    files = {
        'Llama Original':   'C_matrix_llama_task.json',
        'Llama GS(OPN)':    'C_matrix_llama_gs_opn_first.json',
        'Llama GS(CON)':    'C_matrix_llama_gs_con_first.json',
        'Llama MGS(OPN)':   'C_matrix_llama_mgs_opn_first.json',
        'Llama MGS(CON)':   'C_matrix_llama_mgs_con_first.json',
        'Llama Lowdin':     'C_matrix_llama_ortho_lowdin.json',
        'Qwen Original':    'C_matrix_qwen_task.json',
        'Qwen GS(OPN)':     'C_matrix_qwen_gs_opn_first.json',
        'Qwen GS(CON)':     'C_matrix_qwen_gs_con_first.json',
        'Qwen MGS(OPN)':    'C_matrix_qwen_mgs_opn_first.json',
        'Qwen MGS(CON)':    'C_matrix_qwen_mgs_con_first.json',
        'Qwen Lowdin':      'C_matrix_qwen_ortho_lowdin.json',
    }

    results = {}
    print(f"{'Method':16} {'S_unsigned':11} {'S_signed':9} {'SN_unsigned':12} {'SN_signed':10} {'sign_flip?'}")
    print("-" * 75)

    for name, f in files.items():
        C = build_C_from_raw(f)
        su = signal_unsigned(C)
        ss = signal_signed(C)
        iu = interference_unsigned(C)
        flipped = [i for i in BIG5 if C[i][i] < 0]  # 부호 반전된 trait 목록

        results[name] = {
            'signal_unsigned': round(su, 4),
            'signal_signed': round(ss, 4),
            'interference': round(iu, 4),
            'sn_unsigned': round(su / iu, 4),
            'sn_signed': round(ss / iu, 4),
            'flipped_diagonal_traits': flipped,
        }

        flag = ", ".join(flipped) if flipped else "-"
        print(f"{name:16} {su:11.3f} {ss:9.3f} {su/iu:12.2f} {ss/iu:10.2f} {flag}")

    with open('signal_signed_full_results.json', 'w') as fo:
        json.dump(results, fo, indent=2, ensure_ascii=False)
    print("\n저장 완료: signal_signed_full_results.json")


if __name__ == "__main__":
    main()
