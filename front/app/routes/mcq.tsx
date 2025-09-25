import { type ActionFunctionArgs } from "react-router";
import crypto from "node:crypto";

// ──────────────────────────────────────────
// 타입 (기존과 동일)
// ──────────────────────────────────────────
interface MCQRequest {
  article: string;
  question: string;
  options: string[];
}

interface MCQResponse {
  answer: string;
  final_probs: number[];
  model_probs: Record<string, number[]>;
  classified: Record<string, number>;
  weight: Record<string, number>;
}

// ──────────────────────────────────────────
// 유틸리티 함수 (기존과 거의 동일)
// ──────────────────────────────────────────

// 간단 검증
function validateMCQ(body: any): { ok: true; data: MCQRequest } | { ok: false; msg: string } {
  if (typeof body !== "object" || body === null) return { ok: false, msg: "요청 바디가 비어 있거나 객체가 아닙니다." };

  const { article, question, options } = body as Partial<MCQRequest>;

  if (typeof article !== "string" || article.trim().length === 0)
    return { ok: false, msg: "article: 빈 문자열은 허용되지 않습니다." };

  if (typeof question !== "string" || question.trim().length === 0)
    return { ok: false, msg: "question: 빈 문자열은 허용되지 않습니다." };

  if (!Array.isArray(options) || options.length < 2)
    return { ok: false, msg: "options: 2개 이상이어야 합니다." };

  for (let i = 0; i < options.length; i++) {
    const v = options[i];
    if (typeof v !== "string" || v.trim().length === 0) {
      return { ok: false, msg: `options[${i}]: 공백이 아닌 문자열이어야 합니다.` };
    }
  }

  return { ok: true, data: { article, question, options } as MCQRequest };
}

// 더미 응답 생성
function pickDeterministicOption(article: string, question: string, options: string[]): { index: number; value: string } {
  const seed = `${article}||${question}`;
  const hash = crypto.createHash("sha256").update(seed, "utf8").digest();
  const idx = hash[0] % options.length;
  return { index: idx, value: options[idx].trim() };
}

// 시드 기반 난수 (0~1)
function seededRandom(seed: string): number {
  const h = crypto.createHash("sha256").update(seed, "utf8").digest();
  // 32-bit 정수로 변환 후 0~1로 스케일
  const n = h.readUInt32BE(0);
  return n / 0xffffffff;
}

function normalize(arr: number[]): number[] {
  const s = arr.reduce((a, b) => a + b, 0) || 1;
  return arr.map((v) => v / s);
}

const MODEL_KEYS = [
  "gemma-finetuning",
  "gemma-STaR",
  "gemma-teacher-student",
  "phi-finetuning",
  "phi-STaR",
  "phi-teacher-student",
] as const;

// ──────────────────────────────────────────
// Remix의 Action 함수 (POST, PUT, DELETE 등 처리)
// ──────────────────────────────────────────
export async function action({ request }: ActionFunctionArgs) {
  // Remix는 Content-Type이 application/json 또는 application/x-www-form-urlencoded 일 때
  // 자동으로 body를 파싱해줍니다.
  const body = await request.json();
  const valid = validateMCQ(body);

  if (!valid.ok) {
    // 유효성 검사 실패 시, 에러 메시지와 422 상태 코드를 담은 JSON 응답을 반환합니다.
    return Response.json({ message: valid.msg }, { status: 422 });
  }

  const { article, question, options } = valid.data;
  const chosen = pickDeterministicOption(article, question, options);

  // 최종 확률(옵션별) 생성: 선택된 인덱스에 약간의 보너스를 주고 정규화
  const baseProbs = Array.from({ length: options.length }, (_, i) => 0.1 + 0.4 * seededRandom(`${article}|${question}|base|${i}`));
  baseProbs[chosen.index] += 0.8; // 선택지에 가중치 부여
  const final_probs = normalize(baseProbs);

  // 모델별 확률: 최종 확률에 소량의 시드 노이즈를 더해 각 모델 분포 생성 후 정규화
  const model_probs: Record<string, number[]> = {};
  for (const mk of MODEL_KEYS) {
    const arr = final_probs.map((p, i) => {
      const noise = (seededRandom(`${article}|${question}|${mk}|${i}`) - 0.5) * 0.15; // ~[-0.075, 0.075]
      return Math.max(0.0001, p + noise);
    });
    model_probs[mk] = normalize(arr);
  }

  // 분류기 점수: 임의의 5개 라벨을 0~1로 정규화
  const classifierLabels = ["knowledge", "reasoning", "calculation", "language", "world"]; 
  const classifierRaw = classifierLabels.map((k) => 0.2 + seededRandom(`${article}|${question}|clf|${k}`));
  const classifierSum = classifierRaw.reduce((a, b) => a + b, 0) || 1;
  const classified: Record<string, number> = Object.fromEntries(
    classifierLabels.map((k, i) => [k, classifierRaw[i] / classifierSum])
  );

  // 모델 가중치: 모델별 시드 값 정규화
  const weightRaw = MODEL_KEYS.map((mk) => 0.5 + seededRandom(`${article}|${question}|w|${mk}`));
  const weightSum = weightRaw.reduce((a, b) => a + b, 0) || 1;
  const weight: Record<string, number> = Object.fromEntries(
    MODEL_KEYS.map((mk, i) => [mk, weightRaw[i] / weightSum])
  );

  const response: MCQResponse = {
    answer: String.fromCharCode(65 + chosen.index), // A, B, C, ...
    final_probs,
    model_probs,
    classified,
    weight,
  };

  // 성공 시, 결과와 200 상태 코드를 담은 JSON 응답을 반환합니다.
  return Response.json(response, { status: 200 });
}
