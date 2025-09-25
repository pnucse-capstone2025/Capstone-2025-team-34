import type { Route } from "./+types/question";
import { useState, type FormEvent } from "react";
import { Textarea } from "../components/ui/textarea";
import { Label } from "../components/ui/label"
import { RadioGroup, RadioGroupItem } from "../components/ui/radio-group"
import { Input } from "../components/ui/input";
import { Button } from "~/components/ui/button"
import { Card, CardHeader, CardTitle, CardContent } from "../components/ui/card";
import {
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ComposedChart, Line, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar
} from "recharts";

export function meta({}: Route.MetaArgs) {
  return [
    { title: "Question Page" },
    { name: "description", content: "This is a test question page." },
  ];
}

interface McqResult {
  answer: string;
  final_probs: number[];
  model_probs: Record<string, number[]>;
  classified: Record<string, number>;
  weight: Record<string, number>;
}

// 보기 라벨 (A, B, C, D, E)
const letters = (n: number) => Array.from({ length: n }, (_, i) => String.fromCharCode(65 + i));
// 표기 맵
const ModelNameMap: Record<string, string> = {
  "gemma-finetuning": "gemma 파인튜닝",
  "gemma-STaR": "gemma 생성 해설 학습",
  "gemma-teacher-student": "gemma 외부 api 지식 증류",
  "phi-finetuning": "phi 파인튜닝",
  "phi-STaR": "phi 생성 해설 학습",
  "phi-teacher-student": "phi 외부 api 지식 증류",
};

// 내부 데이터 소스 선택 (UI 노출 없음)
// 'server' 로 설정하면 실제 서버에 전송, 'mock' 은 로컬 /mcq 액션으로 전송
const DATA_SOURCE: "server" | "mock" = "server";
const SERVER_ENDPOINT = "http://172.21.49.252:8080/api/problem";

// 전역 일관 정렬: 모델 키는 아래 순서를 우선 적용하고, 없으면 알파벳순
const MODEL_ORDER = [
  'gemma-finetuning',
  'gemma-STaR',
  'gemma-teacher-student',
  'phi-finetuning',
  'phi-STaR',
  'phi-teacher-student',
] as const;

function sortModelKeys<T extends string>(keys: T[]): T[] {
  return [...keys].sort((a, b) => {
    const indexA = MODEL_ORDER.indexOf(a as any);
    const indexB = MODEL_ORDER.indexOf(b as any);
    if (indexA !== -1 && indexB !== -1) return indexA - indexB;
    if (indexA !== -1) return -1;
    if (indexB !== -1) return 1;
    return a.localeCompare(b);
  });
}

// 1) 최종 확률 막대 그래프
function FinalProbsCard({ probs }: { probs: number[] }) {
  const data = probs.map((p, i) => ({ option: letters(probs.length)[i], prob: p }));
  return (
    <Card>
      <CardHeader><CardTitle>최종 확률 (Final Probs)</CardTitle></CardHeader>
      <CardContent className="h-72">
        <ResponsiveContainer>
          <BarChart data={data} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="option" />
            <YAxis domain={[0, 1]} />
            <Tooltip formatter={(v: any) => Number(v).toFixed(4)} />
            <Bar dataKey="prob" name="p(option)" radius={[6, 6, 0, 0]} fill="#ff8822" />
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}

// 2) 모델별 확률: 그룹 막대(모델 × 보기)
function ModelProbsCard({ model }: { model: Record<string, number[]> }) {
  const modelNames = sortModelKeys(Object.keys(model));
  const L = modelNames.length ? model[modelNames[0]].length : 0;
  const cols = letters(L);
  const data = cols.map((opt, idx) => {
    const row: Record<string, any> = { option: opt };
    for (const m of modelNames) row[m] = model[m][idx] ?? 0;
    return row;
  });
  const colorMap = ["#8884d8", "#82ca9d", "#ffc658", "#ff8042", "#8dd1e1", "#a4de6c"];

  return (
    <Card>
      <CardHeader><CardTitle>모델별 확률 비교</CardTitle></CardHeader>
      <CardContent className="h-96">
        <ResponsiveContainer>
          <ComposedChart data={data} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="option" />
            <YAxis domain={[0, 1]} />
            <Tooltip formatter={(v: any) => Number(v).toFixed(4)} />
            <Legend />
            {modelNames.map((m, index) => (
              <Bar key={m} dataKey={m} stackId={undefined} name={ModelNameMap[m]} radius={[6,6,0,0]} fill={colorMap[index % colorMap.length]} />
            ))}
          </ComposedChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}

// (옵션) 모델별 확률을 “각 모델의 분포 모양”으로 보고 싶으면 레이더도 유용
function ModelProbsRadar({ model }: { model: Record<string, number[]> }) {
  const modelNames = sortModelKeys(Object.keys(model));
  const L = modelNames.length ? model[modelNames[0]].length : 0;
  const cols = letters(L);
  const data = cols.map((opt, idx) => {
    const row: Record<string, any> = { option: opt };
    for (const m of modelNames) row[m] = model[m][idx] ?? 0;
    return row;
  });

  const colorMap = ["#8884d8", "#82ca9d", "#ffc658", "#ff8042", "#8dd1e1", "#a4de6c"];

  return (
    <Card>
      <CardHeader><CardTitle>모델별 분포 레이더</CardTitle></CardHeader>
      <CardContent className="h-96">
        <ResponsiveContainer>
          <RadarChart data={data}>
            <PolarGrid />
            <PolarAngleAxis dataKey="option" />
            <PolarRadiusAxis domain={[0, 1]} />
            {modelNames.map((m, index) => (
              <Radar key={m} name={ModelNameMap[m]} dataKey={m} stroke={colorMap[index % colorMap.length]} fill={colorMap[index % colorMap.length]} fillOpacity={0.6} />
            ))}
            <Legend />
            <Tooltip formatter={(v: any) => Number(v).toFixed(4)} />
          </RadarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}

// 3) 모델 가중치: 수평 바
function WeightsCard({ weight }: { weight: Record<string, number> }) {
  const data = Object.entries(weight).map(([m, w]) => ({ model: m, weight: w }));
  // 모델 순서를 전역 기준에 맞춰 고정 (가중치 차트도 다른 차트와 동일한 모델 순서 유지)
  const sortedData = data.sort((a, b) => {
    const indexA = MODEL_ORDER.indexOf(a.model as any);
    const indexB = MODEL_ORDER.indexOf(b.model as any);
    if (indexA !== -1 && indexB !== -1) return indexA - indexB;
    if (indexA !== -1) return -1;
    if (indexB !== -1) return 1;
    return a.model.localeCompare(b.model);
  });

  return (
    <Card>
      <CardHeader><CardTitle>모델 가중치 (Weight)</CardTitle></CardHeader>
      <CardContent className="h-120">
        <ResponsiveContainer>
          <BarChart data={sortedData} margin={{ top: 8, right: 16, bottom: 8, left: 16 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis textAnchor="end" angle={-45} dataKey="model" tickFormatter={(value) => ModelNameMap[value] || value} interval={0} height={150} />
            <YAxis domain={[0, "auto"]} />
            <Tooltip formatter={(v: any) => Number(v).toFixed(4)} />
            <Bar dataKey="weight" name="w(model)" radius={[6, 6, 0, 0]} fill="#ff8822" />
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}

// 4) (선택) 분류기 점수: 막대
function ClassifierCard({ classifier }: { classifier: Record<string, number> }) {
  const data = Object.entries(classifier).map(([k, v]) => ({ label: k, value: v }));
  // 안정적인 표시를 위해 라벨 알파벳순 정렬
  data.sort((a, b) => a.label.localeCompare(b.label));
  return (
    <Card>
      <CardHeader><CardTitle>분류기 확률</CardTitle></CardHeader>
      <CardContent className="h-72">
        <ResponsiveContainer>
          <BarChart data={data} margin={{ top: 8, right: 16, bottom: 32, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="label" angle={-20} textAnchor="end" interval={0} height={48} />
            <YAxis />
            <Tooltip formatter={(v: any) => Number(v).toFixed(4)} />
            <Bar dataKey="value" name="확률" radius={[6, 6, 0, 0]} fill="#ff8822" />
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}

export default function Question() {
  const [answerType, setAnswerType] = useState("4-answers");
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<McqResult | null>(null);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setIsLoading(true);
    setResult(null);
    const formData = new FormData(event.currentTarget);

    const options = [];
    const numAnswers = answerType === "5-answers" ? 5 : 4;
    for (let i = 1; i <= numAnswers; i++) {
      options.push(formData.get(`option-${i}`) as string);
    }

    const submission = {
      article: formData.get("article") as string,
      question: formData.get("question") as string,

      options: options,
    };

    try {
      const endpoint = DATA_SOURCE === "mock" ? "/mcq" : SERVER_ENDPOINT;
      const response = await fetch(endpoint, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(submission),
      });

      if (response.ok) {
        const data: McqResult = await response.json();
        console.log("Received response:", data);
        setResult(data);
      } else {
        alert(`제출에 실패했습니다: ${response.statusText}`);
        setResult(null);
      }
    } catch (error) {
      console.error("An error occurred during submission:", error);
      alert("제출 중 오류가 발생했습니다.");
      setResult(null);
    } finally {
      setIsLoading(false);
    }
  }

  

  return (
    <div className="flex gap-8 p-8">
      {/* Left Column: Input Form */}
      <div className="w-1/2">
        <h1 className="text-2xl font-bold mb-4">문제 입력</h1>
        <p className="mb-6">지문, 문제, 보기를 입력하고 제출 버튼을 누르세요.</p>

        <form className="space-y-6" onSubmit={handleSubmit}>
          <div className="grid w-full gap-1.5">
            <Label htmlFor="article" className="text-lg font-bold">지문</Label>
            <Textarea
              id="article"
              name="article"
              placeholder="지문을 입력해주세요."
            />
          </div>
          <div className="grid w-full gap-1.5">
            <Label htmlFor="question" className="text-lg font-bold">문제</Label>
            <Textarea
              id="question"
              name="question"
              placeholder="문제를 입력해주세요."
            />
          </div>
          <div>
            <h2 className="text-lg font-bold mb-2">보기</h2>
            <div className="grid w-full gap-1.5 mb-4">
              <Label>정답 유형</Label>
              <RadioGroup
                value={answerType}
                onValueChange={setAnswerType}
                className="mt-2 flex flex-col space-y-1"
              >
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="4-answers" id="4-answers" />
                  <Label htmlFor="4-answers">4지선다</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="5-answers" id="5-answers" />
                  <Label htmlFor="5-answers">5지선다</Label>
                </div>
              </RadioGroup>
            </div>
            <div className="space-y-2">
              <div className="flex w-full items-center gap-4">
                <Label htmlFor="option-1" className="w-16 text-right">
                  보기 1
                </Label>
                <Input
                  id="option-1"
                  name="option-1"
                  placeholder="첫 번째 선택지 내용을 입력하세요."
                />
              </div>
              <div className="flex w-full items-center gap-4">
                <Label htmlFor="option-2" className="w-16 text-right">
                  보기 2
                </Label>
                <Input
                  id="option-2"
                  name="option-2"
                  placeholder="두 번째 선택지 내용을 입력하세요."
                />
              </div>
              <div className="flex w-full items-center gap-4">
                <Label htmlFor="option-3" className="w-16 text-right">
                  보기 3
                </Label>
                <Input
                  id="option-3"
                  name="option-3"
                  placeholder="세 번째 선택지 내용을 입력하세요."
                />
              </div>
              <div className="flex w-full items-center gap-4">
                <Label htmlFor="option-4" className="w-16 text-right">
                  보기 4
                </Label>
                <Input
                  id="option-4"
                  name="option-4"
                  placeholder="네 번째 선택지 내용을 입력하세요."
                />
              </div>
              {answerType === "5-answers" && (
                <div className="flex w-full items-center gap-4">
                  <Label htmlFor="option-5" className="w-16 text-right">
                    보기 5
                  </Label>
                  <Input
                    id="option-5"
                    name="option-5"
                    placeholder="다섯 번째 선택지 내용을 입력하세요."
                  />
                </div>
              )}
            </div>
          </div>
          <div>
            <Button type="submit" className="w-full" disabled={isLoading}>
              {isLoading ? "제출 중..." : "제출"}
            </Button>
          </div>
        </form>
      </div>

      {/* Right Column: Result */}
      <div className="w-1/2 border-l pl-8">
        <h1 className="text-2xl font-bold mb-4">풀이 결과</h1>
        <div className="p-4 bg-gray-100 rounded-lg min-h-[200px]">
          {isLoading && <p>결과를 기다리는 중입니다...</p>}
          {result ? (
            <div className="space-y-6">
              <div className="space-y-6">
                <h3 className="text-lg font-bold mb-2">정답</h3>
                <p className="text-gray-700 mb-4">{result.answer}</p>
              </div>
              <div className="space-y-6">
                <ClassifierCard classifier={result.classified} />
                <WeightsCard weight={result.weight} />
                <ModelProbsCard model={result.model_probs} />
                <ModelProbsRadar model={result.model_probs} />
                <FinalProbsCard probs={result.final_probs} />
              </div>
            </div>
          ) : (
            !isLoading && <p>제출하면 결과가 여기에 표시됩니다.</p>
          )}
        </div>
      </div>
    </div>
  );
}
