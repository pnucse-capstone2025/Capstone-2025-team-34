package com.example.graduateserver.external;

import com.example.graduateserver.dto.ProblemRequest;
import com.example.graduateserver.dto.ProblemResponse;
import org.springframework.http.HttpStatusCode;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

@Component
public class AiClientImpl implements AiClient {
    private final WebClient webClient;
    
    public AiClientImpl() {
        this.webClient = WebClient.builder()
            .baseUrl("http://localhost:9009")
            .defaultHeader("Content-Type", "application/json")
            .build();
    }
    
    @Override
    public ProblemResponse getAnswerOfProblem(ProblemRequest problemRequest) {
        return webClient.post()
            .uri("/mcq")
            .bodyValue(problemRequest)  // bodyValue로 JSON 직렬화 보장
            .retrieve()
            .onStatus(HttpStatusCode::isError,
                resp -> Mono.error(new RuntimeException("AI 서버 응답 에러: " + resp.statusCode())))
            .bodyToMono(ProblemResponse.class)  // JSON -> DTO
            .block();  // 동기 호출
    }
}
