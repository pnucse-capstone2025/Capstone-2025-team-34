package com.example.graduateserver.external;

import com.example.graduateserver.dto.ProblemRequest;
import com.example.graduateserver.dto.ProblemResponse;

public interface AiClient {
    ProblemResponse getAnswerOfProblem(ProblemRequest problemRequest);
}
