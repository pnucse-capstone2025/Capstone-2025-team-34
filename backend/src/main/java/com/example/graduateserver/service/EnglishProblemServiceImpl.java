package com.example.graduateserver.service;

import com.example.graduateserver.dto.ProblemRequest;
import com.example.graduateserver.dto.ProblemResponse;
import com.example.graduateserver.external.AiClient;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

@Service
@RequiredArgsConstructor
public class EnglishProblemServiceImpl implements EnglishProblemService {
    
    private final AiClient aiClient;
    
    @Override
    public ProblemResponse getAnswerOfProblem(ProblemRequest problemRequest) {
        ProblemResponse response = aiClient.getAnswerOfProblem(problemRequest);
        return response;
    }
}
