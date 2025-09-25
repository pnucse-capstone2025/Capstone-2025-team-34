package com.example.graduateserver.service;

import com.example.graduateserver.dto.ProblemRequest;
import com.example.graduateserver.dto.ProblemResponse;

public interface EnglishProblemService {
    
    ProblemResponse getAnswerOfProblem(ProblemRequest problemRequest);
}
