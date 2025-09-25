package com.example.graduateserver.dto;

import com.fasterxml.jackson.databind.PropertyNamingStrategies;
import com.fasterxml.jackson.databind.annotation.JsonNaming;
import java.util.List;
import java.util.Map;
import lombok.Getter;
import lombok.RequiredArgsConstructor;

@RequiredArgsConstructor
@Getter
@JsonNaming(PropertyNamingStrategies.SnakeCaseStrategy.class)
public class ProblemResponse {
    private final String answer;
    private final List<Double> finalProbs;
    private final Map<String, Double> classified;
    private final Map<String, List<Double>> modelProbs;
    private final Map<String, Double> weight;
}
