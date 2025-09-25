package com.example.graduateserver.dto;

import java.util.List;
import lombok.Getter;
import lombok.RequiredArgsConstructor;

@RequiredArgsConstructor
@Getter
public class ProblemRequest {
    private final String article;
    private final String question;
    private final List<String> options;
}
