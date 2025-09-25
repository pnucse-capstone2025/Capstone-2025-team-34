package com.example.graduateserver.controller;

import com.example.graduateserver.dto.ProblemRequest;
import com.example.graduateserver.dto.ProblemResponse;
import com.example.graduateserver.service.EnglishProblemService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/problem")
@RequiredArgsConstructor
@CrossOrigin(origins = "*")
public class EnglishProblemController {
    
    private final EnglishProblemService englishProblemService;
    
    @PostMapping
    public ResponseEntity<ProblemResponse> getAnswerOfProblem(
        @RequestBody ProblemRequest problemRequest
    ) {
        ProblemResponse response = englishProblemService.getAnswerOfProblem(problemRequest);
        return ResponseEntity.ok(response);
    }
}
